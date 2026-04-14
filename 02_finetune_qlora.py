"""
02_finetune_qlora.py
--------------------
Script do Lab 7 (José Lucas - estudante).

Objetivo: fazer fine-tuning do modelo Llama-2-7b-chat-hf no dataset gerado
pelo script 01, usando a técnica QLoRA (quantização 4-bit + adaptadores
LoRA). Assim conseguimos treinar um modelo de 7B de parâmetros em uma
GPU de consumidor.

Ideia geral do pipeline:
  1. Carrega o modelo base em 4 bits (NF4) -> ocupa ~4x menos VRAM
  2. Prepara o modelo para treino com k-bit (bitsandbytes)
  3. Injeta adaptadores LoRA apenas nas camadas q_proj e v_proj da atenção
  4. Formata o dataset no template de chat do Llama-2 ([INST] ... [/INST])
  5. Treina com o SFTTrainer da biblioteca trl
  6. Salva SÓ o adapter LoRA em ./lora-adapter (poucos MB)
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# Modelo base (versão do Llama-2-7b-chat no HF que não exige login)
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"

# Pasta onde o adapter final será salvo
OUTPUT_DIR = "./lora-adapter"


def parse_args():
    """Permite passar caminhos de treino/teste pela linha de comando."""
    p = argparse.ArgumentParser(description="Fine-tuning QLoRA - Lab 7")
    p.add_argument("--train_file", default="dataset/train.jsonl",
                   help="Arquivo JSONL com os exemplos de treino")
    p.add_argument("--test_file", default="dataset/test.jsonl",
                   help="Arquivo JSONL com os exemplos de teste")
    return p.parse_args()


def format_example(example):
    """
    Converte {"prompt": ..., "response": ...} no template de chat do
    Llama-2. O modelo foi treinado para entender [INST] ... [/INST].
    """
    example["text"] = (
        f"<s>[INST] {example['prompt'].strip()} [/INST] "
        f"{example['response'].strip()} </s>"
    )
    return example


def main():
    args = parse_args()

    # --- Checagem de GPU ---------------------------------------------------
    # QLoRA depende de bitsandbytes (CUDA). Se não tiver GPU, avisamos —
    # o script ainda tenta rodar, mas o esperado é GPU.
    has_gpu = torch.cuda.is_available()
    print(f"CUDA disponível: {has_gpu}")
    if not has_gpu:
        print(
            "AVISO: sem GPU detectada. QLoRA foi projetado para rodar em "
            "GPU NVIDIA com CUDA. Em CPU, o treino será lento (ou falhar "
            "dependendo da versão do bitsandbytes)."
        )

    # --- Quantização 4-bit (QLoRA) -----------------------------------------
    # load_in_4bit=True        -> carrega os pesos em 4 bits (~4x menos VRAM)
    # bnb_4bit_quant_type="nf4"-> NormalFloat4, ótimo para pesos distribuídos
    #                             aproximadamente como uma normal
    # bnb_4bit_compute_dtype=float16 -> contas intermediárias em fp16 (rápido)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # O Llama-2 não tem pad_token por padrão; usamos o EOS como pad.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Modelo base (quantizado em 4 bits) --------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if has_gpu else None,
        device_map="auto" if has_gpu else None,
        trust_remote_code=True,
    )
    # Desabilita cache de KV durante o treino (incompatível com checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if has_gpu:
        # Habilita gradientes nas camadas certas e ajusta LayerNorms para fp32
        model = prepare_model_for_kbit_training(model)

    # --- Configuração LoRA -------------------------------------------------
    # task_type=CAUSAL_LM          -> modelagem de linguagem autoregressiva
    # r=64                         -> rank dos adaptadores (quanto maior, mais
    #                                 capacidade de aprender, porém mais VRAM)
    # lora_alpha=16                -> fator de escala dos updates (alpha/r)
    # lora_dropout=0.1             -> dropout para regularização
    # bias="none"                  -> não treina biases
    # target_modules=["q_proj","v_proj"] -> injeta LoRA só nas projeções Q e V
    #                                       da atenção (padrão do paper)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    # --- Dataset -----------------------------------------------------------
    # load_dataset("json", ...) lê arquivos JSONL diretamente.
    data_files = {"train": args.train_file, "test": args.test_file}
    ds = load_dataset("json", data_files=data_files)
    ds = ds.map(format_example)  # cria a coluna "text" no formato do Llama-2

    # --- Argumentos de treino ---------------------------------------------
    # optim="paged_adamw_32bit" -> Adam paginado (bitsandbytes), evita OOM
    # lr_scheduler_type="cosine"-> learning rate cai em curva cosseno
    # warmup_ratio=0.03         -> 3% dos passos iniciais para aquecimento
    # num_train_epochs=1        -> 1 passada pelo dataset (é pouco, mas o
    #                              dataset é pequeno; é um experimento)
    # fp16=True                 -> treino em meia precisão (float16)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",  # não envia métricas para W&B/etc.
    )

    # --- SFTTrainer (Supervised Fine-Tuning, da biblioteca trl) ------------
    # Ele cuida da tokenização, collation, aplicação do LoRA e loop de treino.
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1024,
        packing=False,
    )

    trainer.train()

    # --- Salva apenas o adapter LoRA (poucos MB) ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter LoRA salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
