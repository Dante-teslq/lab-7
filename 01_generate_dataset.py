"""
01_generate_dataset.py
----------------------
Script do Lab 7 (José Lucas - estudante).

Objetivo: gerar um dataset sintético de perguntas e respostas sobre
gestão de estoque e WMS usando a API do Google Gemini. O dataset será
usado depois no fine-tuning QLoRA do Llama-2.

Fluxo resumido:
  1. Lê a chave GEMINI_API_KEY do arquivo .env
  2. Pede ao Gemini para gerar pares {"prompt": ..., "response": ...}
  3. Junta 60 pares, embaralha e divide em 90% treino / 10% teste
  4. Salva em dataset/train.jsonl e dataset/test.jsonl (formato JSON Lines)
"""

import os
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm


# --- 1) Carrega variáveis do .env (chave da API) ------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY não encontrada. Crie um arquivo .env com:\n"
        "GEMINI_API_KEY=sua_chave"
    )

# Configura o SDK do Gemini com a chave
genai.configure(api_key=API_KEY)

# Modelo rápido e gratuito, suficiente para gerar dados sintéticos
model = genai.GenerativeModel("gemini-1.5-flash")


# --- 2) Parâmetros do dataset -------------------------------------------
TOTAL = 60   # total de pares que queremos gerar
BATCH = 5    # quantos pares pedir por chamada (evita respostas gigantes)

# Prompt "meta": o que a gente pede ao Gemini para produzir os exemplos.
# Obs.: instruímos explicitamente a retornar SÓ um array JSON válido,
# para facilitar o parsing depois.
SYSTEM_PROMPT = (
    "Você é um gerador de dados de treinamento para fine-tuning de um "
    "assistente especialista em gestão de estoque e WMS "
    "(Warehouse Management System). "
    "Gere exatamente {n} pares únicos e realistas no formato JSON, "
    "cobrindo temas como: controle de inventário, curva ABC, acurácia de "
    "estoque, picking, packing, putaway, endereçamento, FIFO/FEFO/LIFO, "
    "giro de estoque, ponto de pedido, níveis de serviço, SKU, "
    "cross-docking, recebimento, expedição, KPIs de WMS, integração "
    "ERP/WMS, coletor de dados e RFID. "
    "Retorne APENAS um array JSON válido, sem markdown, sem comentários, "
    'no formato: [{{"prompt": "pergunta do usuário", '
    '"response": "resposta técnica e detalhada"}}, ...]'
)


def extract_json(text: str):
    """
    O Gemini às vezes responde com o JSON dentro de ```json ... ``` ou
    com algum texto antes/depois. Aqui a gente limpa essas cercas e
    tenta extrair só o array JSON.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Se vier JSON malformado, ignoramos esse batch
        return []


def generate_batch(n: int):
    """Pede n pares ao Gemini e devolve lista de dicts válidos."""
    resp = model.generate_content(SYSTEM_PROMPT.format(n=n))
    data = extract_json(resp.text)
    # Filtro de sanidade: só mantém itens que têm as duas chaves esperadas
    return [
        {"prompt": str(d["prompt"]), "response": str(d["response"])}
        for d in data
        if isinstance(d, dict) and "prompt" in d and "response" in d
    ]


def main():
    pairs = []

    # Barra de progresso para acompanhar visualmente a geração
    with tqdm(total=TOTAL, desc="Gerando pares") as pbar:
        while len(pairs) < TOTAL:
            remaining = TOTAL - len(pairs)
            size = min(BATCH, remaining)
            try:
                batch = generate_batch(size)
            except Exception as e:
                # Erros de rede/quota: avisa e tenta de novo no próximo loop
                tqdm.write(f"Erro no batch (tentando novamente): {e}")
                continue
            # Nunca pega mais do que o necessário
            batch = batch[:remaining]
            pairs.extend(batch)
            pbar.update(len(batch))

    # --- 3) Embaralha e divide em treino (90%) e teste (10%) -----------
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train, test = pairs[:split], pairs[split:]

    # --- 4) Salva no formato JSONL (um JSON por linha) ------------------
    out_dir = Path("dataset")
    out_dir.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Resumo final
    print(f"Total de pares gerados: {len(pairs)}")
    print(f"  train: {len(train)} -> dataset/train.jsonl")
    print(f"  test:  {len(test)} -> dataset/test.jsonl")


if __name__ == "__main__":
    main()
