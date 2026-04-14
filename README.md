# Lab 7 — Fine-tuning QLoRA de um LLM para Gestão de Estoque / WMS

> Projeto acadêmico desenvolvido por **José Lucas** como estudo prático de fine-tuning de LLMs aplicado ao domínio de gestão de estoque e WMS.

## O que é isso

A ideia é simples: pegar um modelo de linguagem genérico (Llama-2-7b) e ensiná-lo a responder bem sobre estoque e WMS — sem precisar de um servidor caro pra isso.

O pipeline tem três partes:

1. Gerar 60 perguntas e respostas sobre estoque/WMS com a API do **Google Gemini**
2. Fazer fine-tuning do Llama-2 com **QLoRA** (quantização 4-bit + adaptadores LoRA)
3. Salvar o adapter e documentar tudo

O código é comentado porque o objetivo é entender o que está acontecendo em cada etapa, não só executar.

## Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA 11.8+ (o script roda em CPU, mas QLoRA sem GPU não faz sentido na prática)
- Conta no Google AI Studio para gerar a API key: https://aistudio.google.com/app/apikey
- ~15 GB livres em disco pro modelo base do Llama-2

## Configuração

Crie um arquivo `.env` na raiz com sua chave do Gemini: