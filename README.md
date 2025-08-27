
# Fine-Tuning BERT para Classificação de Texto

Projeto de exemplo para **fine-tuning** do modelo pré-treinado BERT (`bert-base-uncased`) usando a biblioteca Hugging Face Transformers. Ideal para treinar modelos de classificação de texto com datasets customizados em JSONL.

## Estrutura do projeto

meu-projeto-bert/
│
├─ notebooks/ # Notebooks do Google Colab
├─ scripts/ # Scripts Python para treinamento
├─ data/ # Datasets (JSONL)
├─ requirements.txt # Dependências do projeto
└─ README.md

## Instalação

Recomendado criar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

🔹 Bloco 1 – Treinamento 

**## Como usar**

1. Ajuste os caminhos para os datasets JSONL (treino.jsonl e teste.jsonl) no script ou notebook.
2. Execute o script ou notebook para treinar o modelo.
3. Verifique os resultados na pasta ./bert-Sales-Challenge-Model-Test.

**## Observações**

1. Modelo usado: bert-base-uncased
2. Fine-tuning para classificação binária: suporte e venda
3. Métrica de avaliação: accuracy
4. Código pensado para ser simples e facilmente adaptável para outros datasets ou tarefas.

🔹 Bloco 2 – Login, push para o Hugging Face Hub e uso do modelo

from huggingface_hub import notebook_login
notebook_login()  # Faz login no Hugging Face Hub

trainer.push_to_hub("LuaxSantos/SalesChallengeModel-Finetuning")

from transformers import pipeline

pipe = pipeline("text-classification", model="LuaxSantos/SalesChallengeModel-Finetuning")

resultado = pipe("quero comprar uma nova TV")
print(resultado)
