from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import os

# %%
dataset = load_dataset("json", data_files={"treino": "/content/treino.jsonl", "teste": "/content/teste.jsonl"})

# %%
dataset

# %%
checkpoint = "bert-base-uncased"

# %%
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

mapDict = {
    "suporte": 0,
    "venda": 1
}

def transform_labels(label):
  label = label["completion"]
  result = [] # Use .get() with a default to handle potential missing keys
  for l in label:
    result.append(mapDict[l])
  return {"label": result}


def tokenize_data(example):
  # Assuming the text to be classified is in the 'completion' column
  return tokenizer(example["prompt"], padding=True, truncation=True)

# %%
tokenized_datasets = dataset.map(tokenize_data, batched=True)
tokenized_datasets = tokenized_datasets.map(transform_labels, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
output_dir = "./bert-Sales-Challenge-Model-Test"

training_args = TrainingArguments(
    output_dir=output_dir,
    report_to='none'
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) # Ajustado para 2 classes

# Adicionar o mapeamento de ID para rótulo à configuração do modelo
# Invertemos o mapDict para ter {id: label_name}
id2label = {v: k for k, v in mapDict.items()}
model.config.id2label = id2label
model.config.label2id = mapDict # Opcional, mas útil ter o mapeamento inverso também

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

# %%
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# %%
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["treino"],
    eval_dataset=tokenized_datasets["teste"], # Alterado para usar o conjunto de teste para avaliação
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %%
trainer.evaluate()

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
trainer.push_to_hub("LuaxSantos/SalesChallengeModel-Finetuning")

# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="LuaxSantos/bert-Sales-Challenge-Model-Test")

# %%
pipe("quero comprar uma nova TV")

# %%


