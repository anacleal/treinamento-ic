import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("destilbert-base-uncased")

def get_bert_embeddings(datafile):
    def get_sentence_vector(text):
        #tokeniza o texto e conerte pra sensores do pytorch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
