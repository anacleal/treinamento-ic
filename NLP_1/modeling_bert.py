import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_bert_embeddings(datafile):
    def get_sentence_vector(text):
        #tokeniza o texto e conerte pra sensores do pytorch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        #o bert retorna um vetor pra cada token; aqui a gnt pega o vetor do token de indice 0, quer representa a frase toda
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings.flatten()
    
    X = np.array([get_sentence_vector(msg) for msg in datafile['text']])
    return X