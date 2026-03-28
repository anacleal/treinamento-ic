from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import torch

##bert + finetuning


#classe pra o bert receber os pedacos de dados no formato necessario

class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels = None):
        self.encodings = encodings
        self.labels = labels    

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
def train_bert_finetuning(train_texts, train_labels):

    #configuracao

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") #quebra o texto em tokens e converte pra ids que o modelo entende

    #prepara os dados pro treinamento
    train_encodings = tokenizer(train_texts, truncation = True, padding = True, max_length=128)
    train_labels = list(train_labels)

    full_dataset = SMSDataset(train_encodings, train_labels)

    ## DIVISAO TREINO E TESTE 20/80

    train_size = int(0.8 * len(full_dataset))
    val_size = len (full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) #########################################################

    # carrega o modelo com 2 classes, spam e ham

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    # configs de treinamento

    training_args = TrainingArguments(
        output_dir = './results',       #diretorio onde os resultados do treinamento serao salvos, como checkpoints e logs
        num_train_epochs=2,             #quantas vezes o modelo vai passar por todo o dataset
        per_device_eval_batch_size=16,  #quantas amostras o modelo vai avaliar de cada vez durante a validacao
        eval_strategy="epoch",          #avalia o modelo no final de cada epoca
        save_strategy="no",             #nao salva o modelo durante o treinamento, ja que a gente so quer o resultado final
        learning_rate=5e-5,             #taxa de aprendizado, que controla o quanto o modelo ajusta seus pesos a cada passo de treinamento
        weight_decay=0.01,              #ajuda a prevenir overfitting, penalizando pesos muito grandes
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset= val_dataset
    )

    trainer.train() #exc do treinamento

    return model, tokenizer, device

def gen_finetuned_vectors(texts, model, tokenizer, device, batch_size = 64):
    model.eval() #coloca o modelo em modo de avaliacao, desativando dropout e outras tecnicas de regularizacao que so sao usadas durante o treinamento
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extraindo embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True) #output_hidden_states=True para obter os estados ocultos de todas as camadas do modelo, 
                                                                 #o que nos permite acessar os embeddings gerados pelo modelo para cada token

        cls_embeddings = outputs.hidden_states[-1][:, 0, :]

    #[-1] pega a ultima camada oculta, que é onde os embeddings finais são gerados, e [:, 0, :] seleciona o vetor correspondente ao token [CLS]
    # que é um token especial usado para representar a frase inteira no BERT.

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)