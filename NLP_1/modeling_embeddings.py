from gensim.models import Word2Vec, FastText
import numpy as np
from pre_processing import clean_text

def get_embeddings(datafile, tipo):
    #transformar as frases em listas de tokens
    tokenized_sentences = datafile['text'].apply(lambda x: clean_text(x, tokenize = True))

    #treinar o modelo nos dados
    if tipo == 'w2v':
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
    else:
        model = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)

    #transformar uma frase em um unico vetor

    def get_mean_vector(sentence_tokens):
        #pega o vetor de cada palavra, se a palavra existir no modelo
        vectors = [model.wv[word] for word in sentence_tokens if word in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0) #media aritmetica dos vetores
        else:
            return np.zeros(100) #vetor de zeros se nao encontrar nenhuma palavra
        
        #criar a matriz final
    X = np.array([get_mean_vector(tokens) for tokens in tokenized_sentences])
    return X