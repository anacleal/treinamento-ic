import matplotlib.pyplot as plt
import pandas as pd

def data_analisis(datafile):

    num_documents = len(datafile)
    num_classes = datafile['label'].nunique()
    documents_p_class = datafile['label'].value_counts()
    datafile['word_count'] = datafile['text'].apply(lambda x: len(x.split()))

    print(f"-- Metricas do dataset --")
    print(f"número de mensagens: {num_documents}")
    print(f"número de classes de mensagens: {num_classes}")
    print(f"mensagens por classe: {documents_p_class}")

    #para os graficos
    #1. distribuicao de classes
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    datafile['label'].value_counts().plot(kind='bar', color=['blue', 'orange'])
    plt.title('Distribuição de classes (spam vs ham)')
    plt.ylabel('contagem')

    #2. distribuicao do tamanho das mensagens
    plt.subplot(1, 2, 2)
    plt.hist(datafile['word_count'], bins=50, color='green', edgecolor='black')
    plt.title('Distribuição de comprimento')
    plt.xlabel('numero de palavras')
    plt.ylabel('numero de amostras')

    plt.tight_layout()
    plt.show()
    return datafile