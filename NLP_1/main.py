import pandas as pd
from data_analisis import data_analisis
from modeling import modeling_tfidf, classifier
from modeling_embeddings import get_embeddings
from modeling_bert import get_bert_embeddings
from bert_fine_tuned import get_finetuned_vectors, train_bert_finetuning

def main():
    datafile = pd.read_csv('dataset/SMSSpamCollection', sep ='\t', names=['label', 'text'])
    
    #datafile = data_analisis(datafile)

    print("\nTFIDF\n")

    X_tfidf, y = modeling_tfidf(datafile)
    classifier(X_tfidf, y)

    print("---------------------\n")

    print("\nWORD2VEC\n")
    X_w2v = get_embeddings(datafile, tipo='w2v')
    classifier(X_w2v, y)

    print("---------------------\n")

    print("\nFASTTEXT\n")
    X_ft = get_embeddings(datafile, tipo='ft')
    classifier(X_ft, y)

    ## fast text -> quando os vetores não são discriminativas o suficiente, o classificador não consegue agir mto bem
    # " porque o fast text (0.84) foi pior que o tfidf (0.96)"
    # " média simples dos vetores diluiu as palavras-chave características de spam em mensagens curtas "
    # " 1. dt set é muito pequeno, entao o tfidf que é estatistico lida melhor com pouco dado / em oposicao com w2v e ft que precisam de dados pra entender semanticamente"


    print("---------------------\n")

    print("\nBERT\n")
    X_bert = get_bert_embeddings(datafile)
    classifier(X_bert, y)

    # o BERT é quase tão bom quanto o TF-IDF para detectar palavras-chave
    # mas com a vantagem de que ele entende o contexto. 
    # Se você recebesse um spam com palavras escritas de forma criativa ou sarcástica
    # o BERT provavelmente pegaria o que o TF-IDF deixaria passar.

    print("---------------------\n")

    print("\nBERT FINE TUNED\n")
    bt_model, bt_tokenizer, bt_device = train_bert_finetuning(datafile)
    X_ft_bert = get_finetuned_vectors(datafile['text'].tolist(), bt_model, bt_tokenizer, bt_device)
    classifier(X_ft_bert, y)


    
if __name__ == "__main__":
    main()