import pandas as pd
from data_analisis import data_analisis
from modeling import modeling_tfidf, classifier
from modeling_embeddings import get_embeddings

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
    

    
if __name__ == "__main__":
    main()