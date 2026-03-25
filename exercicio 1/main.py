import pandas as pd
from data_analisis import data_analisis
from modeling_tfidf import modeling_tfidf, classifier, modeling_gridsearch
from modeling_word2vec import modeling_word2vec
from modeling_fast_text import modeling_fasttext

def main():
    datafile = pd.read_csv('dataset/SMSSpamCollection', sep ='\t', names=['label', 'text'])
    
    #datafile = data_analisis(datafile)

    print("\nTFIDF\n")

    X, y, vectorizer = modeling_tfidf(datafile)
    classifier(datafile, X, y, vectorizer)
    modeling_gridsearch(datafile, X, y, vectorizer)

    # print("\nWORD2VEC\n")
    
    # df_w2v = modeling_word2vec(datafile)

    # print("\nFAST TEXT\n")

    # df_ft = modeling_fasttext(datafile)

if __name__ == "__main__":
    main()