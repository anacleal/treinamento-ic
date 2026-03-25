import re
import pandas as pd
import unicodedata
import nltk
import numpy as np
from scipy import stats

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn import model_selection, metrics, neighbors, svm, tree
from gensim.models import Word2Vec

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def remove_accent(text):

    return ''.join(
        c for c in unicodedata.normalize('NFD', text) 
        if unicodedata.category(c) != 'Mn'
    )

def clean_text_list(text): #lista de palavras

    text = remove_accent(text.lower())
    text = re.sub(r'\d+', 'numtoken', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = word_tokenize(text)
    filtered = [
        stemmer.stem(w) 
        for w in words 
        if w not in stop_words and len(w)>2
    ]
    
    return filtered

def get_mean_vector_w2v(word_list, model):
    valid_words = [word for word in word_list if word in model.wv.key_to_index]

    if len(valid_words) >= 1:
        return np.mean(model.wv[valid_words], axis=0)
    else:
        return np.zeros(model.vector_size)


def modeling_word2vec(datafile):
    datafile['tokens'] = datafile['text'].apply(clean_text_list)

    w2v_model = Word2Vec(sentences=datafile['tokens'], vector_size=100, window=5, min_count=2, workers=4)
    
    vector_list = [get_mean_vector_w2v(tokens, w2v_model) for tokens in datafile['tokens']]

    X = np.array(vector_list)
    y = datafile['label']

    models = {
        "KNN": neighbors.KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        "SVM": svm.SVC(C=10, kernel='rbf'),
        "Tree": tree.DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=None)
    }

    kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'MODELO':<15} | {'MÉDIA F1':<10} | {'DESVIO':<10}")
    print("-" * 40)

    for name, model in models.items():
        scores = model_selection.cross_val_score(model, X, y, cv= kfold, scoring= 'f1_macro')
        results[name]= scores
        print(f"{name:<15} | {scores.mean():<10.4f} | {scores.std():.4f}")

    comparisons = [("KNN", "SVM"), ("KNN", "Tree"), ("SVM", "Tree")]
    alpha = 0.05 / 3
    
    print(f"\nteste T, alpha = {alpha:.4f}")

    for m1, m2 in comparisons: 
        t_stat, p_value = stats.ttest_rel(results[m1], results[m2])

        if p_value < alpha:
            winner = m1 if results[m1].mean() > results[m2].mean() else m2
            print(f"{m1} vs {m2}: p={p_value:.5f} -> diferenca (vencedor: {winner})")
        else:
            print(f"{m1} vs {m2}: p={p_value:.5f} -> empate")

    return datafile

#dataset é muito pequeno para o word2vec, quando ele faz a media
#entre as palavras da frase, as palavras evidentemente de "spam"
#tem a sua importancia diminuida
