import re
import pandas as pd
import unicodedata
import nltk
import numpy as np
from scipy import stats
from pre_processing import clean_text

from sklearn.feature_extraction import text
from sklearn import model_selection, neighbors, svm, tree

from sklearn.model_selection import GridSearchCV

def modeling_tfidf(datafile):
    datafile['text'] = datafile['text'].apply(clean_text)

    # Configura o vetorizador TF-IDF:
    # min_df=5: ignora palavras que aparecem em menos de 5 documentos
    # max_features=5000: seleciona apenas as 5000 palavras com maior score

    vectorizer = text.TfidfVectorizer(
        min_df=5,
        max_features=5000,
        ngram_range=(1,1)
    )

    X = vectorizer.fit_transform(datafile['text']) #matriz com peso de importancia
    y = datafile['label']

    return X, y

def classifier(X, y):

    # Gridsearch, Cross-Validation e Teste T.

    configs = {
        "KNN": (neighbors.KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 8],
            'metric': ['cosine', 'euclidean']
        }),
        "SVM": (svm.SVC(), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        "Decision Tree": (tree.DecisionTreeClassifier(random_state=42), {
            'max_depth': [None, 10, 20],
            'criterion': ['gini', 'entropy']
        })
    }

    # 10 folds - validacao cruzada e estratificada

    results = {}
    kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print(f"\n{'MODELO':<15} | {'MELHORES PARÂMETROS'}")
    print("-" * 60)

    # avalia cada modelo um a um

    for name, (model, params) in configs.items():
        #gridsearch
        grid = GridSearchCV(model, params, cv = 5, scoring='f1_macro', n_jobs=1)
        grid.fit(X,y)
        best_model = grid.best_estimator_

        print(f"{name:<15} | {grid.best_params_}")

        #cross validation usando o melhor modelo encontrado

        scores = model_selection.cross_val_score(best_model, X, y, cv=kfold, scoring='f1_macro')
        results[name] = scores

    print(f"\n{'MODELO':<15} | {'MEDIA F1':<10} | {'DESVIO':<10}")
    for name, scores in results.items():
        print(f"{name:<15} | {scores.mean():10.4f} | {scores.std():.4f}")

    comparisons = [("KNN", "SVM"), ("KNN", "Decision Tree"), ("SVM", "Decision Tree")]
    alpha = 0.05/3

    print(f"\nTeste T (alpha = {alpha:.4f}):")
    for m1, m2 in comparisons:
        t_stat, p_value = stats.ttest_rel(results[m1], results[m2])
        status = "Diferença" if p_value <alpha else "Empate"
        winner = m1 if results[m1].mean() > results[m2].mean() else m2
        print(f"{m1} vs {m2}: p={p_value:.5f} -> {status} (Vencedor: {winner if p_value < alpha else 'N/A'})")