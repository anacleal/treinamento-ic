import re
import pandas as pd
import unicodedata
import nltk
import numpy as np
from scipy import stats

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction import text
from sklearn import model_selection, metrics, neighbors, svm, tree

from sklearn.model_selection import GridSearchCV


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

def clean_text(text):

    text = remove_accent(text.lower())
    text = re.sub(r'\d+', 'numtoken', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = word_tokenize(text)
    filtered = [
        stemmer.stem(w) 
        for w in words 
        if w not in stop_words and len(w)>2
    ]
    
    return " ".join(filtered)

def modeling_gridsearch(datafile):

    datafile['text'] = datafile['text'].apply(clean_text)

    vectorizer = text.TfidfVectorizer(
        min_df=5,
        max_features=5000,
        ngram_range=(1,1)
    )

    x = vectorizer.fit_transform(datafile['text']) #matriz com peso de importancia
    y = datafile['label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "KNN": neighbors.KNeighborsClassifier(),
        "SVM": svm.SVC(),
        "Decision Tree": tree.DecisionTreeClassifier(random_state=42)
    }

    parameter_grids = {
        "KNN": {
            'n_neighbors': [3, 5, 8],
            'metric': ['cosine', 'euclidean']
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "Decision Tree": {
            'max_depth': [None, 10, 20],
            'criterion': ['gini', 'entropy']
        }
    }

    for name, model in models.items():
        print(f"\nOtimizando {name}")

        grid = GridSearchCV(model, parameter_grids[name], cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        print(f"Melhores parametros: {grid.best_params_}")
        print(f"- {name}: Acuracia = {acc:.4f} | Macro F1 = {f1:.4f}")

    return datafile

def modeling(datafile):
    datafile['text'] = datafile['text'].apply(clean_text)

    vectorizer = text.TfidfVectorizer(
        min_df=5,
        max_features=5000,
        ngram_range=(1,1)
    )

    X = vectorizer.fit_transform(datafile['text']) #matriz com peso de importancia
    y = datafile['label']

    models = {
        "KNN": neighbors.KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        "SVM": svm.SVC(C=10, kernel='rbf'),
        "Tree": tree.DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=None)
    }

    # 10 folds - validacao cruzada

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