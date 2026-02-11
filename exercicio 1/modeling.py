import re
import pandas as pd
import unicodedata
import nltk

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

def modeling(datafile):

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
