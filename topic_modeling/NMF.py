import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def nmf_modelling(texts, n_topics=10, max_features=1000):
    # tf idf 

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', max_df=0.85, min_df=2)
    X = vectorizer.fit_transform(texts)
    words = np.array(vectorizer.get_feature_names_out())

    # print(X)
    # print("X = ", words)

    # NMF
    nmf = NMF(n_components=n_topics, random_state=42, solver ='mu')
    W = nmf.fit_transform(X)
    H = nmf.components_

    for i, topic in enumerate(H):
        print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))

    # print(W[:10,:10])
    # print(H[:10,:10])

    return W, H