import pandas as pd
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')
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
        if w not in stop_words and w.isalpha()
    ]
    
    return " ".join(filtered)

def pre_processing(datafile):

    datafile['text'] = datafile['text'].apply(clean_text)
    return datafile