import pandas as pd
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')

wh_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
STOP_WORDS = set(stopwords.words('english')) - wh_words
stemmer = PorterStemmer()

########## pre processing ##########

def remove_accent(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        )

def clean_text(text):
    text = remove_accent(text.lower())
    text = re.sub(r'\d+', 'numtoken', text) #substitui os números por um token
    text = re.sub(r'[^a-z\s]', '', text) #remove caracteres que não sejam letras ou espaços
    text = re.sub(r"\s+", " ", text) #substitui múltiplos espaços por um único espaço

    words = word_tokenize(text)
    filtered = [
        stemmer.stem(w) 
        for w in words 
        if w not in STOP_WORDS and len(w)>2
    ]

    return " ".join(filtered)

########## data analysis ##########

def data_analysis(text_list, title):
    series = pd.Series(text_list)
    word_count = series.apply(lambda x: len(str(x).split()))
    char_count = series.apply(lambda x: len(str(x)))

    print(f"--- Analysis: {title} ---")
    print(f"Mean Words: {word_count.mean():.2f}")
    print(f"Mean Characters: {char_count.mean():.2f}")
    print(f"Maximum Size (words): {word_count.max()}")
    print(f"Minimum Size (words): {word_count.min()}\n")

    return word_count

