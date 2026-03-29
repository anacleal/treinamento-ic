from pre_processing import clean_text, data_analysis
import pandas as pd
from NMF import nmf_modelling
from BERTopic_model import run_bertopic

def main():

    with open('TREC/texts.txt', 'r', encoding='utf-8', errors='ignore') as f:
        linhas = f.read().splitlines()
    df = pd.DataFrame(linhas, columns=['text'])

    data_analysis(df['text'], "Original texts analysis")
    print(f"\n--- CLEANING DATA ---\n")
    df['cleaned_text'] = df['text'].apply(clean_text)
    data_analysis(df['cleaned_text'], "Cleaned texts analysis")

    print("\n--- TOPIC MODELING NMF ---\n")
    nmf_modelling(df['cleaned_text'].tolist())

    print("\n--- TOPIC MODELING BERTopic ---\n")
    model, topics, probs = run_bertopic(df['text'].tolist())

if __name__ == "__main__":
    main()