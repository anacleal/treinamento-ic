import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_analisis import data_analisis, data_cleaning
from algoritmos import MostPopularRecommender, ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from evaluator import Evaluator
from graphs import Visualizer

def main():

    df, ratings_matrix, movie_count, cust_count, rating_count = data_analisis()
    df = data_cleaning(df)
    
    # load títulos para o Content-Based
    movies_df = pd.read_csv('movie_titles.csv', encoding='latin-1', header=None, names=['Movie_Id', 'Year', 'Name'])
    movies_df['genres_str'] = movies_df['Name'] #content based gambiarra :P
    
    # Treino e Teste (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Inicialização e Treino
    print("\nTreinando modelos...")
    
    mp = MostPopularRecommender()
    mp.fit(train_df)

    cb = ContentBasedRecommender(train_df, movies_df)
    cb.fit()

    cf = CollaborativeFilteringRecommender()
    cf.fit(train_df)

    hb = HybridRecommender(cb, cf) # Híbrido 50/50 por padrão

    modelos = [mp, cb, cf, hb]
    
    # Avaliação
    K = 10
    # Pegamos uma amostra de usuários do teste
    usuarios_teste = test_df['Cust_Id'].unique()[:50] 
    resultados = []

    print(f"\nAvaliando modelos com {len(usuarios_teste)} usuários (Top {K})...")

    for modelo in modelos:
        print(f"Calculando métricas para: {modelo.name}...")
        p_list, r_list, n_list = [], [], []

        for user in usuarios_teste:
            # O que o usuário realmente assistiu no set de TESTE
            actual = test_df[test_df['Cust_Id'] == user]['Movie_Id'].tolist()
            
            # O que o modelo recomenda (top 10)
            recs = modelo.recommend(user, top_n=K)
            
            if not recs.empty:
                predicted = recs['Movie_Id'].tolist()
            else:
                predicted = []

            p_list.append(Evaluator.precision_at_k(actual, predicted, K))
            r_list.append(Evaluator.recall_at_k(actual, predicted, K))
            n_list.append(Evaluator.ndcg_at_k(actual, predicted, K))

        resultados.append({
            'Modelo': modelo.name,
            'Precision@K': np.mean(p_list),
            'Recall@K': np.mean(r_list),
            'nDCG@K': np.mean(n_list)
        })

    # Resultados e Gráficos
    df_results = pd.DataFrame(resultados)
    print("\n--- RESULTADO FINAL ---")
    print(df_results)

    viz = Visualizer(output_dir='my_graphs')
    viz.plot_model_comparison(df_results)
    print("\nGráfico comparativo salvo em 'my_graphs/comparacao_modelos.png'")

if __name__ == "__main__":
    main()