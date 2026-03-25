import pandas as pd
import numpy as np

class MostPopularRecommender:
    def __init__(self):
        self.popularity_series = None
        self.train_data = None

    def fit(self, train_data):
        #só contar e ordenar os filmes "most popular"

        self.train_data = train_data
        self.popularity_series = train_data.groupby('Movie_Id')['Cust_Id'].count().sort_values(ascending = False)
        print(f"Modelo treinado com {len(self.popularity_series)} itens.")

    def recommend(self, user_id, top_n=10, remove_seen = True):
        #gera recomendacoes pra um usuario especifico

        recommendations = self.popularity_series.index.tolist()

        if remove_seen:
            seen_items = self.train_data[self.train_data['Cust_Id'] == user_id]['Movie_Id'].values
            recommendations = [item for item in recommendations if item not in seen_items]

        return recommendations[:top_n]