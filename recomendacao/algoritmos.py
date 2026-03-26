import pandas as pd
import numpy as np

#cb recommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
#

class MostPopularRecommender:
    def __init__(self):
        self.popularity_series = None
        self.train_data = None

    def fit(self, train_data, movies_metadata=None):

        counts = train_data.groupby('Movie_Id')['Cust_Id'].count().reset_index()
        counts.columns = ['Movie_Id', 'score']

        # tipo join do sql
        if movies_metadata is not None:
            #se tier os nomes dos filmes, ja guarda aqui
            counts = counts.merge(movies_metadata, on='Movie_Id', how = 'left')


        #só contar e ordenar os filmes "most popular"
        
        self.popularity_df = counts.sort_values('score', ascending = False)
        self.train_data = train_data

    def recommend(self, user_id, top_n=10, remove_seen = True):
        #gera recomendacoes pra um usuario especifico -- mascara do pandas

        seen_items = self.train_data[self.train_data['Cust_Id'] == user_id]['Movie_Id']
        recommendations = self.popularity_df[~self.popularity_df['Movie_Id'].isin(seen_items)]

        return recommendations.head(top_n)
    
#NAO CONSEGUI TESTAR PQ A BASE DE DADOS NÃO TINHA UMA COLUNA DE GENEROS (fiz supondo que tinha)

class ContentBasedRecommender:

    #olha pras caracteristicas dos ITENS (generos dos filmes no caso), e nao para o que outros usuarios estao vendo

    def __init__(self, df_ratings, df_movies):
        self.df_ratings = df_ratings
        self.df_movies = df_movies.reset_index(drop=True) #linha 0 do df = linha 0 da tfidf
        self.name = "Content-Based"
        self.movie_index = None
        self.tfidf_matrix = None

    def fit(self):
        print(f"treinando {self.name}")

        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df_movies['genres_str'])

        #### alinhar o id do filme com a linha da matriz, pro calculo correto do vetor do tfidf

        self.movie_indices = pd.Series(self.df_movies.index, index=self.df_movies['movieId'])

    def recommend(self, user_id, n=10, remove_seen = True):
        #usar so filmes que ele deu >=4 estrelas
        user_history = self.df_ratings[(self.df_ratings['userId'] == user_id) & (self.df_ratings['rating'] >= 4)]

        if user_history.empty:
            return pd.DataFrame(columns=['movieId', 'score'])
        
        liked_ids = user_history['movieId'].tolist()

        ## achar os filmes na matriz

        valid_index = [self.movie_index[mid] for mid in liked_ids if mid in self.movie_index]

        if not valid_index:
            return pd.DataFrame(columns=['movieId', 'score'])
        
        ## similaridade em lote -- linear kernel (calcular o cosseno)

        user_siml_matrix = linear_kernel(self.tfidf_matrix[valid_index], self.tfidf_matrix)

        ## pontuacao final

        total_scores = user_siml_matrix.sum(axis=0)

        ## gerar o ranking

        recs = pd.DataFrame({
            'movieId': self.df_movies['movieId'].values,
            'score': total_scores
        }).sort_values('score', ascending= False)

        ## filtrar o que ela ja assistiu

        if remove_seen:
            seen_items = self.df_ratings[self.df_ratings['userId'] == user_id]['movieId'].unique()
            recs = recs[~recs['movieId'].isin(seen_items)]

        return recs.head(n)
    
# class Collaborative:
#     def __init__(self, df_ratings, df_movies):