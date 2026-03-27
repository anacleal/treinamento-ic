import pandas as pd
import numpy as np

#cb recommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#

#cf recommender
from scipy import sparse
#

class MostPopularRecommender:
    def __init__(self):
        self.popularity_series = None
        self.train_data = None
        self.name = "Most Popular"

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
    
#NAO CONSEGUI TESTAR O CB PQ A BASE DE DADOS NÃO TINHA UMA COLUNA DE GENEROS (fiz supondo que tinha)

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

        self.movie_indices = pd.Series(self.df_movies.index, index=self.df_movies['Movie_Id'])

    def recommend(self, user_id, top_n=10, remove_seen = True):
        #usar so filmes que ele deu >=4 estrelas
        user_history = self.df_ratings[(self.df_ratings['Cust_Id'] == user_id) & (self.df_ratings['Rating'] >= 4)]

        if user_history.empty:
            return pd.DataFrame(columns=['Movie_Id', 'score'])
        
        liked_ids = user_history['Movie_Id'].tolist()

        ## achar os filmes na matriz

        valid_index = [self.movie_index[mid] for mid in liked_ids if mid in self.movie_index]

        if not valid_index:
            return pd.DataFrame(columns=['Movie_Id', 'score'])
        
        ## similaridade em lote -- linear kernel (calcular o cosseno)

        user_siml_matrix = linear_kernel(self.tfidf_matrix[valid_index], self.tfidf_matrix)

        ## pontuacao final

        total_scores = user_siml_matrix.sum(axis=0)

        ## gerar o ranking

        recs = pd.DataFrame({
            'Movie_Id': self.df_movies['Movie_Id'].values,
            'score': total_scores
        }).sort_values('score', ascending= False)

        ## filtrar o que ela ja assistiu

        if remove_seen:
            seen_items = self.df_ratings[self.df_ratings['Cust_Id'] == user_id]['movieId'].unique()
            recs = recs[~recs['movieId'].isin(seen_items)]

        return recs.head(top_n)
    
class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_siml_matrix = None
        self.train_data = None
        self.name = "Collaborative Filtering"

    def fit(self, train_data):
        print(f"treinando {self.name}...")
        self.train_data = train_data

        #cria a matriz pivot - linhas (usuarios) x colunas (filmes)
        #matriz esparsa pra economizar ram

        pivot_df = train_data.pivot(index = 'Cust_Id', columns = 'Movie_Id', values = 'Rating').fillna(0)
        self.user_item_matrix = sparse.csr_matrix(pivot_df.values)

        #calcula o cosseno (similaridade)

        self.user_siml_matrix = cosine_similarity(self.user_item_matrix)
        self.user_ids = pivot_df.index
        self.movie_ids = pivot_df.columns
        print("treinamento concluido")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_ids: return[]

        user_idx = list(self.user_ids).index(user_id)

        #score de similaridade do usuário atual com todos os outros

        siml_scores = self.user_siml_matrix[user_idx]

        #media ponderada similaridade x notas dadas --> gera uma pontuação pra cada filme baseada no gosto dos vizinhos

        preds = siml_scores.dot(self.user_item_matrix.toarray()) / np.array([np.abs(siml_scores).sum()])
        recs = pd.DataFrame({'Movie_Id': self.movie_ids, 'score': preds})

        #filtra o q o usuario ja viu

        seen_items = self.train_data[self.train_data['Cust_Id'] == user_id]['Movie_Id']
        recs = recs[~recs['Movie_Id'].isin(seen_items)]

        return recs.sort_values('score', ascending=False).head(top_n)
    
class HybridRecommender:
    def __init__(self, cb_model, cf_model, cb_weight = 0.5, cf_weight = 0.5):
        # cb_weight: Peso para o Content-Based (0.0 a 1.0)
        # cf_weight: Peso para o Collaborative (0.0 a 1.0)

        self.cb_model = cb_model
        self.cf_model = cf_model
        self.cb_weight = cb_weight
        self.cf_weight = cf_weight
        self.name = "Hybrid"

    def recommend(self, user_id, top_n=10):
        #pega a recomendação de ambos os modelos

        df_cb = self.cb_model.recommend(user_id, n=n*2)
        df_cf = self.cf_model.recommend(user_id, n=n*2)

        #normaliza os scores (entre 0 e 1) pra poder somar
        df_cb['score'] = df_cb['score'] / df_cb['score'].max()
        df_cf['score'] = df_cf['score'] / df_cf['score'].max()
        
        #merge dos resultados pelo ID do Filme
        hybrid_df = pd.merge(df_cb, df_cf, on='Movie_Id', how='outer', suffixes=('_cb', '_cf')).fillna(0)
        
        #média ponderada
        hybrid_df['final_score'] = (hybrid_df['score_cb'] * self.cb_weight) + (hybrid_df['score_cf'] * self.cf_weight)
        
        return hybrid_df.sort_values('final_score', ascending=False).head(top_n)