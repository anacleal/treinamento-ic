import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from most_popular import MostPopularRecommender
from data_analisis import data_analisis, data_cleaning
from run_graphs import run_dt_analisis


def main():
    df, ratings_matrix, movie_count, cust_count, rating_count = data_analisis()

    df = data_cleaning(df)

    #run_graphs(df, ratings_matrix, movie_count, cust_count, rating_count)

    recommender = MostPopularRecommender()
    recommender.fit(df)

    user_test = 1488844
    top_5 = recommender.recommend(user_id=user_test, top_n=5)
    
    print(f"\nRecomendações para o usuário {user_test}:")
    print(top_5)

if __name__ == "__main__":
    main()