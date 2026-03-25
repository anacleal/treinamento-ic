from graphs import Visualizer

def run_graphs(df, ratings_matrix, movie_count, cust_count, rating_count):
    viz = Visualizer(output_dir='my_graphs')
    print("\nGerando e salvando gráficos em PNG...")

    # 2. Distribuição de Notas
    viz.plot_rating_distribution(ratings_matrix, movie_count, cust_count, rating_count)
    
    # 3. Popularidade dos Itens (Cauda Longa)
    movie_pop = df.groupby('Movie_Id')['Cust_Id'].count().sort_values(ascending=False)
    viz.plot_long_tail(movie_pop, entity_name="Itens")
    
    # 4. Histórico dos Usuários (Cauda Longa)
    user_hist = df.groupby('Cust_Id')['Movie_Id'].count().sort_values(ascending=False)
    viz.plot_long_tail(user_hist, entity_name="Usuarios")
    
    # 5. Evolução Temporal
    viz.plot_ratings_over_time(df)
    
    # 6. Crescimento Acumulado
    viz.plot_cumulative_users(df)
    
    print(f"\nArquivos PNG salvos")