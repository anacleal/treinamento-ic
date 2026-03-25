import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    # Gráficos padrão com grades (darkgrid)
sns.set_style("darkgrid")

def data_analisis():
        # Carrega o primeiro dataset
        # header=None: O arquivo não tem cabeçalho
    df1 = pd.read_csv('dataset/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating', 'Date'], usecols=[0,1,2])

        # Converte a coluna Date para o formato de data do Python para permitir análises temporais
    df1['Date'] = pd.to_datetime(df1['Date'])
    
        # Converte a coluna Rating para float. Onde houver o ID do filme (ex: "1:"), o valor ficará como NaN (nulo)
    df1['Rating'] = df1[['Rating']].astype(float)

        # Exibe o tamanho total e alguns exemplos pulando de 5 em 5 milhões de linhas para inspeção rápida
    print('Dataset 1 shape: {}'.format(df1.shape))
    print('-Dataset examples-')
    print(df1.iloc[::5000000, :])

        # Carregamento dos outros datasets (sem a coluna de data para poupar memória)
    # df2 = pd.read_csv('dataset/combined_data_2.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
    # df3 = pd.read_csv('dataset/combined_data_3.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
    # df4 = pd.read_csv('dataset/combined_data_4.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])

        # Garante que todos tenham o Rating como float
    # df2['Rating'] = df2[['Rating']].astype(float)
    # df3['Rating'] = df3[['Rating']].astype(float)
    # df4['Rating'] = df4[['Rating']].astype(float)

    # df = pd.concat([df1, df2, df3, df4], ignore_index=True) 
        # Apenas o df1 para agilizar os testes de desenvolvimento
    df = df1

        # Reinicia o índice para garantir uma sequência numérica contínua de 0 até o fim
    df.index = np.arange(0, len(df))

    # CÁLCULO DE ESPARSIDADE E MÉTRICAS
    # Conta quantos NaNs existem na coluna Rating (isso equivale ao número total de filmes no arquivo)
    movie_count = df.isnull().sum().iloc[1]
    
    # Quantos usuários únicos existem (subtraindo os registros de filmes que aparecem na mesma coluna)
    cust_count = df['Cust_Id'].nunique() - movie_count
    
    # Total de notas dadas (excluindo os registros que são apenas títulos de filmes)
    rating_count = df['Cust_Id'].count() - movie_count
    
    # Cálculo da Esparsidade: (1 - Preenchimento Real / Capacidade Total da Matriz)
    # O quão "vazia" é a matriz de Usuários x Filmes
    sparsity = 1 - (rating_count / (cust_count * movie_count))

    print('\n---------------------------\n')
    print(f'movie count = {movie_count}')
    print(f'users count = {cust_count}')
    print(f'rating count = {rating_count}')
    print(f'sparsity of the dataset = {sparsity * 100:.2f}%\n')

    # Agrupa por nota (1 a 5) e conta a frequência de cada uma para o gráfico de distribuição
    ratings_matrix = df.groupby('Rating')['Rating'].agg(['count'])
    
    return df, ratings_matrix, movie_count, cust_count, rating_count


def data_cleaning(df):
    # LOCALIZAÇÃO DOS IDS DOS FILMES
    # Cria um dataframe booleano onde True indica que a linha é o título de um filme (Rating nulo)
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    # MAPEAMENTO DOS FILMES PARA AS LINHAS DE AVALIAÇÃO
    # Percorre os índices de onde começa um filme até onde começa o próximo
    # i = índice do próximo filme, j = índice do filme atual
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        # Cria um bloco repetindo o ID do filme atual pela quantidade de avaliações dele
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Trata o último filme do arquivo separadamente (já que não tem um "próximo" para comparar o índice)
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)


    # LIMPEZA FINAL
    # Remove as linhas que continham apenas o ID do filme (pois agora o ID estará em uma coluna própria)
    df = df[pd.notnull(df['Rating'])].copy()
    
    # Adiciona a nova coluna Movie_Id convertida para inteiro (economiza memória)
    df['Movie_Id'] = movie_np.astype(int)
    
    # Converte Cust_Id para inteiro para padronizar o dataset
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    
    print('-Dataset pós-limpeza (exemplos)-')
    print(df.iloc[::5000000, :])

    return df