import os
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, output_dir='plots'):

        # Inicializa a classe definindo onde os gráficos serão salvos

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _save_and_close(self, filename):

        #Método interno (auxiliar) para salvar o gráfico atual e fechar a figura

        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close() # Libera a memória RAM
        print(f"Gráfico salvo em: {path}")

    def plot_rating_distribution(self, ratings_matrix, movie_count, cust_count, rating_count):

        plt.figure(figsize=(12, 8))
        # kind='barh' cria barras horizontais
        ax = ratings_matrix.plot(kind='barh', legend=False, color='skyblue', edgecolor='black')
        plt.title(f'Distribuição de Notas\n({movie_count:,} Filmes, {cust_count:,} Usuários)', fontsize=15)
        plt.xlabel('Quantidade de Avaliações')
        plt.ylabel('Nota (Rating)')
        
        # Cálculo para adicionar o texto da porcentagem dentro/sobre as barras
        sum_total = ratings_matrix.sum().iloc[0]
        for i in range(len(ratings_matrix)):
            count = ratings_matrix.iloc[i, 0]
            percentage = count * 100 / sum_total
            # ax.text coloca o texto na posição exata do gráfico
            ax.text(count/4, i, f'{percentage:.1f}%', color='black', weight='bold', va='center')
        
        self._save_and_close('1_distribuicao_notas.png')

    def plot_long_tail(self, data, entity_name="Itens"):

        # Plota o fenômeno da 'Cauda Longa' -> poucos filmes têm muitas avaliações, enquanto a maioria tem pouquíssimas

        plt.figure(figsize=(10, 6))
        plt.plot(data.values, color='red' if entity_name == "Itens" else 'green')
        plt.fill_between(range(len(data)), data.values, alpha=0.2, color='red' if entity_name == "Itens" else 'green')
        
        plt.title(f'Cauda Longa - Popularidade dos {entity_name}')
        plt.xlabel(f'{entity_name} (Ordenados)')
        plt.ylabel('Número de Avaliações')
        
        # A escala logarítmica é essencial para ver a diferença entre 1 e 100.000 avaliações
        plt.yscale('log') 
        self._save_and_close(f'2_long_tail_{entity_name.lower()}.png')

    def plot_ratings_over_time(self, df):

        # Cria um gráfico de linha mostrando a evolução temporal do sistema
        # Agrupa os dados por ano para ver se o volume de avaliações cresceu ou diminuiu

        plt.figure(figsize=(10, 6))
        # .dt.year extrai o ano da coluna de data e conta as ocorrências
        df_time = df['Date'].dt.year.value_counts().sort_index()
        df_time.plot(kind='line', marker='o', color='purple', linewidth=2)
        
        plt.title('Evolução das Avaliações por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Total de Avaliações')
        plt.grid(True, linestyle='--', alpha=0.6)
        self._save_and_close('3_avaliacoes_por_tempo.png')

    def plot_cumulative_users(self, df):
        
        # Mostra o crescimento acumulado da base de usuários ao longo do tempo.
        # Considera a data da PRIMEIRA avaliação como a data de entrada do usuário.

        plt.figure(figsize=(10, 6))
        # Pega a data mínima (primeira vez) de cada usuário único
        user_entry = df.groupby('Cust_Id')['Date'].min().sort_values()
        # np.arange cria uma sequência de 1 até o total de usuários para o eixo acumulado
        cumulative_count = np.arange(1, len(user_entry) + 1)
        
        plt.plot(user_entry, cumulative_count, color='darkblue', linewidth=2)
        plt.title('Crescimento da Base de Usuários (Acumulado)')
        plt.xlabel('Tempo')
        plt.ylabel('Total de Usuários')
        plt.grid(True)
        self._save_and_close('4_crescimento_usuarios.png')