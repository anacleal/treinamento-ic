from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def run_bertopic(docs):
    #uniform mainfold approximation and projection (UMAP) para redução de dimensionalidade
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    #hiearchical density-based spatial clustering of applications with noise (HDBSCAN) para clusterização // baseado em densidade, não precisa definir o número de clusters, e é robusto a ruídos
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    #inicializa o modelo
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True, verbose=True)

    #treina o modelo e retorna os tópicos e as probabilidades de cada documento pertencer a cada tópico
    topics, probs = topic_model.fit_transform(docs)

    #visualizações
    # Mostra a distância entre os tópicos em 2D
    fig_distance = topic_model.visualize_topics()
    fig_distance.write_html("bertopic_graphs/intertopic_distance_map.html")
    
    # Mostra as palavras mais importantes de forma comparativa
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html("bertopic_graphs/bar_graphs.html")
    
    # Hierarquia de como os tópicos se agrupam
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html("bertopic_graphs/hierarchical_clustering.html")

    return topic_model, topics, probs
