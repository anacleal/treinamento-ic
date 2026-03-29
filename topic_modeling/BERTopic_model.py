from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def run_bertopic(docs):
    # 1. Definindo os sub-modelos (Opcional, mas dá controle total)
    # Reduzimos a dimensionalidade para manter a estrutura local e global
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    # Agrupamento que detecta densidade (não precisa definir n_topics de cara)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # 2. Inicializando o BERTopic
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True, verbose=True)

    # 3. Treinamento
    topics, probs = topic_model.fit_transform(docs)

    # 4. Visualizações (O "pulo do gato" do artigo)
    # Mostra a distância entre os tópicos em 2D
    fig_distance = topic_model.visualize_topics()
    fig_distance.write_html("bertopic_graphs/intertopic_distance_map.html")
    
    # Mostra as palavras mais importantes de forma comparativa
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html("bertopic_graphs/bar_graphs.html")
    
    # Hierarquia de como os tópicos se agrupam
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html("bertopic_graphs/hierarchical_clustering.html")

    # Supondo que seu modelo já foi treinado: topic_model.fit_transform(docs)

    return topic_model, topics, probs
