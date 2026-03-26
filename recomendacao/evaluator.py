import numpy as np
class Evaluator:
    @staticmethod
    def precision_at_k(actual, predicted, k):
        act_set = set(actual) #itens que o usuario reamlmente interagiu
        pred_set = set(predicted[:k]) #top k recomendações que o modelo fez 
        if not pred_set: return 0
        # acertos/total
        return len(act_set & pred_set) / float(k)

    @staticmethod
    def recall_at_k(actual, predicted, k):
        act_set = set(actual)
        pred_set = set(predicted[:k])
        if not act_set: return 0
        #acertos predicted/ titulos que o usuar
        return len(act_set & pred_set) / float(len(act_set))

    @staticmethod
    def f1_score(precision, recall):
        #media harmonica
        if (precision + recall) == 0: return 0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(actual, predicted, k):

        #qualidade de uma lista de resultados retornada pelo modelo em relação aos itens que o usuário realmente indicou como relevantes.
        #um resultado relevante que aparece no topo da lista terá uma pontuação mais alta do que se ele aparecesse no final.

        act_set = set(actual)
        res = 0
        for i, p in enumerate(predicted[:k]):
            if p in act_set:
                res += 1 / np.log2(i + 2)
        
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(act_set), k))])
        return res / idcg if idcg > 0 else 0