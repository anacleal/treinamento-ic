import numpy as np
class Evaluator:
    @staticmethod
    def precision_at_k(actual, predicted, k):
        act_set = set(actual) #itens que o usuario reamlmente interagiu
        pred_set = set(predicted[:k])
        if not pred_set: return 0
        return len(act_set & pred_set) / float(k)

    @staticmethod
    def recall_at_k(actual, predicted, k):
        act_set = set(actual)
        pred_set = set(predicted[:k])
        if not act_set: return 0
        return len(act_set & pred_set) / float(len(act_set))

    @staticmethod
    def f1_score(precision, recall):
        if (precision + recall) == 0: return 0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(actual, predicted, k):
        """Calcula o ganho cumulativo descontado normalizado"""
        act_set = set(actual)
        res = 0
        for i, p in enumerate(predicted[:k]):
            if p in act_set:
                res += 1 / np.log2(i + 2)
        
        # Cálculo do IDCG (Ideal DCG)
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(act_set), k))])
        return res / idcg if idcg > 0 else 0