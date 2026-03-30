from utils.register import register_eval
import numpy as np

from utils.logger import get_logger
logger = get_logger(__name__)

def evaluate(preds, gts, topk=20):
    assert len(preds) == len(gts), "Predictions and ground truths must have the same length."
    assert topk > 0, "Top-k must be a positive integer."

    n = len(preds)
    ndcg = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pred = np.asarray(preds[i])
        gt = int(gts[i])
        assert 0 <= gt < pred.size, f"gt index out of range: {gt}"
        pred = np.nan_to_num(pred, nan=-1e30, posinf=1e30, neginf=-1e30)
        order = np.argsort(-pred, kind="stable")
        topk_indices = order[:min(topk, pred.size)]
        hit_pos = np.where(topk_indices == gt)[0]
        if hit_pos.size > 0:
            rank = int(hit_pos[0]) + 1
            ndcg[i] = 1.0 / np.log2(rank + 1)

    return float(np.mean(ndcg))


@register_eval("NDCG1")
def ndcg1(preds, gts, topk=1):
    return evaluate(preds, gts, topk=topk)

@register_eval("NDCG5")
def ndcg5(preds, gts, topk=5):
    return evaluate(preds, gts, topk=topk)

@register_eval("NDCG10")
def ndcg10(preds, gts, topk=10):
    return evaluate(preds, gts, topk=topk)
