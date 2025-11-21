import numpy as np
from sklearn.metrics import precision_score, recall_score

def compute_metrics(true_idx: set, pred_idx: set):
    inter = len(true_idx & pred_idx)
    recall = inter / len(true_idx) if true_idx else 0.0
    precision = inter / len(pred_idx) if pred_idx else 0.0
    return precision, recall

def evaluate_model(model, user_emb, item_emb,
                   test_user_idx, test_item_idx,
                   exact_user_results, exact_item_results,
                   k=20):
    """Returns a dictionary with average metrics."""
    # user-based
    q_times = []
    precisions = []
    recalls = []
    for i, idx in enumerate(test_user_idx):
        _, pred, t = model.query_user(user_emb[idx], k=k)
        prec, rec = compute_metrics(exact_user_results[i], set(pred))
        q_times.append(t)
        precisions.append(prec)
        recalls.append(rec)

    avg_q_user = np.mean(q_times) * 1000
    avg_prec_user = np.mean(precisions)
    avg_rec_user = np.mean(recalls)

    # item-based
    q_times.clear()
    precisions.clear()
    recalls.clear()
    for i, idx in enumerate(test_item_idx):
        _, pred, t = model.query_item(item_emb[idx], k=k)
        prec, rec = compute_metrics(exact_item_results[i], set(pred))
        q_times.append(t)
        precisions.append(prec)
        recalls.append(rec)

    avg_q_item = np.mean(q_times) * 1000
    avg_prec_item = np.mean(precisions)
    avg_rec_item = np.mean(recalls)

    return {
        'Build Time User (s)': model.build_time_user,
        'Build Time Item (s)': model.build_time_item,
        'Total Memory (MB)': model.total_memory,
        'Avg Query Time User (ms)': avg_q_user,
        'Avg Query Time Item (ms)': avg_q_item,
        'Recall@20 User': avg_rec_user,
        'Recall@20 Item': avg_rec_item,
        'Precision@20 User': avg_prec_user,
        'Precision@20 Item': avg_prec_item,
    }

