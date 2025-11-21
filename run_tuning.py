import os
import json
import itertools
import numpy as np
import pandas as pd
from knn_comparison.data import load_embeddings
from knn_comparison.annoy import AnnoyKNN
from knn_comparison.faiss import FAISSKNN
from knn_comparison.hnsw import HNSWKNN
from knn_comparison.exact import ExactKNN

# --- TUNING CONFIGURATION ---
OUTPUT_DIR = 'tuning'
TUNING_GRID = {
    'Annoy': {
        'n_trees': [25, 50, 100, 200]
    },
    'FAISS': {
        'nlist': [100, 400, 800, 1600]
    },
    'HNSW': {
        'ef_construction': [200, 400, 800],
        'M': [16, 32, 48]
    }
}
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'best_params.json')
# ---------------------------------

def main():
    print(f'The results will be saved in the directory: ./{OUTPUT_DIR}')
    print('Loading embeddings for tuning…')
    user_emb, item_emb, *_ = load_embeddings()
    n_test = 100
    k = 20
    rng = np.random.default_rng(42)
    test_user_idx = rng.choice(user_emb.shape[0], n_test, replace=False)
    test_item_idx = rng.choice(item_emb.shape[0], n_test, replace=False)

    print('Creating ground-truth for tuning…')
    exact_model = ExactKNN(n_neighbors=k)
    exact_model.build(user_emb, item_emb)
    gt_user = [set(exact_model.query_user(user_emb[i], k=k)[1]) for i in test_user_idx]
    gt_item = [set(exact_model.query_item(item_emb[i], k=k)[1]) for i in test_item_idx]

    print('\n=== HYPERPARAMETER TUNING (with weighted score) ===')
    best_params_found = {}
    all_tuning_results = []

    # We set weights for our composite metric
    RECALL_WEIGHT = 0.4
    TIME_WEIGHT = 0.6

    for name, params_grid in TUNING_GRID.items():
        print(f'\n--- Tuning {name} ---')
        best_score = -np.inf
        
        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        
        for combination in itertools.product(*param_values):
            current_params = dict(zip(param_names, combination))
            print(f"  Testing with params: {current_params}")
            
            if name == 'Annoy':
                model = AnnoyKNN(n_neighbors=k, **current_params)
            elif name == 'FAISS':
                model = FAISSKNN(n_neighbors=k, **current_params)
            elif name == 'HNSW':
                model = HNSWKNN(n_neighbors=k, **current_params)
            
            model.build(user_emb, item_emb)
            
            # 1. Recall@20
            recalls = []
            for i, idx in enumerate(test_user_idx):
                pred = set(model.query_user(user_emb[idx], k=k)[1])
                true = gt_user[i]
                inter = len(true & pred)
                # Recall = |relevant ∩ retrieved| / |relevant|
                recalls.append(inter / len(true) if true else 0.0)
            avg_recall = np.mean(recalls)

            # 2. Calculating the average request time
            q_times = []
            for i, idx in enumerate(test_user_idx):
                _, _, t = model.query_user(user_emb[idx], k=k)
                q_times.append(t)
            avg_q_time_ms = np.mean(q_times) * 1000

            # 3. Calculate the final composite score
            # Use np.log1p for safety (log(1+x)) to avoid log(0)
            score = RECALL_WEIGHT * avg_recall - TIME_WEIGHT * np.log1p(avg_q_time_ms)
            
            # 4. Display detailed information for this iteration
            print(f"    -> Recall@20: {avg_recall:.4f}, Avg Query Time: {avg_q_time_ms:.2f}ms, Score: {score:.4f}")

            # We save all results for analysis
            all_tuning_results.append({
                'Algorithm': name,
                **current_params,
                'Score': score,
                'Recall@20': avg_recall,
                'Avg Query Time (ms)': avg_q_time_ms
            })

            if score > best_score:
                best_score = score
                best_params_found[name] = current_params
                print(f"    *** NEW BEST SCORE: {best_score:.4f} ***")

        print(f'\nBest params for {name}: {best_params_found[name]} with Score: {best_score:.4f}')

    # Saving the best parameters to a JSON file
    with open(CONFIG_PATH, 'w') as f:
        json.dump(best_params_found, f, indent=4)
    
    # Save all tuning results in CSV format in the tuning directory.
    results_csv_path = os.path.join(OUTPUT_DIR, 'tuning_results.csv')
    pd.DataFrame(all_tuning_results).to_csv(results_csv_path, index=False)
    
    print(f'\nTuning complete. Best parameters saved to {CONFIG_PATH}')
    print(f'All results saved to {results_csv_path}')


if __name__ == '__main__':
    main()
