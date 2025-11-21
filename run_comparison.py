import json
import numpy as np
import pandas as pd
import sys
import os
import argparse
import subprocess
import pickle
import tempfile
from time import time

from knn_comparison.data import load_embeddings
from knn_comparison.exact import ExactKNN
from knn_comparison.annoy import AnnoyKNN
from knn_comparison.faiss import FAISSKNN
from knn_comparison.hnsw import HNSWKNN
from knn_comparison.bench import evaluate_model
from knn_comparison.utils import MemoryTracker
from knn_comparison.viz import plot_comparison

# Configuration
CONFIG_PATH = 'tuning/best_params.json'
DEFAULT_PARAMS = {
    'Annoy': {'n_trees': 50},
    'FAISS': {'nlist': 1600},
    'HNSW': {'ef_construction': 200, 'M': 16},
}

# Temporary files for transferring test and Ground Truth indexes between processes
TEMP_DIR = tempfile.gettempdir()
PATH_TEST_INDICES = os.path.join(TEMP_DIR, 'knn_test_indices.pkl')
PATH_GT_USER = os.path.join(TEMP_DIR, 'knn_gt_user.pkl')
PATH_GT_ITEM = os.path.join(TEMP_DIR, 'knn_gt_item.pkl')

def load_model_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print ("File not found, using standard parameters")
        return DEFAULT_PARAMS

# --- WORKER LOGIC (Runs separately for each model) ---
def run_worker(model_name):
    """
    Этот код выполняется в отдельном, чистом процессе.
    """
    best_params = load_model_config()
    
    # 1. Load the embeddings (yes, every time anew, that's the price of isolation)
    user_emb, item_emb, *_ = load_embeddings()
    
    # 2. Load the prepared test indexes and Ground Truth
    with open(PATH_TEST_INDICES, 'rb') as f:
        indices = pickle.load(f)
        test_user_idx, test_item_idx = indices['user'], indices['item']
    
    with open(PATH_GT_USER, 'rb') as f:
        gt_user = pickle.load(f)
    
    with open(PATH_GT_ITEM, 'rb') as f:
        gt_item = pickle.load(f)

    #3. Initialization of one specific model
    model = None
    if model_name == 'Exact KNN':
        model = ExactKNN(n_neighbors=20)
    elif model_name == 'Annoy':
        model = AnnoyKNN(n_neighbors=20, **best_params.get('Annoy', {}))
    elif model_name == 'FAISS':
        model = FAISSKNN(n_neighbors=20, **best_params.get('FAISS', {}))
    elif model_name == 'HNSW':
        model = HNSWKNN(n_neighbors=20, **best_params.get('HNSW', {}))
    
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    # 4. Building and measuring memory
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        model.build(user_emb, item_emb)
    except Exception as e:
        # If the model crashes, return the error in JSON
        print(json.dumps({'error': str(e)}))
        return

    total_mem = tracker.get_usage()
    
    model.total_memory = total_mem
    
    # 5.Benchmarking
    metrics = evaluate_model(model, user_emb, item_emb,
                             test_user_idx, test_item_idx,
                             gt_user, gt_item, k=20)
    
    # Add the algorithm name (the other metrics should already be inside if evaluate_model collects them)
    metrics['Algorithm'] = model_name
    
    # In case evaluate_model didn't add these fields, let's update them explicitly
    metrics['Build Time (User)'] = getattr(model, 'build_time_user', 0)
    metrics['Build Time (Item)'] = getattr(model, 'build_time_item', 0)
    metrics['Total Memory (MB)'] = total_mem

    # 6. Output the result to stdout as JSON for the master
    print("__RESULT_START__")
    print(json.dumps(metrics))
    print("__RESULT_END__")


# --- ORCHESTRATOR LOGIC (Controls the launch) ---
def run_orchestrator():
    print("=== ORCHESTRATOR STARTED ===")
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to '{RESULTS_DIR}' directory.")

    # 1. Data preparation (one time)
    print("Loading embeddings for Ground Truth generation...")
    user_emb, item_emb, *_ = load_embeddings()
    
    # Generating a test sample
    n_test = 100
    rng = np.random.default_rng(42)
    test_user_idx = rng.choice(user_emb.shape[0], n_test, replace=False)
    test_item_idx = rng.choice(item_emb.shape[0], n_test, replace=False)
    
    # Preserving indexes
    with open(PATH_TEST_INDICES, 'wb') as f:
        pickle.dump({'user': test_user_idx, 'item': test_item_idx}, f)

    # Ground Truth Generation (Exact KNN)
    print("Calculating Ground Truth using ExactKNN...")
    exact = ExactKNN(n_neighbors=20)
    exact.build(user_emb, item_emb)
    
    gt_user = [set(exact.query_user(user_emb[i], k=20)[1]) for i in test_user_idx]
    gt_item = [set(exact.query_item(item_emb[i], k=20)[1]) for i in test_item_idx]
    
    # We save GT to disk so that workers can run them.
    with open(PATH_GT_USER, 'wb') as f:
        pickle.dump(gt_user, f)
    with open(PATH_GT_ITEM, 'wb') as f:
        pickle.dump(gt_item, f)
        
    print("Ground Truth saved. Starting isolated workers...\n")
    
    # List of models for testing
    models_to_test = ['Exact KNN', 'Annoy', 'FAISS', 'HNSW']
    results = []

    for model_name in models_to_test:
        print(f"--- Spawning process for {model_name} ---")
        
        # STARTING A SEPARATE PROCESS
        cmd = [sys.executable, __file__, '--worker', '--model_name', model_name]
        
        try:
            # capture_output=True allows you to intercept what the worker prints in print()
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the process output, looking for JSON between markers
            output = proc.stdout
            start_marker = "__RESULT_START__"
            end_marker = "__RESULT_END__"
            
            if start_marker in output and end_marker in output:
                json_str = output.split(start_marker)[1].split(end_marker)[0]
                res = json.loads(json_str)
                if 'error' in res:
                    print(f"Error in {model_name}: {res['error']}")
                else:
                    results.append(res)
                    print(f"Success. Memory: {res.get('Total Memory (MB)', 0):.2f} MB")
            else:
                print(f"Failed to parse output for {model_name}. Raw output:\n{output}")
                print(f"Stderr:\n{proc.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Process for {model_name} crashed!")
            print(e.stderr)

    # Garbage collection
    for p in [PATH_TEST_INDICES, PATH_GT_USER, PATH_GT_ITEM]:
        if os.path.exists(p):
            os.remove(p)

    # Final report
    if results:
        df = pd.DataFrame(results)
        print('\n=== FINAL ISOLATED RESULTS ===')
        # Let's organize the columns for beauty
        cols = ['Algorithm', 'Total Memory (MB)', 'Build Time (User)', 'Build Time (Item)'] 
        # Check which columns actually exist to avoid a KeyError
        available_cols = [c for c in cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in available_cols]
        print(df[available_cols + other_cols].to_string(index=False))
        
        csv_path = os.path.join(RESULTS_DIR, 'knn_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Plotting a graph
        try:
            plot_comparison(df)
            print("Plot saved.")
        except Exception as e:
            print(f"Could not plot results: {e}")
    else:
        print("No results collected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    parser.add_argument('--model_name', type=str, help='Model name to test')
    args = parser.parse_args()

    if args.worker:
        run_worker(args.model_name)
    else:
        run_orchestrator()
