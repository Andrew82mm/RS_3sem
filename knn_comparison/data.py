import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path('data/processed')

def load_embeddings():
    user_emb = np.load(DATA_DIR / 'user_embeddings_128d.npy')
    item_emb = np.load(DATA_DIR / 'movie_embeddings_128d.npy')

    with open(DATA_DIR / 'user_id_mapping.pkl', 'rb') as f:
        user_id_to_idx = pickle.load(f)
    with open(DATA_DIR / 'movie_id_mapping.pkl', 'rb') as f:
        item_id_to_idx = pickle.load(f)

    idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}

    return (user_emb, item_emb,
            user_id_to_idx, item_id_to_idx,
            idx_to_user_id, idx_to_item_id)
