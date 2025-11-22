import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle

print("="*60)
print("CREATING EMBEDINGS FOR A RECOMMENDATION SYSTEM")
print("="*60)

# === 1. LOADING DATA ===
print("\n[1/5] Loading data...")
start_time = time.time()

movies = pd.read_csv('./data/raw/movies.csv')
ratings = pd.read_csv('./data/raw/ratings.csv')

# Data clearing
movies.drop(['genres'], axis=1, inplace=True)
ratings.drop(['timestamp'], axis=1, inplace=True)

print(f"  ✓ Uploaded {len(movies)} films")
print(f"  ✓ Uploaded {len(ratings)} ratings")
print(f"  ✓ Time: {time.time() - start_time:.2f}s")

# === 2. PREPARATION OF MATRICES ===
print("\n[2/5] Preparing sparse matrices...")
start_time = time.time()

# Filtering active users and popular movies
user_votes = ratings.groupby('userId')['rating'].count()
movie_votes = ratings.groupby('movieId')['rating'].count()
active_users = user_votes[user_votes > 50].index
popular_movies = movie_votes[movie_votes > 10].index

print(f"  ✓ Active userd: {len(active_users)}")
print(f"  ✓ Popular films: {len(popular_movies)}")

# Filtering ratings
ratings_filtered = ratings[
    ratings['userId'].isin(active_users) & 
    ratings['movieId'].isin(popular_movies)
].copy()

print(f"  ✓ Ratings after filtering: {len(ratings_filtered):,}")

# We create a mapping (remove the "gaps" in the matrix 1,3,7... -> 1,2,3...)
user_ids = sorted(ratings_filtered['userId'].unique())
movie_ids = sorted(ratings_filtered['movieId'].unique())

user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_id_to_idx)
ratings_filtered['movie_idx'] = ratings_filtered['movieId'].map(movie_id_to_idx)

# Create a sparse matrix DIRECTLY from data
n_users = len(user_ids)
n_movies = len(movie_ids)

sparse_matrix = csr_matrix(
    (ratings_filtered['rating'].values,
     (ratings_filtered['user_idx'].values, ratings_filtered['movie_idx'].values)),
    shape=(n_users, n_movies)
)

density = sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
print(f"  ✓ Matrix size: {n_users} users × {n_movies} films")
print(f"  ✓ Matrix filling: {density*100:.2f}%")
print(f"  ✓ Known ratings: {sparse_matrix.nnz:,}")
print(f"  ✓ Time: {time.time() - start_time:.2f}s")

# === 3. CREATING EMBEDDINGS ===
print("\n[3/5] Creating embeddings using Matrix Factorization (SVD)...")
print("This may take a few minutes...")

embedding_dim = 128
n_epochs = 20  # can be increased for better quality

start_time = time.time()

# Using TruncatedSVD for Sparse Matrices
svd = TruncatedSVD(
    n_components=embedding_dim,
    n_iter=n_epochs,
    random_state=42
)

# Training on user-movie data
print(f"  → SVD training with {embedding_dim} components...")
user_embeddings = svd.fit_transform(sparse_matrix)

# Obtain movie embeddings from SVD components
movie_embeddings = svd.components_.T

print(f" ✓ User embeddings: {user_embeddings.shape}")
print(f" ✓ Movie embeddings: {movie_embeddings.shape}")
print(f" ✓ Explained variance: {svd.explained_variance_ratio_.sum()*100:.2f}%")
print(f" ✓ Training time: {time.time() - start_time:.2f}s")

# === 4. CREATING A DATAFRAME WITH EMBEDDINGS ===
print("\n[4/5] Creating a DataFrame with embeddings...")
start_time = time.time()

# DataFrame for users
user_embedding_columns = [f'emb_{i}' for i in range(embedding_dim)]
user_embeddings_df = pd.DataFrame(
    user_embeddings,
    index=user_ids, # use the saved list of IDs
    columns=user_embedding_columns
)
user_embeddings_df.index.name = 'userId'

# DataFrame for films
movie_embeddings_df = pd.DataFrame(
    movie_embeddings,
    index=movie_ids,  # use the saved list of IDs
    columns=user_embedding_columns
)
movie_embeddings_df.index.name = 'movieId'

# Adding movie titles for convenience
movie_info = movies.set_index('movieId')['title']
movie_embeddings_df = movie_embeddings_df.join(movie_info, how='left')

print(f"  ✓ User embeddings DataFrame: {user_embeddings_df.shape}")
print(f"  ✓ Movie embeddings DataFrame: {movie_embeddings_df.shape}")
print(f"  ✓ Time: {time.time() - start_time:.2f}s")

# === 5. SAVING RESULTS ===
print("\n[5/5] Saving embeddings...")
start_time = time.time()

# Save in CSV
user_embeddings_df.to_csv('./data/processed/user_embeddings_128d.csv')
movie_embeddings_df.to_csv('./data/processed/movie_embeddings_128d.csv')

# We also save in NumPy format (loads faster)
np.save('./data/processed/user_embeddings_128d.npy', user_embeddings)
np.save('./data/processed/movie_embeddings_128d.npy', movie_embeddings)

# Save the mapping ID -> index for quick access
with open('./data/processed/user_id_mapping.pkl', 'wb') as f:
    pickle.dump(user_id_to_idx, f)

with open('./data/processed/movie_id_mapping.pkl', 'wb') as f:
    pickle.dump(movie_id_to_idx, f)

# We save the SVD model for further training.
with open('./data/processed/svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)

print(f"  ✓ user_embeddings_128d.csv ({user_embeddings_df.shape})")
print(f"  ✓ movie_embeddings_128d.csv ({movie_embeddings_df.shape})")
print(f"  ✓ user_embeddings_128d.npy")
print(f"  ✓ movie_embeddings_128d.npy")
print(f"  ✓ user_id_mapping.pkl")
print(f"  ✓ movie_id_mapping.pkl")
print(f"  ✓ svd_model.pkl")
print(f"  ✓ Time: {time.time() - start_time:.2f}s")

# Size statistics
print("\n" + "="*60)
print("STATISTICS")
print("="*60)

import os

def get_file_size(filename):
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        return size_mb
    return 0

print("\nFile size:")
print(f"  user_embeddings_128d.csv: {get_file_size('./data/processed/user_embeddings_128d.csv'):.2f} MB")
print(f"  movie_embeddings_128d.csv: {get_file_size('./data/processed/movie_embeddings_128d.csv'):.2f} MB")
print(f"  user_embeddings_128d.npy: {get_file_size('./data/processed/user_embeddings_128d.npy'):.2f} MB")
print(f"  movie_embeddings_128d.npy: {get_file_size('./data/processed/movie_embeddings_128d.npy'):.2f} MB")

total_embeddings_size = (get_file_size('./data/processed/user_embeddings_128d.npy') + 
                         get_file_size('./data/processed/movie_embeddings_128d.npy'))

print(f"\nTotal embedding size (npy): {total_embeddings_size: .2f} MB")

# Compare with the original matrix
original_size = sparse_matrix.shape[0] * sparse_matrix.shape[1] * 4 / (1024 * 1024)
compression_ratio = original_size / total_embeddings_size

print(f"Original matrix size (if dense): {original_size: .2f} MB")
print(f"Compression ratio: {compression_ratio: .1f}x")

print("\n" + "="*60)
print("✓ ALL DONE!")
print("="*60)
