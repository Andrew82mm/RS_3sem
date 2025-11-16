# Data Directory Info

## data/raw/
**Dataset:** MovieLens 32M
**Source:** [GroupLens Research](https://grouplens.org/datasets/movielens/)
**Version:** 32M dataset
**Description:** Raw movie ratings from users

### Files:
- `ratings.csv` - user ratings
- `movies.csv` - movie information

---

## 📁 data/processed/
**Generation date:** 11/16/2025

### Generated files:

#### Vector representations (embeddings)
- `user_embeddings_128d.csv` - user embeddings (CSV)
- `movie_embeddings_128d.csv` - movie embeddings (CSV)
- `user_embeddings_128d.npy` - user embeddings (NumPy, fast loading)
- `movie_embeddings_128d.npy` - movie embeddings (NumPy, fast loading)

#### Auxiliary files
- `user_id_mapping.pkl` - user_id -> matrix index mapping
- `movie_id_mapping.pkl` - movie_id -> matrix index mapping
- `svd_model.pkl` - trained SVD model for fine-tuning

### Dimensions:
- User embeddings: (270,000 users × 128 dimensions)
- Movie embeddings: (45,000 movies × 128 dimensions)

### Model parameters:
- Algorithm: TruncatedSVD
- Embedding size: 128
- Epochs: 50
