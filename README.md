# KNN Implementations Comparison for Recommender Systems

## Project Overview
This is a preliminary version of a course project comparing K-Nearest Neighbors (KNN) implementations - both brute force and approximate nearest neighbors (ANN) - in the context of recommender systems using the MovieLens 32M dataset.

## 🎯 Objectives
- Compare traditional brute-force KNN with modern ANN approaches
- Evaluate both user-based and item-based collaborative filtering
- Benchmark performance across different ANN libraries
- Analyze trade-offs between accuracy and computational efficiency

## 🔧 Implementations

### KNN Approaches
1. **Brute Force KNN**
   - Exact nearest neighbors computation
   - Baseline for accuracy comparison
   - Cosine similarity and Euclidean distance metrics

2. **Approximate Nearest Neighbors (ANN)**
   - **Annoy** (Spotify's ANNOY)
   - **FAISS** (Facebook AI Similarity Search)
   - **HNSW** (Hierarchical Navigable Small World)

### Recommendation Approaches
- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**

## 📁 Project Structure
├── data
│   ├── inf.md
│   ├── processed
│   │   ├── movie_embeddings_128d.csv
│   │   ├── movie_embeddings_128d.npy
│   │   ├── movie_id_mapping.pkl
│   │   ├── svd_model.pkl
│   │   ├── user_embeddings_128d.csv
│   │   ├── user_embeddings_128d.npy
│   │   └── user_id_mapping.pkl
│   └── raw
│       ├── movies.csv
│       └── ratings.csv
├── notebooks
│   └── kursah32m.ipynb
├── README.md
└── tex
    ├── Reportv1.pdf
    └── Reportv1.tex

## Future Work
- Improve RAM usage tracking implementation
- Enhance result stability and reproducibility

