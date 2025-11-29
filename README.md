# KNN Implementations Comparison for Recommender Systems
Sergienko Andrew B-82 coursework
## 📌 Introduction

This repository contains a modular and reproducible framework for comparing **exact** and **approximate** K-Nearest Neighbors algorithms in the context of **collaborative filtering recommender systems**.

The project includes:

* Data preprocessing and filtering of the MovieLens dataset
* Construction of sparse interaction matrices
* Dimensionality reduction using Truncated SVD
* Generation of user and item embeddings
* Unified implementations of four KNN algorithms:
  **Exact KNN**, **Annoy**, **FAISS**, **HNSW**
* Isolated benchmarking environment for accurate performance measurement
* Hyperparameter tuning for all ANN methods

The goal is to provide a complete and reproducible environment for testing nearest-neighbor search techniques in large-scale recommender systems.

---

## ▶️ How to Install and Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Andrew82mm/RS_3sem.git
cd RS_3sem
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate           # Linux/macOS
venv\Scripts\activate              # Windows
```

### 3. Install all dependencies from `requirements.txt`

This installs:

* Annoy
* FAISS (CPU version if available)
* hnswlib
* scikit-learn
* pandas, numpy
* psutil for memory profiling
* and all other required modules

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ If you are on Windows and FAISS fails to install, use the CPU Windows build:

```bash
pip install faiss-cpu-windows
```

---

### 4. Prepare the dataset

Place MovieLens files into(or use a dataset from the repository): 

```
data/raw/movies.csv  
data/raw/ratings.csv
```

### 5. Generate embeddings (SVD)

```bash
python data.py
```

This step will:

* filter the dataset
* build the user-item CSR matrix
* compute SVD
* generate embeddings
* save processed data into `data/processed/`

---

### 6. Run hyperparameter tuning (optional)

```bash
python run_tuning.py
```

Generated files:

```
tuning/best_params.json
tuning/tuning_results.csv
```

---

### 7. Run the full KNN comparison benchmark

```bash
python run_comparison.py
```

This will:

* build all KNN indices
* run isolated benchmarking workers
* measure memory usage and timing
* compute accuracy against Exact KNN
* save results into `results/`

---

## 📁 Project Structure

```
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
├── knn_comparison
│   ├── annoy.py
│   ├── base.py
│   ├── bench.py
│   ├── data.py
│   ├── exact.py
│   ├── faiss.py
│   ├── hnsw.py
│   ├── __init__.py
│   ├── utils.py
│   └── viz.py
├── notebooks
│   └── kursah32m.ipynb
├── results
│   ├── knn_comparison.png
│   └── knn_results.csv
├── tex
│   ├── Reportv1.pdf
│   ├── Reportv1.tex
├── tuning
│   ├── best_params.json
│   └── tuning_results.csv
│
├── data.py
├── run_comparison.py
├── run_tuning.py
├── README.md
└── requirements.txt
```

---

## 🧭 Recommendations

* Use **HNSW** for real-time systems requiring fast queries and high accuracy.
* Use **FAISS** for large-scale environments or when GPU acceleration is available.
* Use **Annoy** for extremely fast filtering or low-latency tasks where slight accuracy loss is acceptable.
* Use **Exact KNN** only as a baseline for validation and evaluation.

---
