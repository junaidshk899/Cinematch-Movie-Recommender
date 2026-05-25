<div align="center">

# 🎬 CineMatch — Movie Recommender System

### Collaborative Filtering · SVD Matrix Factorization · MovieLens 100K

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Google_Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.4-FF6B6B?style=for-the-badge)](https://surpriselib.com)
[![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)](LICENSE)
[![Internship](https://img.shields.io/badge/Domain-Data_Science_Intern-1F4E79?style=for-the-badge)](https://github.com/junaidasim899)

<br/>

> **Ever wondered how Netflix always knows what you want to watch next?**
> CineMatch is a fully functional movie recommendation engine that uses the same core technique —
> Singular Value Decomposition — to deliver personalized Top-10 film recommendations.

<br/>

```
 Input: User ID  →  SVD Model  →  Predicted Ratings  →  Top-10 Movies 🎥
```

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Implementation](#-implementation)
- [Results](#-results)
- [Screenshots](#-screenshots)
- [Tech Stack](#️-tech-stack)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

**CineMatch** is an end-to-end Movie Recommendation System built as a Data Science Internship project. It applies **Collaborative Filtering** using **SVD (Singular Value Decomposition)** — a matrix factorization technique that learns hidden user preferences and movie characteristics from 100,000 historical ratings.

The system achieves an **RMSE of ~0.93 stars** on held-out test data, meaning predictions are on average less than 1 star away from a user's true rating on a 1–5 scale.

The entire pipeline is delivered as a **fully documented Google Colab Notebook** with step-by-step explanations, rich visualizations, and an interactive recommendation cell.

---

## ✨ Features

- 📥 **Automated Data Download** — Fetches the MovieLens 100K dataset via KaggleHub with zero manual setup
- 🔍 **Exploratory Data Analysis** — Rating distributions, user behavior analysis, most-rated and highest-rated movies
- 🤖 **SVD Collaborative Filtering** — Matrix factorization model that learns latent user and item embeddings
- 📏 **Rigorous Evaluation** — RMSE and MAE metrics on a proper held-out test split
- 🎬 **Top-N Recommendations** — Personalized ranked movie lists for any user in the dataset (1–943)
- 📊 **4 Professional Visualizations** — Charts embedded throughout the notebook for clear interpretation
- 🔁 **Interactive Section** — Change one variable to instantly get recommendations for any user
- 📄 **Full Documentation** — Every cell has a title, description, code comments, and insight callouts

---

## 📁 Project Structure

```
cinematch-movie-recommender/
│
├── 📓 Movie_Recommender_System.ipynb   # Main Colab Notebook (complete implementation)
├── 📄 README.md                         # This file
├── 📑 Internship_Report.pdf             # Full project report (PDF)
├── 📝 Internship_Report.docx            # Full project report (Word)
└── 📜 LICENSE                           # MIT License
```

---

## 📊 Dataset

This project uses the **[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)** dataset published by GroupLens Research at the University of Minnesota.

| Property | Value |
|---|---|
| Total Ratings | 100,000 |
| Unique Users | 943 |
| Unique Movies | 1,682 |
| Rating Scale | 1–5 stars (integer) |
| Collection Period | September 1997 – April 1998 |
| Matrix Sparsity | ~93.7% |

The dataset is **automatically downloaded** via `kagglehub` — no manual Kaggle API configuration required.

---

## ⚙️ Installation

### Option A — Google Colab (Recommended)

1. Click the badge below to open directly in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

2. Upload `Movie_Recommender_System.ipynb`
3. Click **Runtime → Run All**
4. Done! ✅

### Option B — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/junaidasim899/cinematch-movie-recommender.git
cd cinematch-movie-recommender

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install scikit-surprise kagglehub pandas numpy matplotlib seaborn jupyter

# 4. Launch the notebook
jupyter notebook Movie_Recommender_System.ipynb
```

---

## 🚀 Quick Start

```python
# After running all cells, get recommendations for any user:

USER_ID = 42      # ← Change this (valid range: 1–943)
TOP_N   = 10      # ← How many recommendations?

recs = get_top_n_recommendations(user_id=USER_ID, n=TOP_N)
print(recs[['title', 'PredictedRating']])
```

**Sample Output:**

```
                              title  PredictedRating
1               Schindler's List (1993)            4.62
2                    Casablanca (1942)            4.58
3                    High Noon (1952)            4.55
4                  Rear Window (1954)            4.51
5                The Godfather (1972)            4.49
6              The Usual Suspects (1995)          4.46
7        The Shawshank Redemption (1994)          4.44
8                12 Angry Men (1957)             4.41
9                      Vertigo (1958)            4.39
10                  Notorious (1946)             4.37
```

---

## 🔧 Implementation

The notebook is organized into **10 clearly labeled steps**:

### Step 1 — Install Libraries
```python
!pip install scikit-surprise kagglehub --quiet
```

### Step 2 — Import Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from surprise import Dataset, SVD, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
```

### Step 3 — Load the Dataset
```python
path = kagglehub.dataset_download("bromotdi/movielens-100k-dataset")

ratings = pd.read_csv(os.path.join(path, 'ml-100k', 'u.data'),
                      sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

movies = pd.read_csv(os.path.join(path, 'ml-100k', 'u.item'),
                     sep='|', encoding='latin-1',
                     header=None, usecols=[0, 1], names=['movieId', 'title'])
```

### Step 4 — Exploratory Data Analysis
```python
# Sparsity of the user-movie matrix
sparsity = 1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
print(f"Matrix Sparsity: {sparsity:.2%}")  # → 93.70%

# Rating distribution
print(ratings['rating'].value_counts().sort_index())
# 1: 6110 | 2: 11370 | 3: 27145 | 4: 34174 | 5: 21201
```

### Step 5 — Data Preprocessing
```python
reader   = Reader(rating_scale=(1, 5))
data     = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
full_trainset = data.build_full_trainset()
```

### Step 6 — Train the SVD Model
```python
svd_model = SVD(n_factors=100, n_epochs=20,
                lr_all=0.005, reg_all=0.02, random_state=42)
svd_model.fit(trainset)
```

### Step 7 — Evaluate Performance
```python
predictions = svd_model.test(testset)
print(f"RMSE: {rmse(predictions, verbose=False):.4f}")   # → 0.9347
print(f"MAE:  {mae(predictions,  verbose=False):.4f}")   # → 0.7384
```

### Step 8–10 — Generate & Visualize Recommendations
```python
def get_top_n_recommendations(user_id, n=10):
    all_movie_ids = ratings['movieId'].unique()
    already_rated = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    unseen = [mid for mid in all_movie_ids if mid not in already_rated]

    preds = [(mid, svd_model.predict(user_id, mid).est) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(preds[:n], columns=['movieId', 'PredictedRating'])
    result = result.merge(movies, on='movieId')
    result['PredictedRating'] = result['PredictedRating'].round(2)
    return result
```

---

## 📈 Results

| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | 0.9347 | Average squared-weighted prediction error |
| **MAE** | 0.7384 | Average absolute prediction error (~0.74 stars) |
| **Training Set** | 80,000 ratings | 80% of the full dataset |
| **Test Set** | 20,000 ratings | Held-out for unbiased evaluation |

> 💡 An RMSE of 0.93 is competitive with published SVD baselines on MovieLens 100K.

---

## 🖼️ Screenshots

### Rating Distribution
> Most users rate movies 4 stars — confirming a positive selection bias (users rate what they already enjoy).

### Model Evaluation
> Actual vs Predicted scatter plot shows predictions cluster tightly along the diagonal (perfect prediction line).

### Top-10 Recommendations for User #42
> The model surfaces critically acclaimed classics with predicted ratings above 4.37 stars.

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.12 | Primary language |
| **pandas** | 2.x | Data manipulation |
| **numpy** | 1.26 | Numerical computing |
| **matplotlib** | 3.x | Charting |
| **seaborn** | 0.13 | Statistical visualization |
| **scikit-surprise** | 1.1.4 | SVD & collaborative filtering |
| **kagglehub** | 1.0 | Dataset download |
| **Google Colab** | Latest | Development environment |

---

## 🔬 How SVD Works

SVD decomposes the sparse **User × Movie** rating matrix **R** into two dense factor matrices:

```
R  ≈  U  ×  Vᵀ

Where:
  U  = User matrix   (943 users  × 100 latent factors)
  Vᵀ = Movie matrix  (100 factors × 1682 movies)
```

Each **latent factor** captures a hidden concept — e.g., "prefers drama over comedy", "enjoys classic films". The predicted rating for user *u* on movie *i* is:

```
r̂(u, i) = μ + bᵤ + bᵢ + qᵢᵀ · pᵤ
```

Where `μ` is the global mean, `bᵤ` and `bᵢ` are user/item biases, and `qᵢᵀ · pᵤ` is the dot product of their latent vectors.

---

## 🚀 Future Improvements

- [ ] **Hyperparameter Tuning** — `GridSearchCV` for optimal `n_factors`, `n_epochs`, `lr_all`, `reg_all`
- [ ] **Algorithm Comparison** — Benchmark SVD vs `KNNBasic`, `NMF`, `SlopeOne`
- [ ] **Hybrid Model** — Combine collaborative + content-based filtering to solve cold start
- [ ] **Larger Dataset** — Scale to MovieLens 1M or 20M
- [ ] **Web App** — Deploy with Streamlit or Gradio
- [ ] **Cross-Validation** — Replace single split with k-fold CV for robust estimates

---

## 👤 Author

<div align="center">

| | |
|---|---|
| **Name** | Muhammad Junaid Asim |
| **Role** | Data Science Intern |
| **Email** | [Junaidasim899@gmail.com](mailto:Junaidasim899@gmail.com) |
| **Phone** | +92 3107974444 |
| **Date** | 24 May, 2026 |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2026 Muhammad Junaid Asim
Permission is granted to use, copy, modify, and distribute this software freely.
```

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ using scikit-surprise, pandas, and matplotlib*

</div>
