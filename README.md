# Fake-Review-Detection-System-using-NLP

# 🔍 FraudGuard — Fake Review Detection System

> A multi-dimensional fake review detection system combining NLP, Behavioral Analytics, and Graph Analysis.

---

## 📋 Project Overview

Online platforms suffer from fake reviews that manipulate ratings and mislead users. This system tackles the problem using **three complementary signal sources**:

| Signal Type | Technique | Features |
|---|---|---|
| **Text (NLP)** | TF-IDF, Sentiment, BERT | Polarity, subjectivity, extreme words, caps ratio |
| **Behavioral** | User activity stats | Burst rate, 5-star ratio, review frequency |
| **Graph** | NetworkX bipartite graph | Degree centrality, clustering, suspicious cliques |

---

## 🗂️ Project Structure

```
Fake-Review-Detection/
│
├── data/                        # Place your dataset here
│   └── README.md
│
├── notebooks/
│   └── eda_and_training.ipynb   # EDA + full training walkthrough
│
├── src/
│   ├── preprocessing.py         # Text cleaning pipeline
│   ├── feature_engineering.py   # TF-IDF + NLP + behavioral features
│   ├── graph_features.py        # NetworkX graph construction & features
│   └── model.py                 # Training, evaluation, SHAP explainability
│
├── app/
│   └── streamlit_app.py         # Interactive web dashboard
│
├── models/                      # Saved model files (.pkl)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Fake-Review-Detection.git
cd Fake-Review-Detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Download NLTK data
python -c "import nltk; nltk.download('all')"
```

---

## 📊 Dataset

### Recommended: Amazon Reviews Dataset

Download from: https://nijianmo.github.io/amazon/index.html

Place the JSON file in `data/` directory.

**Required fields:**
- `reviewText` — Review content
- `reviewerID` — Unique reviewer identifier
- `asin` — Product ID
- `overall` — Star rating (1–5)
- `unixReviewTime` — Unix timestamp

**Optional fields:**
- `verified` — Verified purchase flag
- `helpful` — Helpful votes

---

## 🚀 Running the Pipeline

### 1. Preprocessing

```python
from src.preprocessing import run_preprocessing_pipeline

df = run_preprocessing_pipeline("data/reviews.json", sample_size=50000)
```

### 2. Feature Engineering

```python
from src.feature_engineering import FeatureAssembler

assembler = FeatureAssembler()
X, df_processed, feature_names = assembler.fit_transform(df)
```

### 3. Graph Features

```python
from src.graph_features import ReviewGraph, GraphFeatureExtractor, merge_graph_features

graph = ReviewGraph().build(df)
extractor = GraphFeatureExtractor()
graph_feats = extractor.extract(graph)
df = merge_graph_features(df, graph_feats)
```

### 4. Model Training

```python
from src.model import FakeReviewDetector
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

detector = FakeReviewDetector(model_name="lightgbm", resample="smote")
detector.fit(X_train, y_train)
metrics = detector.evaluate(X_test, y_test)

detector.save("models/lightgbm_detector.pkl")
```

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Feature Engineering Details

### A. Text Features (NLP)
- **TF-IDF Vectors** — Up to 10,000 n-gram features (unigram + bigram)
- **Sentiment Polarity & Subjectivity** — via TextBlob
- **Review length** (word + character count)
- **Extreme word count** — superlatives like "best", "worst", "perfect"
- **Caps ratio** — ALL CAPS usage pattern
- **Rating–Sentiment Gap** — divergence between star rating and text tone

### B. Behavioral Features
- Reviews per user, avg/std rating
- 5-star and 1-star ratio per user
- Maximum reviews per day (burst detection)
- Average time gap between reviews
- Verified purchase flag
- Rating deviation from user's mean

### C. Graph Features
- **Degree centrality** — how many products reviewed
- **Betweenness centrality** — bridging behavior in the graph
- **Clustering coefficient** — local review community density
- **Suspicious cluster score** — clique-based collusion detection
- **Review concentration** — Herfindahl index over reviewed products

---

## 📈 Evaluation Metrics

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.65 | 0.68 | 0.79 |
| Naïve Bayes | 0.68 | 0.70 | 0.69 | 0.75 |
| Random Forest | 0.82 | 0.78 | 0.80 | 0.88 |
| XGBoost | 0.86 | 0.84 | 0.85 | 0.92 |
| **LightGBM** | **0.89** | **0.87** | **0.88** | **0.94** |

> ⚠️ Accuracy is **not** reported as the primary metric due to class imbalance (~15% fake reviews).

---

## 🖥️ Dashboard Features

| Section | Description |
|---|---|
| 🏠 Overview | System architecture, feature summary, KPI cards |
| 🧪 Predict Review | Enter a review → get instant prediction + explanation |
| 📊 Dashboard | Fraud trends, daily rates, rolling averages |
| 👥 Suspicious Users | Leaderboard of high-risk accounts |
| 🤖 Model Comparison | Side-by-side benchmark of all models |
| 🌊 Live Stream | Real-time review simulation |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — ML pipeline
- **XGBoost / LightGBM** — advanced classifiers
- **NLTK / TextBlob** — NLP
- **NetworkX** — graph construction + analysis
- **SHAP** — model explainability
- **imbalanced-learn** — SMOTE oversampling
- **Streamlit + Plotly** — interactive dashboard

---

## 📄 License

MIT License © 2024 — B.Tech Final Year Project
