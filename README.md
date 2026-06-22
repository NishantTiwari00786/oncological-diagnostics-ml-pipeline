# Oncological Diagnostics ML Pipeline — Supervised & Unsupervised Methods from Scratch

Eight machine learning algorithms implemented from scratch and applied to the Wisconsin Breast Cancer Diagnostic dataset (UCI) for tumor classification, dimensionality reduction, feature selection, data augmentation, clustering, and anomaly detection. Every model is built with NumPy — no scikit-learn estimators.

**Team:** Nishant Tiwari, Shahd Elmahallawy

**My contributions:** Exploratory data analysis, Decision Tree classifier (entropy/information gain), SVD dimensionality reduction, feature selection via randomization, DBSCAN clustering — all implemented from scratch.

**Dataset:** [Wisconsin Breast Cancer Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 real-valued features from digitized FNA images of breast cell nuclei, binary classification: malignant (212) vs. benign (357).

---

## Phase 1 — Supervised Learning

### Classifiers (10-fold stratified CV, weighted F1)

| Classifier | F1 Score |
|---|---|
| Decision Tree (entropy/info gain, max depth 4) | 0.893 ± 0.045 |
| Gaussian Naive Bayes | **0.931 ± 0.031** |

### SVD Dimensionality Reduction

Train-only factorization (X = USVᵀ) to avoid data leakage, projecting 30 features down to k components.

| SVD Components (k) | Decision Tree F1 | Naive Bayes F1 |
|---|---|---|
| 2 | 0.922 ± 0.041 | 0.910 ± 0.039 |
| 5 | **0.944 ± 0.026** | **0.924 ± 0.030** |
| 10 | 0.944 ± 0.031 | 0.917 ± 0.041 |
| 15 | 0.938 ± 0.029 | 0.897 ± 0.042 |
| 30 (all) | 0.935 ± 0.032 | 0.853 ± 0.037 |

### Feature Selection via Randomization

Permutation importance: shuffle each feature, retrain, measure F1 drop. Run on a 20% stratified subset with 5-fold CV.

**Top features (Naive Bayes):** `area2` (F1 drop = 0.017), `symmetry3` (0.009), `fractal_dimension2` (0.009). Both classifiers independently ranked area-related features highest.

### SMOTE Data Augmentation

Tested at 100%, 200%, 300% oversampling with k=1 and k=5 neighbors. 100% oversampling gave a slight F1 improvement; performance declined beyond 100% as synthetic data introduced boundary noise.

---

## Phase 2 — Unsupervised Learning

### K-means / K-means++

| k | Silhouette (K-means++) |
|---|---|
| 2 | **~0.35** (peak) |
| 3–5 | Steadily decreasing |

Both methods peak at k=2, correctly recovering the true two-class structure without labels.

### DBSCAN

Tested eps ∈ {0.1, 0.2} and min_pts ∈ {5, 10, 15, 20}. At original eps values, nearly all points were classified as noise (silhouette = 0). Scaled eps (×400) produced valid clusters but with low separation scores.

### Spectral Clustering

Unnormalized spectral clustering (Von Luxburg tutorial) with Gaussian similarity at σ ∈ {0.1, 1, 10}.

Best result: **silhouette > 0.6 at σ=10, k=2** — the strongest unsupervised separation across all methods.

### Isolation Forest

100 trees, 256-sample subsamples. Removed top 1–15% anomalies and re-ran K-means++ (k=2).

| Anomalies Removed | Silhouette |
|---|---|
| 1% | **0.341** |
| 5% | 0.336 |
| 10% | 0.332 |
| 15% | 0.334 |

---

## Discussion — What the Two Phases Tell Us Together

**The diagnostic signal is concentrated, not spread across 30 features.** SVD at k=5 actually *beats* the full 30-feature Decision Tree baseline (0.944 vs. 0.893), and permutation importance confirms that area and shape measurements dominate. Most features are redundant — reducing them helps rather than hurts.

**Simple models win when individual features are strong.** Naive Bayes (0.931) outperforms the more expressive Decision Tree (0.893) because each feature independently carries diagnostic signal — exactly the regime where the independence assumption is an advantage, not a liability.

**The two classes are globally separable but not density-separable.** K-means and spectral clustering both recover k=2 from unlabeled data, but DBSCAN fails because there's no empty density gap between malignant and benign in 30-dimensional space. Spectral clustering at σ=10 achieves the best unsupervised result (silhouette > 0.6) because it captures global shape rather than local density.

**More synthetic data isn't always better.** SMOTE at 100% helps; 200–300% hurts by generating points in the ambiguous boundary region. Similarly, Isolation Forest helps at 1% removal but hurts beyond that — aggressive cleaning removes legitimate hard cases, not just outliers.

---

## Algorithms Implemented from Scratch

All implementations use only NumPy. scikit-learn is used solely for metrics, StandardScaler, StratifiedKFold, and clone.

**Supervised:** Decision Tree (entropy/info gain) · Gaussian Naive Bayes · SVD dimensionality reduction · Permutation feature selection · SMOTE

**Unsupervised:** K-means / K-means++ · DBSCAN (BFS neighborhood expansion) · Spectral Clustering (Gaussian similarity → Laplacian → eigenvector embedding) · Isolation Forest (random partitioning, path-length scoring)

## Project Structure

```
oncological-diagnostics-ml-pipeline/
├── oncological_diagnostics_pipeline.ipynb   # Full notebook: EDA → supervised → unsupervised
└── README.md
```

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn (metrics/CV only) · UCI ML Repository API

## Acknowledgements

Course project for CS 235 Data Mining, Fall 2025, UC Riverside. Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
