# Pattern Recognition - Final Exam Solutions

## Quick Reference Guide

Use this guide to quickly find the right solution file for each exam topic.

---

## Example Exam Problems & Solutions

### Example 1: k-NN Classification
**Problem:**
```
Given training data:
  Class 1: (1,1), (1,2), (2,1)
  Class 2: (5,5), (6,5), (6,6)

Classify the query point (3,3) using k=3.
```
**→ Run:** `knn_solution.py`

---

### Example 2: Perceptron Weight Update
**Problem:**
```
Given data points with labels (+1 or -1):
  x₁=(1,1), y₁=+1
  x₂=(2,2), y₂=+1
  x₃=(3,3), y₃=-1

Initial weights w=[0,0], b=0, learning rate η=1.
Perform 2 iterations of perceptron learning.
```
**→ Run:** `perceptron_solution.py`

---

### Example 3: K-Means Clustering
**Problem:**
```
Given points: (1,1), (2,1), (5,5), (6,5)
Initial centroids: μ₁=(1,1), μ₂=(5,5)

Perform 2 iterations of k-means clustering (k=2).
Show assignment and update steps.
```
**→ Run:** `kmeans_solution.py`

---

### Example 4: Hierarchical Clustering
**Problem:**
```
Given points: A(1,1), B(1,2), C(5,5), D(6,5)

Perform agglomerative clustering using single linkage.
Show the distance matrix and merge sequence.
```
**→ Run:** `hierarchical_clustering_solution.py`

---

### Example 5: EM Algorithm
**Problem:**
```
Given 1D data: 1, 2, 5, 6
Initial GMM parameters: μ₁=1, μ₂=6, σ₁=σ₂=1, π₁=π₂=0.5

Perform one iteration of EM algorithm (E-step and M-step).
Calculate responsibilities and update parameters.
```
**→ Run:** `em_algorithm_solution.py`

---

### Example 6: Parzen Window
**Problem:**
```
Given data: 1, 2, 3, 5, 6
Using Gaussian kernel with bandwidth h=1

Estimate the density p(x) at x=2.5.
```
**→ Run:** `parzen_window_solution.py`

---

### Example 7: Performance Metrics
**Problem:**
```
Given true labels:    [1, 1, 1, 1, 0, 0, 0, 0]
Given predictions:    [1, 1, 0, 0, 0, 0, 1, 1]

Calculate:
  a) Confusion matrix (TP, TN, FP, FN)
  b) Accuracy
  c) Precision
  d) Recall
  e) F1-score
```
**→ Run:** `performance_metrics_solution.py`

---

### Example 8: ROC/AUC
**Problem:**
```
Given classifier scores and true labels:
  Sample 1: score=0.9, true=1
  Sample 2: score=0.7, true=1
  Sample 3: score=0.5, true=0
  Sample 4: score=0.3, true=0

Calculate TPR and FPR at threshold=0.6.
Sketch the ROC curve and estimate AUC.
```
**→ Run:** `roc_auc_solution.py`

---

### Example 9: AdaBoost
**Problem:**
```
Given 5 samples with equal initial weights (w=0.2 each):
  x₁=1, y₁=+1
  x₂=2, y₂=+1
  x₃=3, y₃=-1
  x₄=6, y₄=-1
  x₅=7, y₅=-1

Weak classifier h₁(x) = sign(x - 4) makes 1 error.
Calculate:
  a) Weighted error ε₁
  b) Classifier weight α₁
  c) Updated sample weights
```
**→ Run:** `adaboost_solution.py`

---

### Example 10: Random Forest / Bagging
**Problem:**
```
Dataset has 10 samples.

a) Create a bootstrap sample (sample with replacement).
b) What is the probability that a sample is NOT selected?
c) What are out-of-bag (OOB) samples?
d) If d=4 features, how many features should be considered
   at each split for classification?
```
**→ Run:** `random_forest_solution.py`

---

### Example 11.5: Classification Error Estimation
**Problem:**
```
A classifier is evaluated on a test set of 100 samples.
60 samples are correctly classified.

a) Calculate the error rate
b) Calculate the 95% confidence interval
c) If we use 10 bootstrap iterations with average OOB error = 0.25
   and training error = 0.10, calculate the .632 bootstrap estimate
```
**→ Run:** `error_estimation_solution.py`

---

### Example 12: Cross-Validation
**Problem:**
```
Dataset: 12 samples
Use k=4 fold cross-validation.

a) How many samples in each fold?
b) How many samples used for training in each iteration?
c) If fold accuracies are: 0.8, 0.9, 0.7, 0.85
   What is the average accuracy?
```
**→ Run:** `cross_validation_solution.py`

---

### Example 0: Decision Tree
**Problem:**
```
Given training data:
  Sample 1: x₁=1, x₂=1, Class=0
  Sample 2: x₁=1, x₂=2, Class=0
  Sample 3: x₁=2, x₂=1, Class=0
  Sample 4: x₁=4, x₂=4, Class=1
  Sample 5: x₁=5, x₂=5, Class=1
  Sample 6: x₁=5, x₂=6, Class=1

Build a decision tree using Information Gain (Entropy).
Find the best split at the root node.
```
**→ Run:** `decision_tree_solution.py`

---

### Example 0.5: Linear Discriminant Function (LDF)
**Problem:**
```
Given two classes with means μ₁=(2,2) and μ₂=(6,6),
and shared covariance matrix Σ = [[1,0],[0,1]].

a) Derive the Linear Discriminant Function g(x) = w^T*x + w_0
b) Find the decision boundary
c) Classify the query point (4,4)
```
**→ Run:** `ldf_solution.py`

---

## Chapter 3: Classification & Clustering

### 0. Decision Tree
**File:** `decision_tree_solution.py`

**When to use:**
- Building decision trees from data
- Information Gain calculation (Entropy)
- Gini Impurity calculation
- Finding best split points

**Key Formulas:**
```
Entropy: H(D) = -Σ p_i × log₂(p_i)
Gini: Gini(D) = 1 - Σ p_i²
Information Gain: IG(D,A) = H(D) - Σ (|D_v|/|D|) × H(D_v)
```

**Run:**
```bash
python decision_tree_solution.py
```

---

### 1. Linear Discriminant Function (LDF)
**File:** `ldf_solution.py`

**When to use:**
- Linear decision boundary classification
- Gaussian classes with equal covariance
- Multi-class one-vs-rest classification
- Finding weight vector w and bias w_0

**Key Formulas:**
```
Decision function: g(x) = w^T * x + w_0
For Gaussian + equal covariance:
  w = Σ⁻¹(μ₁ - μ₂)
  w_0 = -½(μ₁ + μ₂)^T * Σ⁻¹(μ₁ - μ₂)
Decision: sign(g(x))
```

**Run:**
```bash
python ldf_solution.py
```

---

### 1. k-Nearest Neighbors (k-NN)
**File:** `knn_solution.py`

**When to use:**
- Questions about distance calculation (Euclidean)
- Classifying a query point using nearest neighbors
- Effect of different k values on classification
- Majority voting for class decision

**Key Formulas:**
```
d(x, x_i) = sqrt(sum((x_j - x_ij)^2))
Classification = majority vote among k neighbors
```

**Run:**
```bash
python knn_solution.py
```

---

### 2. Perceptron Algorithm
**File:** `perceptron_solution.py`

**When to use:**
- Questions about linear classifier training
- Weight updates for misclassified points
- Finding decision boundary from weights
- Convergence of perceptron

**Key Formulas:**
```
Decision: y_pred = sign(w·x + b)
Update:   w(k+1) = w(k) + η * y_i * x_i  (for misclassified)
          b(k+1) = b(k) + η * y_i
```

**Run:**
```bash
python perceptron_solution.py
```

---

### 3. K-Means Clustering
**File:** `kmeans_solution.py`

**When to use:**
- Unsupervised clustering problems
- Centroid initialization and updates
- Assignment and update steps
- WCSS (Within-Cluster Sum of Squares) calculation

**Key Formulas:**
```
Assignment: c_i = argmin_j ||x_i - μ_j||^2
Update:     μ_j = (1/n_j) * sum(x_i) for all x_i in cluster j
```

**Run:**
```bash
python kmeans_solution.py
```

---

### 4. Hierarchical Clustering
**File:** `hierarchical_clustering_solution.py`

**When to use:**
- Agglomerative (bottom-up) clustering
- Linkage methods: single, complete, average
- Dendrogram interpretation
- Distance matrix updates

**Key Formulas:**
```
Single Linkage:   d(C_i, C_j) = min{d(x,y) : x∈C_i, y∈C_j}
Complete Linkage: d(C_i, C_j) = max{d(x,y) : x∈C_i, y∈C_j}
Average Linkage:  d(C_i, C_j) = average{d(x,y) : x∈C_i, y∈C_j}
```

**Run:**
```bash
python hierarchical_clustering_solution.py
```

---

### 5. EM Algorithm (Gaussian Mixture Model)
**File:** `em_algorithm_solution.py`

**When to use:**
- Soft clustering with probability assignments
- Gaussian Mixture Models (GMM)
- E-step and M-step calculations
- Responsibilities (γ) calculation

**Key Formulas:**
```
E-step: γ(z_nk) = π_k * N(x_n | μ_k, Σ_k) / Σ_j π_j * N(x_n | μ_j, Σ_j)
M-step: μ_k = (1/N_k) * Σ_n γ(z_nk) * x_n
        Σ_k = (1/N_k) * Σ_n γ(z_nk) * (x_n - μ_k)(x_n - μ_k)^T
        π_k = N_k / N
```

**Run:**
```bash
python em_algorithm_solution.py
```

---

### 6. Parzen Window (Kernel Density Estimation)
**File:** `parzen_window_solution.py`

**When to use:**
- Non-parametric density estimation
- Kernel functions (Gaussian, uniform)
- Effect of bandwidth (h) on smoothness
- Classification using density estimates

**Key Formulas:**
```
p(x) = (1/n) * (1/h^d) * Σ K((x - x_i) / h)

Gaussian Kernel: K(u) = (1/√(2π)) * exp(-u²/2)
```

**Run:**
```bash
python parzen_window_solution.py
```

---

## Chapter 4: Ensemble & Evaluation

### 7. Performance Metrics
**File:** `performance_metrics_solution.py`

**When to use:**
- Confusion matrix calculation
- Accuracy, Precision, Recall, F1-score
- Specificity, Sensitivity
- Macro vs Micro averaging (multi-class)

**Key Formulas:**
```
Accuracy    = (TP + TN) / (TP + TN + FP + FN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)  (also Sensitivity, TPR)
Specificity = TN / (TN + FP)
F1-score    = 2 * (Precision * Recall) / (Precision + Recall)
```

**Run:**
```bash
python performance_metrics_solution.py
```

---

### 8. ROC Curve and AUC
**File:** `roc_auc_solution.py`

**When to use:**
- Plotting ROC curve (TPR vs FPR)
- Calculating AUC using trapezoidal rule
- Comparing classifiers
- Threshold selection

**Key Formulas:**
```
TPR (Sensitivity) = TP / (TP + FN)
FPR (1-Specificity) = FP / (FP + TN)
AUC = area under ROC curve (trapezoidal rule)
```

**Run:**
```bash
python roc_auc_solution.py
```

---

### 9. AdaBoost Algorithm
**File:** `adaboost_solution.py`

**When to use:**
- Ensemble learning with boosting
- Weighted error calculation
- Classifier weight (α) calculation
- Sample weight updates

**Key Formulas:**
```
Weighted Error: ε_t = Σ w_i * [y_i ≠ h_t(x_i)]
Classifier Weight: α_t = 0.5 * ln((1-ε_t)/ε_t)
Weight Update: w_i ← w_i * exp(-α_t * y_i * h_t(x_i))
Final: H(x) = sign(Σ α_t * h_t(x))
```

**Run:**
```bash
python adaboost_solution.py
```

---

### 10. Random Forest / Bagging
**File:** `random_forest_solution.py`

**When to use:**
- Bootstrap sampling (sampling with replacement)
- Bagging (Bootstrap Aggregating)
- Random feature selection at each split
- Out-of-Bag (OOB) error estimation

**Key Formulas:**
```
Bootstrap: Sample n points with replacement
P(in bootstrap) ≈ 1 - e^(-1) ≈ 0.632
P(OOB) ≈ e^(-1) ≈ 0.368
Features per split: m = √d (classification) or d/3 (regression)
```

**Run:**
```bash
python random_forest_solution.py
```

---

### 12. K-Fold Cross-Validation
**File:** `cross_validation_solution.py`

**When to use:**
- Model evaluation methodology
- Splitting data into k folds
- Calculating average performance
- Stratified k-fold for imbalanced data

**Key Formulas:**
```
Average Accuracy = (1/k) * Σ Accuracy_i
Training size per fold = n * (k-1)/k
Test size per fold = n/k
```

**Run:**
```bash
python cross_validation_solution.py
```

---

### 13. Classification Error Estimation
**File:** `error_estimation_solution.py`

**When to use:**
- Holdout method
- Bootstrap error estimation
- .632 Bootstrap
- Confidence intervals for error rate

**Key Formulas:**
```
Holdout Error: E = errors / n_test
Bootstrap: Average OOB error over B iterations
.632 Bootstrap: E_.632 = 0.368×E_train + 0.632×E_boot
95% CI: E ± 1.96 × sqrt(E(1-E)/n)
```

**Run:**
```bash
python error_estimation_solution.py
```

---

## Quick Topic Lookup

| If the question asks about... | Run this file |
|------------------------------|---------------|
| Linear decision boundary, w^T*x + w_0 | `ldf_solution.py` |
| Distance calculation, nearest neighbors | `knn_solution.py` |
| Linear classifier weight updates | `perceptron_solution.py` |
| Centroids, cluster assignment | `kmeans_solution.py` |
| Dendrogram, merging clusters | `hierarchical_clustering_solution.py` |
| Soft clustering, responsibilities | `em_algorithm_solution.py` |
| Kernel density, bandwidth | `parzen_window_solution.py` |
| TP, TN, FP, FN, Precision, Recall | `performance_metrics_solution.py` |
| ROC curve, AUC, threshold | `roc_auc_solution.py` |
| Boosting, weak classifiers, α | `adaboost_solution.py` |
| Bootstrap, OOB, random features | `random_forest_solution.py` |
| Folds, average accuracy, stratified | `cross_validation_solution.py` |

---

## Quick Problem Matching

| Keywords in Problem | Solution File |
|---------------------|---------------|
| "linear discriminant", "w^T*x + w_0", "decision boundary", "LDF" | `ldf_solution.py` |
| "classify using nearest", "k neighbors", "Euclidean distance" | `knn_solution.py` |
| "weight update", "misclassified", "sign(w·x+b)" | `perceptron_solution.py` |
| "centroid", "cluster assignment", "WCSS" | `kmeans_solution.py` |
| "dendrogram", "single linkage", "merge clusters" | `hierarchical_clustering_solution.py` |
| "E-step", "M-step", "responsibility", "GMM" | `em_algorithm_solution.py` |
| "kernel", "bandwidth h", "density estimate" | `parzen_window_solution.py` |
| "confusion matrix", "precision", "recall", "F1" | `performance_metrics_solution.py` |
| "TPR", "FPR", "ROC", "AUC", "threshold" | `roc_auc_solution.py` |
| "boosting", "α_t", "weighted error", "weak classifier" | `adaboost_solution.py` |
| "bootstrap", "OOB", "bagging", "random forest" | `random_forest_solution.py` |
| "k-fold", "folds", "cross-validation", "average accuracy" | `cross_validation_solution.py` |

---

## Midterm Solutions (Reference)

Located in: `/midterm/`

| File | Topic | Example Problem |
|------|-------|-----------------|
| `ml_vs_map_comparison.py` | ML vs MAP Estimation | Given data D={4,6,8}, σ²=4, prior μ~N(10,4). Find μ_ML and μ_MAP |
| `fld_solution.py` | Fisher Linear Discriminant | Find projection vector v = S_W⁻¹(μ₁-μ₂) for two classes |
| `glcm_solution.py` | GLCM | Calculate GLCM for 5×5 image at d=1, angles 0°, 45°, 90°, 135° |
| `bayesian_classifier_solution.py` | Bayesian Classifier | Given two Gaussians, find decision boundary for minimum error |
| `pca_solution.py` | PCA | Find first principal component and project data onto 1D |
| `map_estimator_solution.py` | MAP Derivation | Derive MAP estimator for Gaussian mean with Gaussian prior |
| `sfs_solution.py` | Sequential Forward Selection | Select best 2 features using SFS algorithm |

---

## Tips for Exam Day

1. **Read the question carefully** - identify which algorithm/metric is being asked
2. **Check the formulas** - each file shows key formulas at the top
3. **Follow step-by-step** - solutions show all intermediate calculations
4. **Verify your answer** - check the summary section for expected results
5. **Match keywords** - use the Quick Problem Matching table above

Good luck on your exam! 🎓
