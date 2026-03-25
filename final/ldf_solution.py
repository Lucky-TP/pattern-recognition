"""
Linear Discriminant Function (LDF) Classifier - Step-by-Step Solution
======================================================================
Pattern Recognition Exam Practice

LDF is a linear classifier that uses a linear decision boundary to separate classes.

For two classes:
  Decision function: g(x) = w^T * x + w_0
  Decision rule: Choose ω₁ if g(x) > 0, else choose ω₂

For multi-class (one-vs-rest):
  g_i(x) = w_i^T * x + w_i0
  Decision: Choose class i where g_i(x) is maximum

Key Concepts:
- Linear decision boundary: g(x) = 0 is a hyperplane
- For Gaussian with equal covariance: LDF is optimal (Bayes classifier)
- Training: Find w and w_0 that minimize classification error
"""

import numpy as np


def print_vector(name, v):
    """Pretty print a vector."""
    print(f"{name} = [{v[0]:.4f}, {v[1]:.4f}]^T")


def ldf_classifier():
    print("=" * 70)
    print("Linear Discriminant Function (LDF) Classifier - Step-by-Step Solution")
    print("=" * 70)

    # ===== Part 1: Two-Class LDF =====
    print("\n" + "="*70)
    print("PART 1: TWO-CLASS LINEAR DISCRIMINANT FUNCTION")
    print("="*70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Class 1 data
    X1 = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2]
    ])

    # Class 2 data
    X2 = np.array([
        [5, 5],
        [5, 6],
        [6, 5],
        [6, 6]
    ])

    y1 = np.ones(len(X1))    # Class 1 labels = +1
    y2 = -np.ones(len(X2))    # Class 2 labels = -1

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    print("\nClass ω₁ (label = +1):")
    for i, x in enumerate(X1):
        print(f"  x_{i+1}^{(1)} = ({x[0]}, {x[1]})")

    print("\nClass ω₂ (label = -1):")
    for i, x in enumerate(X2):
        print(f"  x_{i+1}^{(2)} = ({x[0]}, {x[1]})")

    # ===== Step 2: Calculate Class Means =====
    print("\n" + "="*70)
    print("STEP 2: Calculate Class Means")
    print("="*70)

    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    print(f"\nμ₁ = (1/N₁) × Σ x_i^{(1)}")
    print(f"   = (1/{len(X1)}) × (sum of Class 1 points)")
    print(f"   = ({mu1[0]:.2f}, {mu1[1]:.2f})")

    print(f"\nμ₂ = (1/N₂) × Σ x_i^{(2)}")
    print(f"   = (1/{len(X2)}) × (sum of Class 2 points)")
    print(f"   = ({mu2[0]:.2f}, {mu2[1]:.2f})")

    # ===== Step 3: LDF for Gaussian with Equal Covariance =====
    print("\n" + "="*70)
    print("STEP 3: LDF for Gaussian with Equal Covariance (Bayes Optimal)")
    print("="*70)

    print("""
For two Gaussian classes with equal covariance Σ₁ = Σ₂ = Σ:

The optimal (Bayes) decision function is linear:

  g(x) = w^T * x + w_0

where:
  w = Σ⁻¹(μ₁ - μ₂)
  w_0 = -½(μ₁ + μ₂)^T Σ⁻¹(μ₁ - μ₂) + ln(P(ω₁)/P(ω₂))

For equal priors P(ω₁) = P(ω₂):
  w_0 = -½(μ₁ + μ₂)^T Σ⁻¹(μ₁ - μ₂)
""")

    # Estimate shared covariance
    n1, n2 = len(X1), len(X2)
    _d = X.shape[1]  # dimension (unused but kept for reference)

    # Centered data
    X1_centered = X1 - mu1
    X2_centered = X2 - mu2

    # Pooled covariance (within-class scatter)
    S1 = X1_centered.T @ X1_centered
    S2 = X2_centered.T @ X2_centered
    S_pooled = (S1 + S2) / (n1 + n2 - 2)

    print("\n--- Covariance Estimation ---")
    print("\nPooled covariance matrix Σ_pooled:")
    print(f"  Σ = (S₁ + S₂) / (n₁ + n₂ - 2)")
    print(f"\n  Σ = [[{S_pooled[0,0]:.4f}, {S_pooled[0,1]:.4f}],")
    print(f"       [{S_pooled[1,0]:.4f}, {S_pooled[1,1]:.4f}]]")

    # ===== Step 4: Calculate Weight Vector =====
    print("\n" + "="*70)
    print("STEP 4: Calculate Weight Vector w and Bias w_0")
    print("="*70)

    # Inverse covariance
    S_inv = np.linalg.inv(S_pooled)

    print("\nInverse covariance Σ⁻¹:")
    print(f"  [[{S_inv[0,0]:.4f}, {S_inv[0,1]:.4f}],")
    print(f"   [{S_inv[1,0]:.4f}, {S_inv[1,1]:.4f}]]")

    # Weight vector
    w = S_inv @ (mu1 - mu2)

    print(f"\nWeight vector w = Σ⁻¹(μ₁ - μ₂):")
    print(f"  μ₁ - μ₂ = ({mu1[0]-mu2[0]:.2f}, {mu1[1]-mu2[1]:.2f})")
    print_vector("  w", w)

    # Bias term (assuming equal priors)
    w0 = -0.5 * (mu1 + mu2) @ S_inv @ (mu1 - mu2)

    print(f"\nBias w_0 = -½(μ₁ + μ₂)^T × Σ⁻¹ × (μ₁ - μ₂):")
    print(f"  μ₁ + μ₂ = ({mu1[0]+mu2[0]:.2f}, {mu1[1]+mu2[1]:.2f})")
    print(f"  w_0 = {w0:.4f}")

    # ===== Step 5: Decision Function =====
    print("\n" + "="*70)
    print("STEP 5: Linear Discriminant Function")
    print("="*70)

    print(f"\nDecision function:")
    print(f"  g(x) = w^T × x + w_0")
    print(f"       = ({w[0]:.4f})×x₁ + ({w[1]:.4f})×x₂ + ({w0:.4f})")

    print(f"\nDecision rule:")
    print(f"  If g(x) > 0  → Classify as ω₁")
    print(f"  If g(x) < 0  → Classify as ω₂")
    print(f"  If g(x) = 0  → On decision boundary")

    print(f"\nDecision boundary: g(x) = 0")
    print(f"  ({w[0]:.4f})×x₁ + ({w[1]:.4f})×x₂ + ({w0:.4f}) = 0")

    if w[1] != 0:
        slope = -w[0] / w[1]
        intercept = -w0 / w[1]
        print(f"\n  In slope-intercept form:")
        print(f"    x₂ = {slope:.4f}×x₁ + {intercept:.4f}")

    # ===== Step 6: Classify Training Data =====
    print("\n" + "="*70)
    print("STEP 6: Classify Training Data Points")
    print("="*70)

    print("\nClassification of all training points:")
    print("-" * 70)
    print("  Point         x₁      x₂      g(x)        Decision    True    Correct?")
    print("-" * 70)

    correct = 0
    for i, (x, true_label) in enumerate(zip(X, y)):
        g = w @ x + w0
        decision = 1 if g > 0 else -1
        is_correct = decision == true_label
        if is_correct:
            correct += 1

        true_class = "ω₁" if true_label == 1 else "ω₂"
        dec_class = "ω₁" if decision == 1 else "ω₂"
        marker = "✓" if is_correct else "✗"

        print(f"  ({x[0]}, {x[1]})      {x[0]:.1f}     {x[1]:.1f}     {g:+.4f}       {dec_class:3s}        {true_class:3s}      {marker}")

    accuracy = correct / len(X)
    print(f"\nClassification accuracy: {correct}/{len(X)} = {accuracy:.2%}")

    # ===== Step 7: Classify New Points =====
    print("\n" + "="*70)
    print("STEP 7: Classify New Query Points")
    print("="*70)

    query_points = np.array([
        [3, 3],
        [4, 4],
        [1.5, 1.5],
        [5.5, 5.5]
    ])

    print("\nClassifying new points:")
    print("-" * 60)
    print("  Query Point     g(x)        Decision    Class")
    print("-" * 60)

    for x in query_points:
        g = w @ x + w0
        decision = "ω₁" if g > 0 else "ω₂"
        print(f"  ({x[0]}, {x[1]})        {g:+.4f}       {decision:3s}        {decision}")

    # ===== Part 2: Multi-Class LDF =====
    print("\n" + "="*70)
    print("PART 2: MULTI-CLASS LDF (One-vs-Rest)")
    print("="*70)

    print("""
For K classes, we train K binary classifiers:

For each class i:
  - Positive: class i (label +1)
  - Negative: all other classes (label -1)

Decision: Choose class i where g_i(x) is maximum

g_i(x) = w_i^T * x + w_i0

Classify: x → argmax_i g_i(x)
""")

    # Three-class example
    X_multi = np.array([
        [1, 1], [1, 2], [2, 1],           # Class 1
        [4, 4], [4, 5], [5, 4],           # Class 2
        [7, 7], [7, 8], [8, 7]            # Class 3
    ])

    y_multi = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

    print("\nMulti-class data (3 classes):")
    print("-" * 50)
    for c in [1, 2, 3]:
        idx = y_multi == c
        print(f"  Class ω_{c}: {[tuple(int(v) for v in x) for x in X_multi[idx]]}")

    # Train one-vs-rest classifiers
    print("\n--- Training One-vs-Rest Classifiers ---")

    classifiers = {}
    for target_class in [1, 2, 3]:
        # Create binary labels
        y_binary = np.where(y_multi == target_class, 1, -1)

        # Get positive and negative samples
        pos_idx = y_binary == 1
        neg_idx = y_binary == -1

        X_pos = X_multi[pos_idx]
        X_neg = X_multi[neg_idx]

        mu_pos = np.mean(X_pos, axis=0)
        mu_neg = np.mean(X_neg, axis=0)

        # Simple covariance estimation
        all_centered = np.vstack([X_pos - mu_pos, X_neg - mu_neg])
        S = np.cov(all_centered.T) + 0.1 * np.eye(2)  # Regularization

        S_inv = np.linalg.inv(S)

        w = S_inv @ (mu_pos - mu_neg)
        w0 = -0.5 * (mu_pos + mu_neg) @ S_inv @ (mu_pos - mu_neg)

        # Adjust for class imbalance
        n_pos = len(X_pos)
        n_neg = len(X_neg)
        w0 += np.log(n_pos / n_neg)

        classifiers[target_class] = (w, w0)

        print(f"\nClass ω_{target_class} vs Rest:")
        print_vector(f"  w_{target_class}", w)
        print(f"  w_{target_class}0 = {w0:.4f}")

    # Classify using all classifiers
    print("\n--- Multi-class Classification ---")
    print("\nClassifying training points:")
    print("-" * 70)
    print("  Point        g₁(x)     g₂(x)     g₃(x)     Decision    True")
    print("-" * 70)

    for x, true_label in zip(X_multi, y_multi):
        scores = {}
        for c, (w_c, w0_c) in classifiers.items():
            scores[c] = w_c @ x + w0_c

        predicted = max(scores.keys(), key=lambda k: scores[k])

        scores_str = f"{scores[1]:+.2f}    {scores[2]:+.2f}    {scores[3]:+.2f}"
        print(f"  ({x[0]}, {x[1]})     {scores_str}       ω_{predicted}          ω_{true_label}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Linear Discriminant Function (LDF)")
    print("="*70)

    print(f"""
Two-Class LDF:
  Decision function: g(x) = w^T × x + w_0

  For Gaussian with equal covariance (Bayes optimal):
    w = Σ⁻¹(μ₁ - μ₂)
    w_0 = -½(μ₁ + μ₂)^T × Σ⁻¹(μ₁ - μ₂)

  Decision rule:
    g(x) > 0 → Class ω₁
    g(x) < 0 → Class ω₂

Multi-Class LDF (One-vs-Rest):
  Train K classifiers: g_i(x) for each class i
  Decision: argmax_i g_i(x)

Key Properties:
• Linear decision boundary (hyperplane)
• Optimal for Gaussian classes with equal covariance
• Simple and interpretable
• Fast classification (just dot product)
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Given class means and covariance, derive LDF
2. Calculate g(x) for a query point
3. Find decision boundary equation
4. Compare LDF with other classifiers

Important Points:
• LDF = Linear classifier (hyperplane decision boundary)
• For Gaussian + equal covariance: LDF is Bayes optimal
• w is perpendicular to decision boundary
• Distance from point to boundary = |g(x)| / ||w||

LDF vs FLD (Fisher Linear Discriminant):
• FLD: Finds projection direction (unsupervised in a sense)
• LDF: Finds decision boundary (supervised classifier)
• For 2-class: FLD projection + threshold = LDF

Quick Calculations:
• g(x) = w^T × x + w_0
• Decision boundary: w^T × x + w_0 = 0
• Classify: sign(g(x))
""")

    return w, w0


if __name__ == "__main__":
    ldf_classifier()
