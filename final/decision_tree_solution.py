"""
Decision Tree Classifier - Step-by-Step Solution
=================================================
Pattern Recognition Exam Practice

Decision Tree Algorithm:
1. Select best feature to split using impurity measure
2. Split data based on feature values
3. Recursively build subtrees
4. Stop when pure or stopping criterion met

Key Concepts:
- Information Gain (based on Entropy)
- Gini Impurity
- Split criteria

Key Formulas:
    Entropy: H(D) = -Σ p_i × log₂(p_i)
    Gini: Gini(D) = 1 - Σ p_i²
    Information Gain: IG(D, A) = H(D) - Σ (|D_v|/|D|) × H(D_v)

where p_i = proportion of class i in dataset D
"""

import numpy as np


def entropy(y):
    """Calculate entropy of a label array."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def gini(y):
    """Calculate Gini impurity of a label array."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)


def information_gain(y, feature_values, split_value):
    """Calculate information gain for a binary split."""
    n = len(y)
    parent_entropy = entropy(y)

    # Split into left and right
    left_mask = feature_values <= split_value
    right_mask = ~left_mask

    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0

    # Weighted average of children entropy
    child_entropy = (n_left / n) * entropy(y[left_mask]) + (n_right / n) * entropy(y[right_mask])

    return parent_entropy - child_entropy


def gini_gain(y, feature_values, split_value):
    """Calculate Gini gain for a binary split."""
    n = len(y)
    parent_gini = gini(y)

    left_mask = feature_values <= split_value
    right_mask = ~left_mask

    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0

    child_gini = (n_left / n) * gini(y[left_mask]) + (n_right / n) * gini(y[right_mask])

    return parent_gini - child_gini


def decision_tree():
    print("=" * 70)
    print("Decision Tree Classifier - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Dataset: [Feature1, Feature2, Class]
    # Features: Outlook (Sunny=0, Overcast=1, Rain=2), Wind (Weak=0, Strong=1)
    # Target: Play Tennis (No=0, Yes=1)

    data = np.array([
        [0, 0, 0],  # Sunny, Weak, No
        [0, 1, 0],  # Sunny, Strong, No
        [1, 0, 1],  # Overcast, Weak, Yes
        [1, 1, 1],  # Overcast, Strong, Yes
        [2, 0, 1],  # Rain, Weak, Yes
        [2, 1, 0],  # Rain, Strong, No
    ])

    # Alternative numeric example
    X = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [4, 4],
        [5, 5],
        [5, 6],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])  # Class 0 or Class 1

    print("\nTraining Data (numeric example):")
    print("-" * 50)
    print("  Sample    Feature1 (x₁)    Feature2 (x₂)    Class (y)")
    print("-" * 50)
    for i in range(len(X)):
        print(f"    {i+1}           {X[i,0]}                {X[i,1]}             {y[i]}")

    # ===== Step 2: Calculate Root Entropy =====
    print("\n" + "="*70)
    print("STEP 2: Calculate Root Node Entropy (Parent)")
    print("="*70)

    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)

    print(f"\nClass distribution at root:")
    for c, cnt in zip(classes, counts):
        p = cnt / n_total
        print(f"  Class {c}: {cnt} samples, p_{c} = {cnt}/{n_total} = {p:.4f}")

    root_entropy = entropy(y)
    root_gini = gini(y)

    print(f"\nEntropy formula: H(D) = -Σ p_i × log₂(p_i)")
    print(f"Entropy at root: H(D) = ", end="")
    for i, (c, cnt) in enumerate(zip(classes, counts)):
        p = cnt / n_total
        if i > 0:
            print(" - ", end="")
        print(f"-{p:.4f}×log₂({p:.4f})", end="")
    print(f" = {root_entropy:.4f}")

    print(f"\nGini formula: Gini(D) = 1 - Σ p_i²")
    print(f"Gini at root: Gini(D) = 1 - ", end="")
    for i, (c, cnt) in enumerate(zip(classes, counts)):
        p = cnt / n_total
        if i > 0:
            print(" - ", end="")
        print(f"{p:.4f}²", end="")
    print(f" = {root_gini:.4f}")

    # ===== Step 3: Find Best Split =====
    print("\n" + "="*70)
    print("STEP 3: Find Best Feature and Split Point")
    print("="*70)

    print("\nEvaluating all possible splits:")
    print("-" * 70)

    best_gain = -1
    best_feature = 0  # Initialize with default
    best_split = 0.0  # Initialize with default

    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        print(f"\n--- Feature {feature_idx + 1} (x{feature_idx + 1}) ---")
        print(f"Unique values: {unique_values}")

        # Try midpoints between consecutive values
        for i in range(len(unique_values) - 1):
            split_value = (unique_values[i] + unique_values[i + 1]) / 2

            ig = information_gain(y, feature_values, split_value)
            gg = gini_gain(y, feature_values, split_value)

            left_mask = feature_values <= split_value
            right_mask = ~left_mask

            left_y = y[left_mask]
            right_y = y[right_mask]

            print(f"\n  Split at x{feature_idx + 1} ≤ {split_value}:")
            print(f"    Left (x{feature_idx + 1} ≤ {split_value}): {len(left_y)} samples, classes = {list(left_y)}")
            print(f"      H(left) = {entropy(left_y):.4f}")
            print(f"    Right (x{feature_idx + 1} > {split_value}): {len(right_y)} samples, classes = {list(right_y)}")
            print(f"      H(right) = {entropy(right_y):.4f}")
            print(f"    Information Gain = {root_entropy:.4f} - "
                  f"({len(left_y)}/{n_total})×{entropy(left_y):.4f} - "
                  f"({len(right_y)}/{n_total})×{entropy(right_y):.4f}")
            print(f"                      = {ig:.4f}")
            print(f"    Gini Gain = {gg:.4f}")

            if ig > best_gain:
                best_gain = ig
                best_feature = feature_idx
                best_split = split_value

    print(f"\n{'='*70}")
    print(f"BEST SPLIT: Feature {best_feature + 1} (x{best_feature + 1}) ≤ {best_split}")
    print(f"Information Gain: {best_gain:.4f}")
    print(f"{'='*70}")

    # ===== Step 4: Build Tree Structure =====
    print("\n" + "="*70)
    print("STEP 4: Build Decision Tree Structure")
    print("="*70)

    print(f"""
Decision Tree (first level):

                    Root
                     │
         ┌───────────┴───────────┐
         │                       │
    x{best_feature + 1} ≤ {best_split}            x{best_feature + 1} > {best_split}
         │                       │
    Class 0 (Pure)          Class 1 (Pure)
    (3 samples)             (3 samples)

Tree Rules:
  IF x{best_feature + 1} ≤ {best_split} THEN Class 0
  IF x{best_feature + 1} > {best_split} THEN Class 1
""")

    # ===== Step 5: Classify New Points =====
    print("\n" + "="*70)
    print("STEP 5: Classify New Query Points")
    print("="*70)

    query_points = np.array([
        [1.5, 1.5],
        [3.0, 3.0],
        [4.5, 4.5],
        [5.5, 5.5],
    ])

    print("\nClassifying new points using the decision tree:")
    print("-" * 60)
    print("  Query Point    x₁ ≤ 3?    Decision")
    print("-" * 60)

    for x in query_points:
        if x[best_feature] <= best_split:
            decision = 0
        else:
            decision = 1
        check = "Yes" if x[best_feature] <= best_split else "No"
        print(f"  ({x[0]}, {x[1]})        {check:3s}       Class {decision}")

    # ===== Step 6: Entropy vs Gini Comparison =====
    print("\n" + "="*70)
    print("STEP 6: Entropy vs Gini Impurity Comparison")
    print("="*70)

    print("""
Entropy (Information Gain):
  Formula: H(D) = -Σ p_i × log₂(p_i)
  Range: [0, log₂(k)] where k = number of classes
  - H = 0: Pure node (all same class)
  - H = 1: Maximum impurity (binary, 50-50 split)

Gini Impurity:
  Formula: Gini(D) = 1 - Σ p_i²
  Range: [0, 1 - 1/k]
  - Gini = 0: Pure node
  - Gini = 0.5: Maximum impurity (binary, 50-50 split)

Comparison:
  • Both measure node impurity
  • Gini is faster to compute (no log)
  • Entropy based on information theory
  • Results usually similar

Information Gain Ratio (C4.5):
  IGR(D, A) = IG(D, A) / SplitInformation(D, A)
  SplitInformation(D, A) = -Σ (|D_v|/|D|) × log₂(|D_v|/|D|)
  - Fixes bias toward features with many values
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Decision Tree Algorithm")
    print("="*70)

    print(f"""
Training Data: {n_total} samples, 2 features, 2 classes

Root Node:
  Entropy: {root_entropy:.4f}
  Gini: {root_gini:.4f}

Best Split:
  Feature: x{best_feature + 1}
  Threshold: {best_split}
  Information Gain: {best_gain:.4f}

Algorithm Steps:
1. Calculate impurity (Entropy/Gini) at current node
2. For each feature, try all possible splits
3. Calculate Information Gain / Gini Gain for each split
4. Choose split with maximum gain
5. Recursively repeat for child nodes
6. Stop when: pure node, max depth, min samples

Stopping Criteria:
• All samples same class (pure)
• Maximum depth reached
• Minimum samples per node
• No improvement from splits
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Calculate entropy and information gain
2. Find best split point
3. Build decision tree step by step
4. Compare Entropy vs Gini

Important Formulas:
  Entropy: H = -Σ p_i × log₂(p_i)
  Gini: G = 1 - Σ p_i²
  Info Gain: IG = H(parent) - Σ w_i × H(child_i)

Quick Calculations:
• Binary entropy: H = -p×log₂(p) - (1-p)×log₂(1-p)
• p = 0.5 → H = 1 (maximum)
• p = 0 or p = 1 → H = 0 (pure)
• Pure node: all same class, entropy = 0

Decision Tree Types:
• ID3: Uses Entropy (Information Gain)
• C4.5: Uses Information Gain Ratio
• CART: Uses Gini Impurity
""")

    return best_feature, best_split, best_gain


if __name__ == "__main__":
    decision_tree()
