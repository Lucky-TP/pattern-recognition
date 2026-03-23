"""
Random Forest / Bagging - Step-by-Step Solution
===============================================
Pattern Recognition Exam Practice

Bagging (Bootstrap Aggregating):
1. Create B bootstrap samples from training data
2. Train B decision trees on bootstrap samples
3. Aggregate predictions: majority vote (classification) or average (regression)

Random Forest Enhancement:
- At each split, only consider random subset of m features
- m = √d for classification, m = d/3 for regression
- Decorrelates trees, improves diversity

Key Formulas:
    Bootstrap sample: Sample n points with replacement
    Out-of-Bag (OOB) error: Error on samples not in bootstrap

    Feature importance: Measure decrease in impurity or permutation importance
"""

import numpy as np


def bootstrap_sample(X, y, random_state=None):
    """Generate a bootstrap sample."""
    n = len(X)
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices], indices


def random_forest():
    print("=" * 70)
    print("Random Forest / Bagging - Step-by-Step Solution")
    print("=" * 70)

    np.random.seed(42)  # For reproducibility

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Simple 2D classification data
    X = np.array([
        [1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
        [5, 5], [5, 6], [6, 5], [6, 6],  # Class 1
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    n = len(X)
    d = X.shape[1]

    print("\nTraining Data:")
    print("-" * 50)
    print("  Index    Features    Class")
    print("-" * 50)
    for i in range(n):
        print(f"    {i+1}      ({X[i,0]}, {X[i,1]})       {y[i]}")

    print(f"\nTotal samples: n = {n}")
    print(f"Number of features: d = {d}")

    # ===== Step 2: Bootstrap Sampling =====
    print("\n" + "="*70)
    print("STEP 2: Bootstrap Sampling")
    print("="*70)

    print("""
Bootstrap Sample:
  - Sample n points WITH REPLACEMENT from training data
  - Some points appear multiple times, some not at all
  - On average: ~63.2% unique samples per bootstrap

Formula for probability of being selected:
  P(not selected) = (1 - 1/n)^n ≈ e^(-1) ≈ 0.368
  P(selected at least once) ≈ 1 - 0.368 ≈ 0.632
""")

    B = 3  # Number of trees (bootstrap samples)

    print(f"\nCreating {B} bootstrap samples:")
    print("="*60)

    bootstrap_samples = []

    for b in range(B):
        print(f"\nBootstrap Sample {b+1}:")
        print("-" * 40)

        X_boot, y_boot, indices = bootstrap_sample(X, y)

        print("  Selected indices:", indices + 1)
        print("\n  Bootstrap dataset:")
        print("  Index    Features    Class")
        print("  " + "-" * 35)
        for i, (x, label) in enumerate(zip(X_boot, y_boot)):
            print(f"   {i+1:2d}      ({x[0]}, {x[1]})       {label}")

        # Find out-of-bag samples
        unique_indices = set(indices)
        oob_indices = [i for i in range(n) if i not in unique_indices]

        print(f"\n  Out-of-bag (OOB) samples: indices {oob_indices}")

        bootstrap_samples.append((X_boot, y_boot, indices, oob_indices))

    # ===== Step 3: Feature Random Selection =====
    print("\n" + "="*70)
    print("STEP 3: Random Feature Selection at Each Split")
    print("="*70)

    print(f"""
Random Forest Feature Selection:
  - At each node split, randomly select m features
  - Only consider these m features for best split
  - This decorrelates trees

Recommended m values:
  Classification: m = √d = √{d} = {int(np.sqrt(d))} feature(s)
  Regression:     m = d/3 = {d}/3 = {d/3:.1f} feature(s)

For our 2D example:
  - At each split, we consider m = {int(np.sqrt(d))} randomly chosen feature(s)
  - This ensures trees are different even with same bootstrap sample
""")

    # ===== Step 4: Train Individual Trees =====
    print("\n" + "="*70)
    print("STEP 4: Train Individual Decision Trees")
    print("="*70)

    print("""
Each decision tree is trained on:
  1. A bootstrap sample of the data
  2. Using random subset of features at each split

Decision Tree Training (simplified):
  1. Find best split using only m random features
  2. Split data into left/right children
  3. Recursively build tree until stopping criterion

For our example, we'll show simplified tree structures:
""")

    # Simulate tree predictions (for demonstration)
    tree_predictions = [
        [0, 0, 0, 0, 1, 1, 1, 1],  # Tree 1 predictions
        [0, 0, 0, 0, 1, 1, 1, 1],  # Tree 2 predictions
        [0, 0, 0, 1, 1, 1, 1, 1],  # Tree 3 predictions (makes one error)
    ]

    for b in range(B):
        print(f"\nTree {b+1} predictions on original training data:")
        print("-" * 40)
        print("  Index    True    Predicted")
        print("-" * 40)
        for i in range(n):
            pred = tree_predictions[b][i]
            marker = "" if y[i] == pred else " ✗"
            print(f"    {i+1}       {y[i]}         {pred}       {marker}")

    # ===== Step 5: Aggregate Predictions (Voting) =====
    print("\n" + "="*70)
    print("STEP 5: Aggregate Predictions (Majority Voting)")
    print("="*70)

    print("""
Ensemble Prediction:
  - Classification: Majority vote among all trees
  - Regression: Average of all tree predictions
""")

    print("\nFinal ensemble predictions:")
    print("-" * 60)
    print("  Index    True    T1    T2    T3    Votes      Final    Correct?")
    print("-" * 60)

    correct = 0
    for i in range(n):
        votes = [tree_predictions[b][i] for b in range(B)]
        vote_0 = votes.count(0)
        vote_1 = votes.count(1)
        final = 1 if vote_1 > vote_0 else 0
        is_correct = final == y[i]
        if is_correct:
            correct += 1

        vote_str = f"0:{vote_0}, 1:{vote_1}"
        print(f"    {i+1}       {y[i]}      {votes[0]}     {votes[1]}     {votes[2]}     {vote_str}      {final}        {'✓' if is_correct else '✗'}")

    accuracy = correct / n
    print(f"\nEnsemble Accuracy: {correct}/{n} = {accuracy:.2%}")

    # ===== Step 6: Out-of-Bag Error Estimation =====
    print("\n" + "="*70)
    print("STEP 6: Out-of-Bag (OOB) Error Estimation")
    print("="*70)

    print("""
OOB Error:
  - For each sample, use trees that did NOT include it in training
  - Average predictions from those trees
  - Compare with true label

This provides a free cross-validation estimate!
""")

    # Simulated OOB predictions
    print("\nOOB predictions for each sample:")
    print("-" * 60)
    print("  Sample    OOB Trees    OOB Prediction    True    Correct?")
    print("-" * 60)

    # For demonstration
    oob_results = [
        (1, [2, 3], 0, 0, True),
        (2, [1, 3], 0, 0, True),
        (3, [2], 0, 0, True),
        (4, [1], 0, 0, True),
        (5, [3], 1, 1, True),
        (6, [1], 1, 1, True),
        (7, [2, 3], 1, 1, True),
        (8, [1, 2], 1, 1, True),
    ]

    oob_correct = 0
    for sample, oob_trees, pred, true, correct in oob_results:
        print(f"    {sample}        {oob_trees}            {pred}                {true}        {'✓' if correct else '✗'}")
        if correct:
            oob_correct += 1

    oob_error = 1 - oob_correct / n
    print(f"\nOOB Error Estimate: {oob_correct}/{n} = {1-oob_error:.2%} accuracy")
    print(f"OOB Error Rate: {oob_error:.2%}")

    # ===== Step 7: Feature Importance =====
    print("\n" + "="*70)
    print("STEP 7: Feature Importance")
    print("="*70)

    print("""
Feature Importance Calculation:
  1. Mean Decrease in Impurity (MDI):
     - Sum the decrease in Gini/entropy for each feature across all trees
     - Average over all trees

  2. Permutation Importance:
     - Permute feature values in OOB samples
     - Measure increase in error
     - Larger increase = more important feature

For our example:
  - Feature 1 (x1): High importance (main separator)
  - Feature 2 (x2): Lower importance
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Random Forest / Bagging")
    print("="*70)

    print(f"""
Data: {n} samples, {d} features
Number of trees: B = {B}

Bagging Process:
1. Create B bootstrap samples (sample with replacement)
2. Train B decision trees on bootstrap samples
3. Aggregate predictions by majority vote

Random Forest Enhancement:
- Random feature selection at each split (m = √d features)
- Increases tree diversity, reduces correlation

Results:
  Training Accuracy: {accuracy:.2%}
  OOB Error Estimate: {oob_error:.2%}

Key Properties:
  • Reduces variance (compared to single tree)
  • Robust to overfitting
  • Provides OOB error estimate
  • Can estimate feature importance
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Explain bootstrap sampling process
2. Calculate OOB error
3. Compare Random Forest with single Decision Tree
4. Explain why Random Forest reduces overfitting

Important Points:
• Bootstrap: sample n points with replacement
• ~63.2% unique samples per bootstrap
• OOB samples: ~36.8% not selected
• Random Forest = Bagging + Random Feature Selection

Key Differences:
• Bagging: Uses all features for split
• Random Forest: Uses m random features for split
• This decorrelation improves performance

Why Random Forest Works:
1. Bootstrap reduces variance
2. Random features reduce tree correlation
3. Averaging uncorrelated predictors reduces error more

Quick Formulas:
• P(in bootstrap) ≈ 1 - e^(-1) ≈ 0.632
• m_features = √d (classification) or d/3 (regression)
""")

    return accuracy, oob_error


if __name__ == "__main__":
    random_forest()
