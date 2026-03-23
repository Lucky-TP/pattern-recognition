"""
K-Fold Cross-Validation - Step-by-Step Solution
===============================================
Pattern Recognition Exam Practice

Algorithm:
1. Split data into k equal-sized folds
2. For each fold i (i = 1 to k):
   - Train on k-1 folds (all except fold i)
   - Test on fold i
   - Record performance
3. Average performance across all k iterations

Key Benefits:
- More reliable estimate of model performance
- Uses all data for both training and testing
- Reduces variance compared to single train/test split
"""

import numpy as np


def cross_validation():
    print("=" * 70)
    print("K-Fold Cross-Validation - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Data")
    print("="*70)

    # Simple dataset with indices
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    n = len(X)
    k = 4  # Number of folds

    print(f"\nDataset: {n} samples")
    print("-" * 40)
    print("  Index    X    y (class)")
    print("-" * 40)
    for i in range(n):
        print(f"    {i+1:2d}     {X[i]:2d}      {y[i]}")

    print(f"\nClass distribution:")
    print(f"  Class 1: {np.sum(y == 1)} samples")
    print(f"  Class 0: {np.sum(y == 0)} samples")

    # ===== Step 2: Create Folds =====
    print("\n" + "="*70)
    print(f"STEP 2: Create {k} Folds")
    print("="*70)

    print(f"\nFold size: {n//k} samples per fold")

    # Create fold indices
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        fold_indices = list(range(start, end))
        folds.append(fold_indices)

    print("\nFold assignments:")
    print("-" * 50)
    for i, fold in enumerate(folds):
        samples = [(X[j], y[j]) for j in fold]
        print(f"  Fold {i+1}: indices {fold} → {samples}")

    # ===== Step 3: K-Fold Iterations =====
    print("\n" + "="*70)
    print("STEP 3: K-Fold Cross-Validation Iterations")
    print("="*70)

    # Simulated accuracy scores (for demonstration)
    # In real scenario, you would train a classifier here
    simulated_accuracies = [0.67, 1.0, 0.67, 1.0]

    fold_results = []

    for i in range(k):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1} of {k}")
        print(f"{'='*60}")

        # Define train and test sets
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])

        test_X = X[test_indices]
        test_y = y[test_indices]
        train_X = X[train_indices]
        train_y = y[train_indices]

        print(f"\nTraining set: indices {train_indices}")
        print(f"  X_train = {list(train_X)}")
        print(f"  y_train = {list(train_y)}")

        print(f"\nTest set (Fold {i+1}): indices {test_indices}")
        print(f"  X_test = {list(test_X)}")
        print(f"  y_test = {list(test_y)}")

        print(f"\nTraining classifier on {len(train_indices)} samples...")
        print(f"Testing on {len(test_indices)} samples...")

        # Simulated prediction and accuracy
        accuracy = simulated_accuracies[i]
        fold_results.append(accuracy)

        print(f"\n*** Fold {i+1} Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%) ***")

    # ===== Step 4: Calculate Average Performance =====
    print("\n" + "="*70)
    print("STEP 4: Calculate Average Performance")
    print("="*70)

    print("\nAccuracy for each fold:")
    print("-" * 40)
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i+1}: {acc:.4f}")

    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)

    print("-" * 40)
    print(f"  Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.1f}%)")
    print(f"  Std Deviation: {std_accuracy:.4f}")

    print(f"""
Cross-Validation Result:
  Average Accuracy = (1/k) × Σ Accuracy_i
                   = (1/{k}) × ({' + '.join([f'{a:.2f}' for a in fold_results])})
                   = {sum(fold_results):.2f} / {k}
                   = {mean_accuracy:.4f}
""")

    # ===== Step 5: Stratified K-Fold =====
    print("\n" + "="*70)
    print("STEP 5: Stratified K-Fold Cross-Validation")
    print("="*70)

    print("""
Stratified K-Fold:
  - Maintains class distribution in each fold
  - Important for imbalanced datasets
  - Each fold has same proportion of each class as full dataset

Example with stratified folds:
""")

    # Manual stratified split
    class1_indices = [0, 1, 2, 3, 4, 5]   # Class 1
    class0_indices = [6, 7, 8, 9, 10, 11]  # Class 0

    print("\nOriginal class distribution: 50% Class 1, 50% Class 0")
    print("\nStratified folds (maintaining 50-50 ratio):")
    print("-" * 50)

    strat_folds = [
        [0, 1, 6, 7],    # 2 Class 1, 2 Class 0
        [2, 3, 8, 9],    # 2 Class 1, 2 Class 0
        [4, 5, 10, 11],  # 2 Class 1, 2 Class 0
    ]

    for i, fold in enumerate(strat_folds):
        fold_y = [y[j] for j in fold]
        class1_count = fold_y.count(1)
        class0_count = fold_y.count(0)
        print(f"  Fold {i+1}: indices {fold}")
        print(f"         Class 1: {class1_count}, Class 0: {class0_count}")

    # ===== Step 6: Choosing K =====
    print("\n" + "="*70)
    print("STEP 6: Choosing the Number of Folds (K)")
    print("="*70)

    print("""
Common choices for K:

  K = 5  : Most common choice
           - Good balance between bias and variance
           - 80% train, 20% test per fold

  K = 10 : Standard for model selection
           - Lower bias, higher variance
           - 90% train, 10% test per fold

  K = N  : Leave-One-Out Cross-Validation (LOOCV)
           - Lowest bias, highest variance
           - Computationally expensive
           - Each fold: train on N-1, test on 1

Guidelines:
  • Small datasets: Use higher K (e.g., K=10 or LOOCV)
  • Large datasets: K=5 is usually sufficient
  • Classification: Prefer stratified K-fold
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: K-Fold Cross-Validation")
    print("="*70)

    print(f"""
Dataset: {n} samples, {k} folds

Process:
1. Split data into {k} equal folds
2. For each fold i: train on {k-1} folds, test on fold i
3. Calculate mean and std of accuracies

Results:
  Fold Accuracies: {[f'{a:.2f}' for a in fold_results]}
  Mean Accuracy: {mean_accuracy:.4f}
  Std Deviation: {std_accuracy:.4f}

Advantages:
• More reliable performance estimate
• Uses all data for training and testing
• Reduces overfitting to single train/test split
• Provides variance estimate (model stability)

Disadvantages:
• K times more computation
• May have high variance for small datasets
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform k-fold cross-validation step by step
2. Calculate mean accuracy from fold results
3. Explain difference between k-fold and stratified k-fold
4. Compare cross-validation with hold-out method

Important Points:
• K-fold gives K different train/test splits
• Each sample is used for testing exactly once
• Each sample is used for training K-1 times
• Stratified maintains class proportions

Quick Calculations:
• Total training samples per iteration: N × (K-1)/K
• Total test samples per iteration: N/K
• Final accuracy = mean of fold accuracies

When to use what:
• K=5 or K=10: Standard for most cases
• Stratified: Classification with class imbalance
• LOOCV: Very small datasets (K=N)
""")

    return mean_accuracy, std_accuracy


if __name__ == "__main__":
    cross_validation()
