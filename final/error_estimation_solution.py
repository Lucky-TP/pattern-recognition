"""
Classification Error Estimation - Step-by-Step Solution
========================================================
Pattern Recognition Exam Practice

Topics Covered:
- Holdout Method
- Bootstrap Error Estimation
- Bias-Variance Tradeoff in Error Estimation
- .632 Bootstrap
- Confidence Intervals

Key Formulas:
    Holdout Error: E = (1/n_test) × Σ [y_i ≠ ŷ_i]

    Bootstrap Error: E_boot = (1/n) × Σ [y_i ≠ ŷ_bootstrap(x_i)]

    .632 Bootstrap: E_.632 = 0.368 × E_train + 0.632 × E_boot

    95% Confidence Interval: E ± 1.96 × sqrt(E(1-E)/n)
"""

import numpy as np


def error_estimation():
    print("=" * 70)
    print("Classification Error Estimation - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Dataset and True Error Rate")
    print("="*70)

    np.random.seed(42)

    # Simulated classifier results
    n_samples = 100
    true_labels = np.random.randint(0, 2, n_samples)
    predicted_labels = np.random.randint(0, 2, n_samples)  # Random classifier

    # Make some predictions correct (simulate ~60% accuracy)
    for i in range(60):
        idx = np.random.randint(0, n_samples)
        predicted_labels[idx] = true_labels[idx]

    print(f"\nSimulated dataset: {n_samples} samples")
    print(f"True labels: {np.sum(true_labels==0)} class 0, {np.sum(true_labels==1)} class 1")
    print(f"Predicted labels: {np.sum(predicted_labels==0)} class 0, {np.sum(predicted_labels==1)} class 1")

    # ===== Step 2: Holdout Method =====
    print("\n" + "="*70)
    print("STEP 2: Holdout Method")
    print("="*70)

    print("""
Holdout Method:
  1. Split data into training set and test set (e.g., 70%/30%)
  2. Train classifier on training set
  3. Evaluate on test set
  4. Report test error as estimate of true error
""")

    # Split 70/30
    n_train = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    y_train = true_labels[train_idx]
    y_test = true_labels[test_idx]
    y_pred_test = predicted_labels[test_idx]

    # Calculate holdout error
    holdout_errors = np.sum(y_test != y_pred_test)
    holdout_error_rate = holdout_errors / len(y_test)

    print(f"Split: {n_train} training, {n_samples - n_train} test")
    print(f"\nHoldout Error Calculation:")
    print(f"  Number of errors on test set: {holdout_errors}")
    print(f"  Test set size: {len(y_test)}")
    print(f"  Holdout Error: E_hold = {holdout_errors}/{len(y_test)} = {holdout_error_rate:.4f}")
    print(f"  Holdout Accuracy: {1 - holdout_error_rate:.4f}")

    print(f"""
Advantages of Holdout:
  • Simple and fast
  • Independent test set

Disadvantages of Holdout:
  • Reduces training data
  • High variance (depends on split)
  • Pessimistic bias (less training data)
""")

    # ===== Step 3: Bootstrap Error Estimation =====
    print("\n" + "="*70)
    print("STEP 3: Bootstrap Error Estimation")
    print("="*70)

    print("""
Bootstrap Method:
  1. Draw n samples with replacement to create bootstrap sample
  2. Train classifier on bootstrap sample
  3. Evaluate on out-of-bag (OOB) samples
  4. Repeat B times and average

Key Property:
  • P(sample in bootstrap) ≈ 1 - e^(-1) ≈ 0.632
  • P(sample NOT in bootstrap) ≈ e^(-1) ≈ 0.368
""")

    B = 10  # Number of bootstrap iterations
    bootstrap_errors = []

    print(f"\nPerforming {B} bootstrap iterations:")
    print("-" * 50)

    for b in range(B):
        # Bootstrap sample
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        oob_idx = np.array([i for i in range(n_samples) if i not in bootstrap_idx])

        if len(oob_idx) == 0:
            continue

        # Evaluate on OOB samples
        oob_errors = np.sum(true_labels[oob_idx] != predicted_labels[oob_idx])
        oob_error = oob_errors / len(oob_idx)
        bootstrap_errors.append(oob_error)

        print(f"  Iteration {b+1}: OOB samples = {len(oob_idx)}, "
              f"OOB error = {oob_error:.4f}")

    avg_bootstrap_error = np.mean(bootstrap_errors)

    print("-" * 50)
    print(f"\nAverage Bootstrap Error: E_boot = {avg_bootstrap_error:.4f}")

    # ===== Step 4: .632 Bootstrap =====
    print("\n" + "="*70)
    print("STEP 4: .632 Bootstrap Estimator")
    print("="*70)

    # Training error (resubstitution error)
    train_errors = np.sum(true_labels != predicted_labels)
    train_error = train_errors / n_samples

    print(f"\nTraining Error (Resubstitution):")
    print(f"  E_train = {train_errors}/{n_samples} = {train_error:.4f}")

    print(f"\n.632 Bootstrap Formula:")
    print(f"  E_.632 = 0.368 × E_train + 0.632 × E_boot")

    error_632 = 0.368 * train_error + 0.632 * avg_bootstrap_error

    print(f"\n  E_.632 = 0.368 × {train_error:.4f} + 0.632 × {avg_bootstrap_error:.4f}")
    print(f"        = {0.368 * train_error:.4f} + {0.632 * avg_bootstrap_error:.4f}")
    print(f"        = {error_632:.4f}")

    print("""
Why .632 Bootstrap?
  • Training error is optimistically biased (underestimates true error)
  • Bootstrap error is pessimistically biased (overestimates true error)
  • .632 bootstrap combines both to reduce bias
  • Weight 0.632 comes from P(in bootstrap) ≈ 0.632
""")

    # ===== Step 5: Confidence Intervals =====
    print("\n" + "="*70)
    print("STEP 5: Confidence Intervals for Error Rate")
    print("="*70)

    print("""
Confidence Interval for Error Rate:

For large n, error rate E follows approximately normal distribution:

  E ~ N(p, p(1-p)/n)

where p is the true error rate.

95% Confidence Interval:
  CI = E ± 1.96 × sqrt(E(1-E)/n)

99% Confidence Interval:
  CI = E ± 2.576 × sqrt(E(1-E)/n)
""")

    # Calculate CI for holdout error
    n_test = len(y_test)
    se = np.sqrt(holdout_error_rate * (1 - holdout_error_rate) / n_test)

    ci_95_lower = holdout_error_rate - 1.96 * se
    ci_95_upper = holdout_error_rate + 1.96 * se

    print(f"95% Confidence Interval for Holdout Error:")
    print(f"  Standard Error: SE = sqrt(E(1-E)/n)")
    print(f"                 = sqrt({holdout_error_rate:.4f} × {1-holdout_error_rate:.4f} / {n_test})")
    print(f"                 = {se:.4f}")
    print(f"\n  CI_95 = {holdout_error_rate:.4f} ± 1.96 × {se:.4f}")
    print(f"        = [{max(0, ci_95_lower):.4f}, {min(1, ci_95_upper):.4f}]")

    # ===== Step 6: Bias-Variance Tradeoff =====
    print("\n" + "="*70)
    print("STEP 6: Bias-Variance Tradeoff in Error Estimation")
    print("="*70)

    print("""
Error Estimation Methods Comparison:

┌─────────────────┬────────────────┬────────────────┐
│     Method      │      Bias      │    Variance    │
├─────────────────┼────────────────┼────────────────┤
│ Resubstitution  │ Pessimistic    │     Low        │
│ (Training Error)│ (underestimate)│                │
├─────────────────┼────────────────┼────────────────┤
│    Holdout      │ Pessimistic    │     High      │
│                 │ (less data)    │ (depends on   │
│                 │                │  split)       │
├─────────────────┼────────────────┼────────────────┤
│   K-Fold CV     │    Low         │   Medium      │
│                 │                │                │
├─────────────────┼────────────────┼────────────────┤
│   Bootstrap     │ Optimistic     │     Low       │
│                 │ (overestimate) │                │
├─────────────────┼────────────────┼────────────────┤
│  .632 Bootstrap │   Very Low     │     Low       │
│                 │  (best balance)│                │
└─────────────────┴────────────────┴────────────────┘

Guidelines:
• Small dataset: Use .632 bootstrap or LOOCV
• Large dataset: Holdout or k-fold CV is sufficient
• Need low variance: Use k-fold CV with k=10
• Need low bias: Use .632 bootstrap
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Classification Error Estimation")
    print("="*70)

    print(f"""
Dataset: {n_samples} samples

Error Estimates:
  Training Error (Resubstitution): {train_error:.4f}
  Holdout Error (70/30 split):     {holdout_error_rate:.4f}
  Bootstrap Error (avg of {B} iter):   {avg_bootstrap_error:.4f}
  .632 Bootstrap Error:            {error_632:.4f}

95% CI for Holdout: [{max(0, ci_95_lower):.4f}, {min(1, ci_95_upper):.4f}]

Key Formulas:
  Holdout Error: E = errors / n_test

  Bootstrap: Average OOB error over B iterations

  .632 Bootstrap: E_.632 = 0.368×E_train + 0.632×E_boot

  95% CI: E ± 1.96 × sqrt(E(1-E)/n)

Method Selection:
  • Large n: Holdout or k-fold CV
  • Small n: .632 bootstrap
  • Model selection: Cross-validation
  • Final evaluation: Holdout with independent test set
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Calculate holdout error from given data
2. Explain bias-variance tradeoff in error estimation
3. Compare different error estimation methods
4. Calculate confidence interval for error rate

Important Points:
• Resubstitution error underestimates true error
• Holdout has high variance
• K-fold CV reduces variance compared to holdout
• .632 bootstrap reduces bias

Quick Calculations:
• P(in bootstrap) = 1 - (1-1/n)^n ≈ 1 - e^(-1) ≈ 0.632
• SE = sqrt(p(1-p)/n)
• 95% CI: E ± 1.96 × SE

Key Numbers to Remember:
• e^(-1) ≈ 0.368
• 1 - e^(-1) ≈ 0.632
• 1.96 for 95% CI
• 2.576 for 99% CI
""")

    return holdout_error_rate, error_632


if __name__ == "__main__":
    error_estimation()
