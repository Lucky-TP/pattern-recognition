"""
AdaBoost Algorithm - Step-by-Step Solution
==========================================
Pattern Recognition Exam Practice

Algorithm:
1. Initialize weights: w_i = 1/N for all samples
2. For t = 1 to T iterations:
   a. Train weak classifier h_t using weighted samples
   b. Calculate weighted error: ε_t = Σ w_i * I(y_i ≠ h_t(x_i))
   c. Calculate classifier weight: α_t = 0.5 * ln((1-ε_t)/ε_t)
   d. Update sample weights: w_i ← w_i * exp(-α_t * y_i * h_t(x_i))
   e. Normalize weights: w_i ← w_i / Σ w_j
3. Final classifier: H(x) = sign(Σ α_t * h_t(x))

Key Formulas:
    Weighted Error: ε_t = Σ w_i * [y_i ≠ h_t(x_i)]
    Classifier Weight: α_t = 0.5 × ln((1-ε_t)/ε_t)
    Weight Update: w_i ← w_i × exp(-α_t × y_i × h_t(x_i))
"""

import numpy as np


def adaboost_algorithm():
    print("=" * 70)
    print("AdaBoost Algorithm - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Training Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Simple 1D data for easy visualization
    # Features: x (1D), Labels: y (+1 or -1)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, 1])

    N = len(X)

    print("\nTraining Data:")
    print("-" * 40)
    print("  i    x_i    y_i")
    print("-" * 40)
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"  {i+1:2d}    {x:2d}     {label:+d}")

    print(f"\nTotal samples: N = {N}")

    # ===== Step 2: Initialize Weights =====
    print("\n" + "="*70)
    print("STEP 2: Initialize Sample Weights")
    print("="*70)

    w = np.ones(N) / N

    print(f"\nInitial weights: w_i = 1/N = 1/{N} = {1/N:.4f}")
    print(f"\nWeights: {[f'{wi:.4f}' for wi in w]}")
    print(f"Sum of weights: {np.sum(w):.4f}")

    # ===== AdaBoost Iterations =====
    print("\n" + "="*70)
    print("ADABOOST ITERATIONS")
    print("="*70)

    T = 3  # Number of weak classifiers
    alphas = []
    classifiers = []
    best_predictions = np.zeros(N, dtype=int)  # Initialize

    for t in range(1, T + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION t = {t}")
        print(f"{'='*70}")

        # ===== Train Weak Classifier =====
        print(f"\n--- Training Weak Classifier h_{t} ---")
        print(f"Current sample weights: {[f'{wi:.4f}' for wi in w]}")

        # Simple decision stump: h(x) = sign(x - threshold)
        # Try all possible thresholds
        best_threshold = None
        best_error = float('inf')
        best_predictions = None

        print("\nEvaluating all possible decision stumps (thresholds):")
        print("-" * 60)
        print("  Threshold    Predictions                Weighted Error")
        print("-" * 60)

        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            predictions = np.where(X > threshold, 1, -1)

            # Calculate weighted error
            error = np.sum(w * (predictions != y))

            pred_str = ''.join(['+' if p == 1 else '-' for p in predictions])
            marker = " ← BEST" if error < best_error else ""
            print(f"    {threshold:4.1f}       [{pred_str}]           ε = {error:.4f}{marker}")

            if error < best_error:
                best_error = error
                best_threshold = threshold
                best_predictions = predictions

        print(f"\nBest threshold: {best_threshold}")
        print(f"Best weak classifier: h_{t}(x) = sign(x - {best_threshold})")

        # ===== Calculate Classifier Weight =====
        print(f"\n--- Calculating Classifier Weight α_{t} ---")

        epsilon_t = best_error
        print(f"\nWeighted error: ε_{t} = {epsilon_t:.4f}")

        if epsilon_t == 0:
            alpha_t = float('inf')
            print(f"Perfect classifier! ε_{t} = 0, α_{t} = ∞")
        elif epsilon_t == 1:
            alpha_t = float('-inf')
            print(f"Worst classifier! ε_{t} = 1, α_{t} = -∞")
        else:
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            print(f"\nFormula: α_{t} = 0.5 × ln((1 - ε_{t}) / ε_{t})")
            print(f"       = 0.5 × ln((1 - {epsilon_t:.4f}) / {epsilon_t:.4f})")
            print(f"       = 0.5 × ln({(1-epsilon_t)/epsilon_t:.4f})")
            print(f"       = 0.5 × {np.log((1-epsilon_t)/epsilon_t):.4f}")
            print(f"       = {alpha_t:.4f}")

        alphas.append(alpha_t)
        classifiers.append((best_threshold, best_predictions))

        # ===== Update Weights =====
        print(f"\n--- Updating Sample Weights ---")

        print(f"\nFormula: w_i ← w_i × exp(-α_{t} × y_i × h_{t}(x_i))")

        # Calculate weight updates
        weight_factors = np.exp(-alpha_t * y * best_predictions)

        print("\nWeight updates for each sample:")
        print("-" * 70)
        print("  i    y_i   h_{t}(x_i)   y_i×h_{t}(x_i)   exp(-α×y×h)    Old w_i     New w_i (unnormalized)")
        print("-" * 70)

        new_w = w * weight_factors
        for i in range(N):
            yh = y[i] * best_predictions[i]
            factor = weight_factors[i]
            print(f"  {i+1:2d}   {y[i]:+d}      {best_predictions[i]:+d}          {yh:+d}           {factor:.4f}        {w[i]:.4f}        {new_w[i]:.4f}")

        # Normalize weights
        w_sum = np.sum(new_w)
        w = new_w / w_sum

        print(f"\nNormalization: sum = {w_sum:.4f}")
        print(f"Normalized weights: {[f'{wi:.4f}' for wi in w]}")

        # Show misclassified samples get higher weights
        misclassified = best_predictions != y
        if np.any(misclassified):
            print(f"\nMisclassified samples: {[i+1 for i, m in enumerate(misclassified) if m]}")
            print("These samples now have HIGHER weights for next iteration.")

    # ===== Final Classifier =====
    print("\n" + "="*70)
    print("FINAL ENSEMBLE CLASSIFIER")
    print("="*70)

    print(f"\nFinal classifier: H(x) = sign(Σ α_t × h_t(x))")
    print(f"\nClassifier weights and thresholds:")
    for t, (alpha, (threshold, _)) in enumerate(zip(alphas, classifiers), 1):
        print(f"  h_{t}: threshold = {threshold}, α_{t} = {alpha:.4f}")

    print(f"\nFinal predictions for training data:")
    print("-" * 60)
    print("  x    y    h_1   h_2   h_3   Σ(α×h)     H(x)    Correct?")
    print("-" * 60)

    for i, x in enumerate(X):
        # Calculate weighted sum of weak classifiers
        weighted_sum = 0
        h_values = []
        for t, (alpha, (threshold, _)) in enumerate(zip(alphas, classifiers), 1):
            h = 1 if x > threshold else -1
            h_values.append(h)
            weighted_sum += alpha * h

        final_pred = 1 if weighted_sum > 0 else -1
        correct = "✓" if final_pred == y[i] else "✗"

        h_str = "   ".join([f"{h:+d}" for h in h_values])
        print(f"  {x:2d}   {y[i]:+d}   {h_str}   {weighted_sum:+.2f}    {final_pred:+d}       {correct}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: AdaBoost Algorithm")
    print("="*70)

    print(f"""
Training Data: {N} samples

Weak Classifiers (Decision Stumps):
  h_1: threshold = {classifiers[0][0]}, α_1 = {alphas[0]:.4f}
  h_2: threshold = {classifiers[1][0]}, α_2 = {alphas[1]:.4f}
  h_3: threshold = {classifiers[2][0]}, α_3 = {alphas[2]:.4f}

Final Classifier:
  H(x) = sign({alphas[0]:.4f} × h_1(x) + {alphas[1]:.4f} × h_2(x) + {alphas[2]:.4f} × h_3(x))

Key Properties:
• AdaBoost focuses on misclassified samples
• Each weak classifier is weighted by its accuracy
• Lower error → higher weight (α)
• Final classifier is weighted vote of weak classifiers
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform AdaBoost iterations step by step
2. Calculate α from given ε
3. Update weights after misclassification
4. Combine weak classifiers into final prediction

Important Formulas:
• ε_t = Σ w_i × [h_t(x_i) ≠ y_i]  (weighted error)
• α_t = 0.5 × ln((1-ε_t)/ε_t)     (classifier weight)
• w_i ← w_i × exp(-α_t × y_i × h_t(x_i))  (weight update)

Key Insights:
• α_t > 0 when ε_t < 0.5 (good classifier)
• α_t < 0 when ε_t > 0.5 (bad classifier, flip prediction)
• Correctly classified: weight decreases (factor < 1)
• Misclassified: weight increases (factor > 1)

Quick Checks:
• Weights always sum to 1 after normalization
• α = 0 when ε = 0.5 (random guessing)
• Final H(x) uses weighted voting
""")

    return alphas, classifiers


if __name__ == "__main__":
    adaboost_algorithm()
