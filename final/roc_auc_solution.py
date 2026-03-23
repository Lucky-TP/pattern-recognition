"""
ROC Analysis and AUC Calculation - Step-by-Step Solution
=========================================================
Pattern Recognition Exam Practice

Key Concepts:
- ROC Curve: Plot of TPR (Recall) vs FPR at different thresholds
- AUC: Area Under the ROC Curve (measures classifier quality)

Key Formulas:
    TPR (True Positive Rate) = TP / (TP + FN) = Recall = Sensitivity
    FPR (False Positive Rate) = FP / (FP + TN) = 1 - Specificity

    AUC = ∫ TPR(FPR) d(FPR)  (trapezoidal rule for calculation)
"""

import numpy as np


def roc_auc_analysis():
    print("=" * 70)
    print("ROC Analysis and AUC Calculation - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Classifier Scores =====
    print("\n" + "="*70)
    print("STEP 1: Define Classifier Output Scores")
    print("="*70)

    # Binary classification with scores (probabilities or confidence)
    # Higher score = more confident in positive class
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_scores = np.array([0.95, 0.90, 0.80, 0.70, 0.55,
                         0.45, 0.30, 0.20, 0.15, 0.05])

    print("\nClassifier scores for each sample:")
    print("-" * 50)
    print("  Sample    True Class    Score    Rank")
    print("-" * 50)

    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        print(f"    {idx+1:2d}         {y_true[idx]}        {y_scores[idx]:.2f}      {rank}")

    print(f"\nTotal positives (P): {np.sum(y_true == 1)}")
    print(f"Total negatives (N): {np.sum(y_true == 0)}")

    # ===== Step 2: Calculate TPR and FPR at Different Thresholds =====
    print("\n" + "="*70)
    print("STEP 2: Calculate TPR and FPR at Different Thresholds")
    print("="*70)

    print("""
Key Formulas:
  TPR (Sensitivity) = TP / P = TP / (TP + FN)
  FPR (1-Specificity) = FP / N = FP / (FP + TN)

Where P = total positives, N = total negatives
""")

    # Define thresholds (between consecutive scores + boundaries)
    thresholds = [1.0] + list(y_scores) + [0.0]
    thresholds = sorted(set(thresholds), reverse=True)

    print("\nCalculating TPR and FPR at each threshold:")
    print("-" * 80)
    print("  Threshold    TP    FP    TN    FN    TPR      FPR")
    print("-" * 80)

    roc_points = []

    for threshold in thresholds[:8]:  # Show first 8 thresholds
        y_pred = (y_scores >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0

        roc_points.append((FPR, TPR))

        print(f"   {threshold:.2f}        {TP}     {FP}     {TN}     {FN}    {TPR:.4f}   {FPR:.4f}")

    # ===== Step 3: Plot ROC Curve Points =====
    print("\n" + "="*70)
    print("STEP 3: ROC Curve Points")
    print("="*70)

    # Get unique points sorted by FPR
    roc_points = sorted(set(roc_points))

    print("\nROC Curve points (FPR, TPR):")
    print("-" * 40)
    for fpr, tpr in roc_points:
        print(f"  ({fpr:.4f}, {tpr:.4f})")

    print("""
Visual representation of ROC curve:

  TPR
  1.0 |                    ●●●●
      |                 ●●
      |              ●●
  0.5 |           ●●
      |        ●●
      |     ●●
      |  ●●
  0.0 |●
      +-------------------- FPR
         0.0  0.5  1.0
""")

    # ===== Step 4: Calculate AUC (Trapezoidal Rule) =====
    print("\n" + "="*70)
    print("STEP 4: Calculate AUC using Trapezoidal Rule")
    print("="*70)

    print("""
Trapezoidal Rule for AUC:
  AUC = Σ (FPR_{i+1} - FPR_i) × (TPR_i + TPR_{i+1}) / 2

This calculates the area under the ROC curve by summing
trapezoid areas between consecutive points.
""")

    print("\nCalculation:")
    print("-" * 70)
    print("  i    FPR_i    TPR_i    ΔFPR     Area_i")
    print("-" * 70)

    auc = 0
    for i in range(len(roc_points) - 1):
        fpr1, tpr1 = roc_points[i]
        fpr2, tpr2 = roc_points[i + 1]

        delta_fpr = fpr2 - fpr1
        area = delta_fpr * (tpr1 + tpr2) / 2
        auc += area

        print(f"  {i}    {fpr1:.4f}   {tpr1:.4f}   {delta_fpr:.4f}   {area:.4f}")

    print("-" * 70)
    print(f"  Total AUC = {auc:.4f}")

    # ===== Step 5: Interpretation =====
    print("\n" + "="*70)
    print("STEP 5: AUC Interpretation")
    print("="*70)

    print(f"""
AUC = {auc:.4f}

Interpretation:
  AUC = 1.0  : Perfect classifier
  AUC = 0.9+ : Excellent classifier
  AUC = 0.8+ : Good classifier
  AUC = 0.7+ : Fair classifier
  AUC = 0.5  : Random guessing (diagonal line)
  AUC < 0.5  : Worse than random (flip predictions)

Current AUC = {auc:.4f} → {"Excellent" if auc > 0.9 else "Good" if auc > 0.8 else "Fair" if auc > 0.7 else "Random"} classifier
""")

    # ===== Step 6: Compare Classifiers =====
    print("\n" + "="*70)
    print("STEP 6: Comparing Multiple Classifiers")
    print("="*70)

    print("""
Example: Compare 3 classifiers by their ROC curves

  TPR
  1.0 |              C1
      |           ●●●
      |        ●●  ●●
      |      ●●    ●● C2
  0.5 |    ●●   ●●●●
      |  ●●  ●●●●   ●● C3
      | ●●●●●●   ●●●●
  0.0 |●●●●●●●●●●●●●●●●
      +-------------------- FPR
         0.0  0.5  1.0

  AUC(C1) > AUC(C2) > AUC(C3)

Higher AUC = Better overall performance across all thresholds
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: ROC Analysis and AUC")
    print("="*70)

    print(f"""
Data: {len(y_true)} samples ({np.sum(y_true==1)} positive, {np.sum(y_true==0)} negative)

ROC Curve:
  X-axis: FPR = FP / N (False Positive Rate)
  Y-axis: TPR = TP / P (True Positive Rate / Recall)

AUC Calculation:
  Method: Trapezoidal rule
  Result: AUC = {auc:.4f}

Key Properties:
  • AUC is threshold-independent
  • AUC = Probability that classifier ranks random positive
          higher than random negative
  • Good for comparing classifiers on imbalanced data
  • AUC = 0.5 is the diagonal (random guessing)
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Calculate TPR and FPR at given threshold
2. Draw ROC curve from given points
3. Calculate AUC using trapezoidal rule
4. Compare classifiers using AUC

Important Points:
• ROC shows trade-off between TPR and FPR
• Upper-left corner (0,1) is perfect classification
• Diagonal line (y=x) is random guessing
• AUC summarizes ROC curve in single number

Quick Calculations:
• TPR = Sensitivity = Recall = TP/P
• FPR = 1 - Specificity = FP/N
• TNR = Specificity = TN/N

When to use ROC/AUC:
• Imbalanced classes (AUC not affected by class distribution)
• Comparing classifiers across all thresholds
• When you care about ranking quality
""")

    return auc, roc_points


if __name__ == "__main__":
    roc_auc_analysis()
