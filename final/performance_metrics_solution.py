"""
Classification Performance Metrics - Step-by-Step Solution
===========================================================
Pattern Recognition Exam Practice

Metrics Covered:
- Confusion Matrix
- Accuracy, Precision, Recall, F1-score
- Macro vs Micro averaging
- Specificity, Sensitivity

Key Formulas:
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)  (also called Sensitivity)
    F1-score  = 2 × (Precision × Recall) / (Precision + Recall)
    Specificity = TN / (TN + FP)
"""

import numpy as np


def performance_metrics():
    print("=" * 70)
    print("Classification Performance Metrics - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Predictions =====
    print("\n" + "="*70)
    print("STEP 1: Define True Labels and Predictions")
    print("="*70)

    # Binary classification example
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                       1, 0, 1, 0, 1, 0, 1, 1, 0, 0])

    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
                       1, 0, 1, 1, 0, 0, 1, 1, 0, 1])

    print("\nTrue labels (y_true) and Predictions (y_pred):")
    print("-" * 50)
    print("  Sample    True    Predicted    Correct?")
    print("-" * 50)

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        correct = "✓" if t == p else "✗"
        print(f"    {i+1:2d}       {t}         {p}          {correct}")

    # ===== Step 2: Build Confusion Matrix =====
    print("\n" + "="*70)
    print("STEP 2: Build Confusion Matrix (Binary)")
    print("="*70)

    # Calculate TP, TN, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    print("\nDefinitions:")
    print("  TP (True Positive):  Actual = 1, Predicted = 1")
    print("  TN (True Negative):  Actual = 0, Predicted = 0")
    print("  FP (False Positive): Actual = 0, Predicted = 1 (Type I error)")
    print("  FN (False Negative): Actual = 1, Predicted = 0 (Type II error)")

    print("\nCounting each category:")
    print(f"  TP: {[(i+1) for i, (t,p) in enumerate(zip(y_true,y_pred)) if t==1 and p==1]}")
    print(f"  TN: {[(i+1) for i, (t,p) in enumerate(zip(y_true,y_pred)) if t==0 and p==0]}")
    print(f"  FP: {[(i+1) for i, (t,p) in enumerate(zip(y_true,y_pred)) if t==0 and p==1]}")
    print(f"  FN: {[(i+1) for i, (t,p) in enumerate(zip(y_true,y_pred)) if t==1 and p==0]}")

    print(f"\nConfusion Matrix:")
    print("-" * 50)
    print("                  Predicted")
    print("                 Class 0   Class 1")
    print("    Actual")
    print(f"    Class 0      {TN:5d}      {FP:5d}")
    print(f"    Class 1      {FN:5d}      {TP:5d}")

    print(f"\nSummary:")
    print(f"  TP = {TP}")
    print(f"  TN = {TN}")
    print(f"  FP = {FP}")
    print(f"  FN = {FN}")

    # ===== Step 3: Calculate Metrics =====
    print("\n" + "="*70)
    print("STEP 3: Calculate Performance Metrics")
    print("="*70)

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"\n--- Accuracy ---")
    print(f"  Formula: (TP + TN) / (TP + TN + FP + FN)")
    print(f"  = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN})")
    print(f"  = {TP + TN} / {TP + TN + FP + FN}")
    print(f"  = {accuracy:.4f} = {accuracy*100:.2f}%")

    # Precision (Positive Predictive Value)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    print(f"\n--- Precision (Positive Predictive Value) ---")
    print(f"  Formula: TP / (TP + FP)")
    print(f"  = {TP} / ({TP} + {FP})")
    print(f"  = {TP} / {TP + FP}")
    print(f"  = {precision:.4f} = {precision*100:.2f}%")
    print(f"  Interpretation: Of all predicted positives, {precision*100:.1f}% were correct")

    # Recall (Sensitivity, True Positive Rate)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    print(f"\n--- Recall (Sensitivity, TPR) ---")
    print(f"  Formula: TP / (TP + FN)")
    print(f"  = {TP} / ({TP} + {FN})")
    print(f"  = {TP} / {TP + FN}")
    print(f"  = {recall:.4f} = {recall*100:.2f}%")
    print(f"  Interpretation: Of all actual positives, {recall*100:.1f}% were detected")

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    print(f"\n--- Specificity (True Negative Rate) ---")
    print(f"  Formula: TN / (TN + FP)")
    print(f"  = {TN} / ({TN} + {FP})")
    print(f"  = {TN} / {TN + FP}")
    print(f"  = {specificity:.4f} = {specificity*100:.2f}%")
    print(f"  Interpretation: Of all actual negatives, {specificity*100:.1f}% were detected")

    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n--- F1-Score ---")
    print(f"  Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"  = 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"  = 2 × {precision*recall:.4f} / {precision+recall:.4f}")
    print(f"  = {f1:.4f} = {f1*100:.2f}%")
    print(f"  Interpretation: Harmonic mean of precision and recall")

    # ===== Step 4: Multi-class Example =====
    print("\n" + "="*70)
    print("STEP 4: Multi-class Metrics (3 Classes)")
    print("="*70)

    y_true_multi = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred_multi = np.array([0, 0, 1, 1, 1, 1, 2, 0, 2, 2, 0, 1])

    print("\nMulti-class data (3 classes: 0, 1, 2):")
    print("-" * 40)
    print("  Sample    True    Predicted")
    print("-" * 40)
    for i, (t, p) in enumerate(zip(y_true_multi, y_pred_multi)):
        marker = "" if t == p else " ✗"
        print(f"    {i+1:2d}       {t}         {p}       {marker}")

    # Build multi-class confusion matrix
    n_classes = 3
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_multi, y_pred_multi):
        cm[t, p] += 1

    print(f"\nMulti-class Confusion Matrix:")
    print("-" * 50)
    print("                  Predicted")
    print(f"                 Class 0  Class 1  Class 2")
    print("    Actual")
    for i in range(n_classes):
        row = f"    Class {i}     "
        for j in range(n_classes):
            row += f"  {cm[i,j]:5d}    "
        print(row)

    # Per-class metrics
    print(f"\n--- Per-Class Metrics ---")
    class_metrics = []
    for c in range(n_classes):
        tp_c = cm[c, c]
        fp_c = np.sum(cm[:, c]) - tp_c
        fn_c = np.sum(cm[c, :]) - tp_c
        tn_c = np.sum(cm) - tp_c - fp_c - fn_c

        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0

        class_metrics.append({'precision': prec_c, 'recall': rec_c, 'f1': f1_c})

        print(f"\nClass {c}:")
        print(f"  TP={tp_c}, FP={fp_c}, FN={fn_c}")
        print(f"  Precision = {prec_c:.4f}")
        print(f"  Recall    = {rec_c:.4f}")
        print(f"  F1-score  = {f1_c:.4f}")

    # Macro averaging
    print(f"\n--- Macro Averaging ---")
    print("(Average of per-class metrics)")
    macro_precision = np.mean([m['precision'] for m in class_metrics])
    macro_recall = np.mean([m['recall'] for m in class_metrics])
    macro_f1 = np.mean([m['f1'] for m in class_metrics])

    print(f"  Macro Precision = ({' + '.join([f'{m["precision"]:.4f}' for m in class_metrics])}) / {n_classes}")
    print(f"                   = {macro_precision:.4f}")
    print(f"  Macro Recall    = {macro_recall:.4f}")
    print(f"  Macro F1        = {macro_f1:.4f}")

    # Micro averaging
    print(f"\n--- Micro Averaging ---")
    print("(Calculate globally by counting total TP, FP, FN)")
    total_tp = np.sum([cm[c, c] for c in range(n_classes)])
    total_fp = np.sum([np.sum(cm[:, c]) - cm[c, c] for c in range(n_classes)])
    total_fn = np.sum([np.sum(cm[c, :]) - cm[c, c] for c in range(n_classes)])

    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    print(f"  Total TP = {total_tp}, Total FP = {total_fp}, Total FN = {total_fn}")
    print(f"  Micro Precision = {total_tp} / ({total_tp} + {total_fp}) = {micro_precision:.4f}")
    print(f"  Micro Recall    = {total_tp} / ({total_tp} + {total_fn}) = {micro_recall:.4f}")
    print(f"  Micro F1        = {micro_f1:.4f}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Performance Metrics")
    print("="*70)

    print(f"""
Binary Classification Results:
  TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}

  Accuracy    = {accuracy:.4f} ({accuracy*100:.2f}%)
  Precision   = {precision:.4f}
  Recall      = {recall:.4f}
  Specificity = {specificity:.4f}
  F1-score    = {f1:.4f}

Multi-class Averaging:
  Macro F1 = {macro_f1:.4f}  (treats all classes equally)
  Micro F1 = {micro_f1:.4f}  (weighted by class frequency)

When to use which metric:
  • Accuracy: When classes are balanced
  • Precision: When FP is costly (e.g., spam detection)
  • Recall: When FN is costly (e.g., disease detection)
  • F1-score: When you need balance between precision and recall
  • Macro avg: When all classes are equally important
  • Micro avg: When you want to weight by class size
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Build confusion matrix from predictions
2. Calculate all metrics from confusion matrix
3. Compare macro vs micro averaging
4. Explain when to use precision vs recall

Key Relationships:
  • Precision = TP / (TP + FP) = "How many predicted positives are correct?"
  • Recall = TP / (TP + FN) = "How many actual positives did we find?"
  • F1 = Harmonic mean (penalizes extreme values)
  • Micro F1 = Accuracy for multi-class

Tricky Points:
  • Precision and Recall are inversely related
  • Macro average treats all classes equally
  • Micro average is dominated by larger classes
  • F1 = 0 if precision = 0 or recall = 0
""")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    performance_metrics()
