"""
Gradient Descent Single Sample Rule - Step-by-Step Solution
============================================================
Pattern Recognition Exam Practice

Problem Type:
- Two-class classification
- Use gradient descent with single sample rule
- Find linear discriminant (decision boundary)
- With normalized samples (augmented vectors)

Key Concepts:
- Normalized Samples: y = [1, x1, x2, ..., xd]^T (add bias as first element)
- Single Sample Rule: Update weights one sample at a time
- Misclassified: a^T * y ≤ 0 (dot product with weight vector)

Key Formulas:
    Weight Update: a(k+1) = a(k) + η(k) * y_i

    where y_i is the normalized sample that was misclassified

    Decision: If a^T * y > 0 → correctly classified
              If a^T * y ≤ 0 → misclassified → update weights
"""

import numpy as np


def gradient_descent_single_sample():
    print("=" * 70)
    print("Gradient Descent Single Sample Rule - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Class ω1 samples (label = +1)
    X1 = np.array([
        [1, 1, 1],
        [1, -1, 1]
    ])

    # Class ω2 samples (label = -1)
    X2 = np.array([
        [-1, 1, 1],
        [-1, -1, -1]
    ])

    print("\nClass ω₁ (label = +1):")
    for i, x in enumerate(X1):
        print(f"  x₁^{i+1} = {x}")

    print("\nClass ω₂ (label = -1):")
    for i, x in enumerate(X2):
        print(f"  x₂^{i+1} = {x}")

    # ===== Step 2: Normalize Samples =====
    print("\n" + "="*70)
    print("STEP 2: Normalize Samples (Augmented Vectors)")
    print("="*70)

    print("""
Normalization for Linear Discriminant:
  - Add a 1 at the beginning of each sample for bias term
  - Multiply by class label (+1 for ω₁, -1 for ω₂)

  y_i = label_i × [1, x_i]^T

This transforms the problem so that:
  - All samples should satisfy: a^T * y > 0
  - If a^T * y ≤ 0 → misclassified → update weights
""")

    # Normalize samples: add 1 for bias, multiply by label
    Y1 = np.array([[1] + list(x) for x in X1])   # Add bias term, label = +1
    Y2 = np.array([[-1] + list(-np.array(x)) for x in X2])  # Add bias, multiply by -1

    print("Normalized samples for ω₁ (y = +1 × [1, x]^T):")
    for i, y in enumerate(Y1):
        print(f"  y₁^{i+1} = +1 × [1, {X1[i][0]}, {X1[i][1]}, {X1[i][2]}]^T = {y}")

    print("\nNormalized samples for ω₂ (y = -1 × [1, x]^T):")
    for i, y in enumerate(Y2):
        print(f"  y₂^{i+1} = -1 × [1, {X2[i][0]}, {X2[i][1]}, {X2[i][2]}]^T = {y}")

    # Combine all normalized samples in specified order
    all_Y = np.vstack([Y1, Y2])
    sample_order = ["y₁^1", "y₁^2", "y₂^1", "y₂^2"]

    print(f"\nSample order for updates: {' → '.join(sample_order)}")

    # ===== Step 3: Initialize Weights =====
    print("\n" + "="*70)
    print("STEP 3: Initialize Weight Vector")
    print("="*70)

    a = np.array([0, 0, 0, 0])
    eta = 1  # Learning rate

    print(f"\nInitial weight vector: a(1) = {a}")
    print(f"Learning rate: η(k) = {eta}")

    # ===== Step 4: Single Sample Rule Updates =====
    print("\n" + "="*70)
    print("STEP 4: Single Sample Rule - One Complete Cycle")
    print("="*70)

    print("""
Single Sample Rule:
  For each sample y_i:
    1. Calculate a^T * y_i
    2. If a^T * y_i ≤ 0 (misclassified):
       a(k+1) = a(k) + η(k) * y_i
    3. Otherwise, no update
""")

    iteration = 0

    for idx, (y, name) in enumerate(zip(all_Y, sample_order)):
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Process {name}")
        print(f"{'='*60}")

        print(f"\nCurrent weight vector: a({iteration}) = {a}")
        print(f"Current sample: {name} = {y}")

        # Calculate a^T * y
        aty = np.dot(a, y)
        print(f"\nCalculate: a^T × {name} = {a} · {y}")
        print(f"         = {' + '.join([f'{a[i]}×{y[i]}' for i in range(len(a))])}")
        print(f"         = {aty}")

        # Check if misclassified
        if aty <= 0:
            print(f"\nSince a^T × y = {aty} ≤ 0 → MISCLASSIFIED")
            print(f"\nUpdate rule: a(k+1) = a(k) + η(k) × y")
            print(f"           = {a} + {eta} × {y}")
            print(f"           = {a} + {y}")
            a = a + eta * y
            print(f"           = {a}")
        else:
            print(f"\nSince a^T × y = {aty} > 0 → CORRECTLY CLASSIFIED")
            print(f"No update needed.")
            print(f"a({iteration+1}) = {a}")

    # ===== Step 5: Final Decision Boundary =====
    print("\n" + "="*70)
    print("STEP 5: Final Decision Boundary")
    print("="*70)

    print(f"\nFinal weight vector after 1 complete cycle: a = {a}")
    print(f"\nDecision boundary: a^T × y = 0")
    print(f"  [{a[0]}, {a[1]}, {a[2]}, {a[3]}] · [1, x₁, x₂, x₃]^T = 0")
    print(f"  {a[0]} + {a[1]}x₁ + {a[2]}x₂ + {a[3]}x₃ = 0")

    # ===== Step 6: Verify Classification =====
    print("\n" + "="*70)
    print("STEP 6: Verify Classification After Update")
    print("="*70)

    print("\nVerifying all samples with final weight vector:")
    print("-" * 60)

    all_correct = True
    for y, name in zip(all_Y, sample_order):
        aty = np.dot(a, y)
        status = "CORRECT" if aty > 0 else "MISCLASSIFIED"
        if aty <= 0:
            all_correct = False
        print(f"  {name}: a^T × y = {aty:+.2f} → {status}")

    if all_correct:
        print("\n✓ All samples correctly classified after 1 cycle!")
    else:
        print("\n✗ Some samples still misclassified. May need more cycles.")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Gradient Descent Single Sample Rule")
    print("="*70)

    print(f"""
Data:
  ω₁: {X1.tolist()}
  ω₂: {X2.tolist()}

Normalized Samples (y = label × [1, x]^T):
  {Y1.tolist()} (ω₁)
  {Y2.tolist()} (ω₂)

Initial: a(1) = [0, 0, 0, 0]^T
Learning Rate: η = 1

Update Rule: a(k+1) = a(k) + η × y_i  (if a^T × y_i ≤ 0)

Final Weight: a = {a.tolist()}

Decision Boundary:
  {a[0]} + {a[1]}x₁ + {a[2]}x₂ + {a[3]}x₃ = 0
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Key Steps to Remember:
1. Normalize samples: y = label × [1, x₁, x₂, ...]^T
2. Initialize weight vector a (usually zeros)
3. For each sample in order:
   - Calculate a^T × y
   - If ≤ 0: update a = a + η × y
   - If > 0: no update
4. Continue until all classified or max iterations

Common Mistakes:
- Forgetting to multiply by label when normalizing
- Wrong sign in the update rule
- Not adding the bias term (1)

Important Formulas:
- Normalized sample: y = label × [1, x]^T
- Classification: a^T × y > 0 → correct, ≤ 0 → misclassified
- Update: a(k+1) = a(k) + η × y

For ω₁ (label +1): y = [1, x₁, x₂, ...]^T
For ω₂ (label -1): y = [-1, -x₁, -x₂, ...]^T
""")

    return a


if __name__ == "__main__":
    gradient_descent_single_sample()
