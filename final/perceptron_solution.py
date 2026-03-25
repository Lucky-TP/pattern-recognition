"""
Perceptron Algorithm - Step-by-Step Solution
============================================
Pattern Recognition Exam Practice

Algorithm:
1. Initialize weights w = 0 and bias b = 0
2. For each misclassified point (x_i, y_i):
   - w(k+1) = w(k) + η * y_i * x_i
   - b(k+1) = b(k) + η * y_i
3. Repeat until all points correctly classified (linearly separable)

Key Formulas:
    Decision: y_pred = sign(w·x + b)
    Update:   w(k+1) = w(k) + η * y_i * x_i  (for misclassified)
              b(k+1) = b(k) + η * y_i

Convergence: Guaranteed for linearly separable data (Perceptron Convergence Theorem)
"""

import numpy as np


def perceptron_algorithm():
    print("=" * 70)
    print("Perceptron Algorithm - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Training Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data (Linearly Separable)")
    print("="*70)

    # Training data: 2D points with labels (+1 or -1)
    X = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [5, 4],
        [6, 5],
        [5, 5],
        [6, 6],
    ])

    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])  # Class labels: -1 or +1

    print("\nTraining Data:")
    print("-" * 50)
    print("  Point    Features (x1, x2)    Class (y)")
    print("-" * 50)

    class1_points = []
    class2_points = []
    for i, (x, label) in enumerate(zip(X, y)):
        sign = "+" if label == 1 else "-"
        print(f"  x_{i+1}        ({x[0]}, {x[1]})            {sign}1")
        if label == -1:
            class1_points.append(x)
        else:
            class2_points.append(x)

    print(f"\n  Class ω₁ (y = -1): {[tuple(int(v) for v in x) for x in class1_points]}")
    print(f"  Class ω₂ (y = +1): {[tuple(int(v) for v in x) for x in class2_points]}")

    # ===== Step 2: Initialize Weights =====
    print("\n" + "="*70)
    print("STEP 2: Initialize Weights and Bias")
    print("="*70)

    # Initialize
    w = np.array([0.0, 0.0])  # Weight vector
    b = 0.0                    # Bias
    eta = 1.0                  # Learning rate

    print(f"\nInitial weights: w = [{w[0]}, {w[1]}]^T")
    print(f"Initial bias: b = {b}")
    print(f"Learning rate: η = {eta}")

    print("\nDecision function: f(x) = sign(w·x + b)")
    print("                  = sign(w1*x1 + w2*x2 + b)")

    # ===== Step 3: Perceptron Iterations =====
    print("\n" + "="*70)
    print("STEP 3: Perceptron Learning Iterations")
    print("="*70)

    iteration = 0
    max_iterations = 20  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}")
        print(f"{'='*70}")

        print(f"\nCurrent weights: w = [{w[0]:.2f}, {w[1]:.2f}]^T")
        print(f"Current bias: b = {b:.2f}")

        # Check all points
        misclassified = []
        predictions = []

        print("\nEvaluating all training points:")
        print("-" * 70)
        print("  Point    (x1, x2)    y    w·x+b        sign    Correct?")
        print("-" * 70)

        for i, (x, label) in enumerate(zip(X, y)):
            # Calculate w·x + b
            wx = np.dot(w, x) + b
            prediction = 1 if wx >= 0 else -1
            correct = prediction == label
            predictions.append(prediction)

            correct_str = "✓ Yes" if correct else "✗ No (MISCLASSIFIED)"
            print(f"  x_{i+1}     ({x[0]}, {x[1]})     {label:+d}   {wx:+.2f}       {prediction:+d}     {correct_str}")

            if not correct:
                misclassified.append((i, x, label))

        # Check convergence
        if not misclassified:
            print(f"\n*** All points correctly classified! ***")
            print(f"*** Algorithm converged after {iteration-1} iterations ***")
            break

        # Update using first misclassified point
        idx, x_mis, y_mis = misclassified[0]
        print(f"\nUpdating using misclassified point x_{idx+1}:")
        print(f"  x_{idx+1} = ({x_mis[0]}, {x_mis[1]}), y = {y_mis:+d}")

        # Show update calculation
        print(f"\nWeight update:")
        print(f"  w_new = w_old + η × y × x")
        print(f"       = [{w[0]:.2f}, {w[1]:.2f}] + {eta} × ({y_mis:+d}) × ({x_mis[0]}, {x_mis[1]})")
        print(f"       = [{w[0]:.2f}, {w[1]:.2f}] + ({eta * y_mis * x_mis[0]:.2f}, {eta * y_mis * x_mis[1]:.2f})")
        print(f"       = [{w[0] + eta * y_mis * x_mis[0]:.2f}, {w[1] + eta * y_mis * x_mis[1]:.2f}]")

        print(f"\nBias update:")
        print(f"  b_new = b_old + η × y")
        print(f"       = {b:.2f} + {eta} × ({y_mis:+d})")
        print(f"       = {b:.2f} + ({eta * y_mis:.2f})")
        print(f"       = {b + eta * y_mis:.2f}")

        # Apply update
        w = w + eta * y_mis * x_mis
        b = b + eta * y_mis

    # ===== Step 4: Final Decision Boundary =====
    print("\n" + "="*70)
    print("STEP 4: Final Decision Boundary")
    print("="*70)

    print(f"\nFinal weights: w = [{w[0]:.2f}, {w[1]:.2f}]^T")
    print(f"Final bias: b = {b:.2f}")

    print(f"\nDecision function:")
    print(f"  f(x) = sign({w[0]:.2f}·x1 + {w[1]:.2f}·x2 + {b:.2f})")

    print(f"\nDecision boundary equation:")
    print(f"  {w[0]:.2f}·x1 + {w[1]:.2f}·x2 + {b:.2f} = 0")

    if w[1] != 0:
        x2_intercept = -b / w[1]
        slope = -w[0] / w[1]
        print(f"\n  In slope-intercept form:")
        print(f"    x2 = {slope:.2f}·x1 + {x2_intercept:.2f}")
        print(f"    Slope: {slope:.2f}")
        print(f"    Intercept: {x2_intercept:.2f}")

    # ===== Step 5: Verify Classification =====
    print("\n" + "="*70)
    print("STEP 5: Verify Final Classification")
    print("="*70)

    print("\nFinal classification of all points:")
    print("-" * 50)

    correct_count = 0
    for i, (x, label) in enumerate(zip(X, y)):
        wx = np.dot(w, x) + b
        prediction = 1 if wx >= 0 else -1
        correct = prediction == label
        if correct:
            correct_count += 1
        print(f"  x_{i+1} ({x[0]}, {x[1]}): f(x) = {wx:+.2f} → class {prediction:+d} (true: {label:+d}) {'✓' if correct else '✗'}")

    accuracy = correct_count / len(X) * 100
    print(f"\nAccuracy: {correct_count}/{len(X)} = {accuracy:.1f}%")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Perceptron Algorithm")
    print("="*70)

    print(f"""
Training Data:
  Class ω₁ (y=-1): {[tuple(int(v) for v in x) for x in class1_points]}
  Class ω₂ (y=+1): {[tuple(int(v) for v in x) for x in class2_points]}

Final Model:
  Weights: w = [{w[0]:.2f}, {w[1]:.2f}]^T
  Bias: b = {b:.2f}

Decision Boundary:
  {w[0]:.2f}·x1 + {w[1]:.2f}·x2 + {b:.2f} = 0

Convergence:
  Iterations: {iteration}
  (Perceptron converged - data is linearly separable)

Key Properties:
• Guaranteed convergence for linearly separable data
• Solution is not unique (depends on initialization and order)
• Single-layer perceptron can only learn linear boundaries
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform perceptron iterations step by step
2. Draw decision boundary from final weights
3. Determine if data is linearly separable
4. Compare perceptron with other linear classifiers

Important Points:
• Labels must be {-1, +1} (or {0, 1} with modified update)
• Update only on misclassified points
• Convergence guaranteed ONLY for linearly separable data
• If not separable: algorithm cycles (won't converge)

Quick Checks:
• Decision boundary is perpendicular to weight vector w
• Positive class (y=+1) is in direction of w
• If all points classified, any scaling of (w,b) works
• Learning rate η only affects speed, not final solution
""")

    return w, b


if __name__ == "__main__":
    perceptron_algorithm()
