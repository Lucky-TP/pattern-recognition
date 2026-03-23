"""
Parzen Window (Kernel Density Estimation) - Step-by-Step Solution
=================================================================
Pattern Recognition Exam Practice

Algorithm:
1. Place a kernel function at each data point
2. Sum all kernels to get density estimate
3. Normalize to get probability density

Key Formulas:
    p(x) = (1/n) × (1/h^d) × Σ K((x - x_i) / h)

Where:
  - n = number of samples
  - h = window width (bandwidth)
  - d = dimensionality
  - K = kernel function (e.g., Gaussian, uniform)

Common Kernels:
    Gaussian:   K(u) = (1/√(2π)) × exp(-u²/2)
    Uniform:    K(u) = 1/2 if |u| ≤ 1, else 0
    Epanechnikov: K(u) = (3/4)(1-u²) if |u| ≤ 1, else 0
"""

import numpy as np


def gaussian_kernel(u):
    """Standard Gaussian kernel."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


def uniform_kernel(u):
    """Uniform kernel."""
    return 0.5 if np.abs(u) <= 1 else 0


def parzen_window():
    print("=" * 70)
    print("Parzen Window (Kernel Density Estimation) - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Data")
    print("="*70)

    # 1D data for simplicity
    X = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

    n = len(X)

    print("\n1D Training Data:")
    print("-" * 30)
    for i, x in enumerate(X):
        print(f"  x_{i+1} = {x:.1f}")

    print(f"\nNumber of samples: n = {n}")

    # ===== Step 2: Define Kernel and Bandwidth =====
    print("\n" + "="*70)
    print("STEP 2: Define Kernel and Bandwidth")
    print("="*70)

    h = 1.0  # Bandwidth (window width)

    print(f"\nKernel: Gaussian")
    print(f"  K(u) = (1/√(2π)) × exp(-u²/2)")

    print(f"\nBandwidth: h = {h}")

    print("""
Parzen Window Density Estimate Formula:
  p(x) = (1/n) × (1/h) × Σ K((x - x_i) / h)

For Gaussian kernel:
  p(x) = (1/n) × (1/h) × Σ (1/√(2π)) × exp(-(x - x_i)²/(2h²))
""")

    # ===== Step 3: Calculate Density at Query Points =====
    print("\n" + "="*70)
    print("STEP 3: Calculate Density at Query Points")
    print("="*70)

    # Query points
    query_points = [0.5, 2.0, 3.0, 4.5, 6.0]

    print("\nQuery points to evaluate density:")
    print("-" * 40)
    for x_q in query_points:
        print(f"  x_query = {x_q}")

    # Calculate density at each query point
    print("\n" + "="*60)
    print("Density Calculation for each query point:")
    print("="*60)

    densities = []

    for x_q in query_points:
        print(f"\n--- Query point: x = {x_q} ---")

        kernel_sum = 0
        contributions = []

        for i, x_i in enumerate(X):
            # Calculate u = (x - x_i) / h
            u = (x_q - x_i) / h

            # Calculate kernel value
            K_u = gaussian_kernel(u)

            kernel_sum += K_u
            contributions.append((x_i, u, K_u))

        # Calculate density
        density = (1 / n) * (1 / h) * kernel_sum
        densities.append(density)

        print(f"\nKernel contributions:")
        print("  x_i     u=(x-x_i)/h    K(u)")
        print("  " + "-" * 40)
        for x_i, u, K_u in contributions:
            print(f"  {x_i:.1f}     {u:+.3f}        {K_u:.4f}")

        print(f"\nSum of kernels: Σ K(u) = {kernel_sum:.4f}")
        print(f"\nDensity calculation:")
        print(f"  p({x_q}) = (1/{n}) × (1/{h}) × {kernel_sum:.4f}")
        print(f"        = {1/n:.4f} × {1/h:.4f} × {kernel_sum:.4f}")
        print(f"        = {density:.4f}")

    # ===== Step 4: Effect of Bandwidth =====
    print("\n" + "="*70)
    print("STEP 4: Effect of Bandwidth (h)")
    print("="*70)

    x_test = 3.0

    print(f"\nEvaluating density at x = {x_test} with different bandwidths:")
    print("-" * 60)
    print("  h      Σ K(u)      p(x)")
    print("-" * 60)

    for h_test in [0.2, 0.5, 1.0, 2.0, 5.0]:
        kernel_sum = 0
        for x_i in X:
            u = (x_test - x_i) / h_test
            kernel_sum += gaussian_kernel(u)

        density = (1 / n) * (1 / h_test) * kernel_sum
        print(f"  {h_test:.1f}    {kernel_sum:.4f}     {density:.4f}")

    print("""
Effect of bandwidth:
  • Small h (h → 0): Undersmoothing
    - Density has many local maxima (spiky)
    - Each point creates its own peak
    - High variance, low bias

  • Large h (h → ∞): Oversmoothing
    - Density is very smooth
    - Details are lost
    - Low variance, high bias

  • Optimal h: Balance between bias and variance
    - Cross-validation can be used to find optimal h
""")

    # ===== Step 5: Compare Kernels =====
    print("\n" + "="*70)
    print("STEP 5: Compare Different Kernels")
    print("="*70)

    x_test = 2.5
    h = 1.0

    print(f"\nEvaluating at x = {x_test}, h = {h}")
    print("-" * 60)

    # Gaussian kernel
    gaussian_sum = sum(gaussian_kernel((x_test - x_i) / h) for x_i in X)
    gaussian_density = (1 / n) * (1 / h) * gaussian_sum

    # Uniform kernel
    uniform_sum = sum(uniform_kernel((x_test - x_i) / h) for x_i in X)
    uniform_density = (1 / n) * (1 / h) * uniform_sum

    print(f"\nGaussian Kernel:")
    print(f"  Σ K((2.5 - x_i)/h) = {gaussian_sum:.4f}")
    print(f"  p(2.5) = {gaussian_density:.4f}")

    print(f"\nUniform Kernel:")
    print(f"  Σ K((2.5 - x_i)/h) = {uniform_sum:.4f}")
    print(f"  p(2.5) = {uniform_density:.4f}")

    print("""
Kernel Comparison:
  • Gaussian: Smooth, differentiable, infinite support
  • Uniform: Simple, compact support, not differentiable
  • Epanechnikov: Optimal in MSE sense, compact support

Note: Choice of kernel is less important than choice of bandwidth!
""")

    # ===== Step 6: Classification using Parzen Window =====
    print("\n" + "="*70)
    print("STEP 6: Classification using Parzen Window")
    print("="*70)

    print("""
Bayes Classifier using Parzen Window Density Estimation:

For two classes ω₁ and ω₂:
  P(x|ω₁) estimated using Parzen window on class 1 samples
  P(x|ω₂) estimated using Parzen window on class 2 samples

Decision rule:
  Choose ω₁ if P(x|ω₁) × P(ω₁) > P(x|ω₂) × P(ω₂)

This is a non-parametric classifier (no assumption about distribution)
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Parzen Window Density Estimation")
    print("="*70)

    print(f"""
Data: {n} 1D samples
Kernel: Gaussian
Bandwidth: h = {h}

Key Formula:
  p(x) = (1/n) × (1/h) × Σ K((x - x_i) / h)

Properties:
  • Non-parametric method (no distribution assumption)
  • Consistent estimator (converges to true density as n → ∞)
  • Bandwidth h is crucial parameter
  • Small h → undersmoothing (high variance)
  • Large h → oversmoothing (high bias)

Advantages:
  • Can estimate any distribution shape
  • Simple to implement
  • Only parameter is bandwidth h

Disadvantages:
  • Computationally expensive O(n) per query
  • Needs many samples for good estimate
  • Bandwidth selection is critical
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Calculate density estimate at a query point
2. Show effect of bandwidth on density shape
3. Compare Parzen window with k-NN density estimation
4. Use Parzen window for classification

Important Points:
• Parzen window = kernel density estimation
• Each data point contributes a kernel
• Bandwidth h controls smoothness
• Gaussian kernel is most common

Quick Calculations:
• For Gaussian: K(u) = (1/√(2π)) × exp(-u²/2)
• For uniform: K(u) = 0.5 if |u| ≤ 1
• Density = (1/nh) × Σ K((x-x_i)/h)

Comparison with k-NN:
• Parzen: Fixed h, variable number of neighbors
• k-NN: Fixed k, variable radius
• Both are non-parametric density estimators
""")

    return densities


if __name__ == "__main__":
    parzen_window()
