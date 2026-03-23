"""
Bayesian Classifier with Unequal Priors - Step-by-Step Solution
================================================================
Pattern Recognition Exam Practice

Problem:
- μ₁ = [1,2]ᵀ, μ₂ = [6,6]ᵀ
- Σ = 2I (equal covariance, σ² = 2)
- P(ω₁) = 0.8, P(ω₂) = 0.2
- Find decision boundary

Key Formula (Equal Covariance Case):
    g(x) = wᵀx + w₀
    where: w = Σ⁻¹(μ₁ - μ₂)
           w₀ = -½(μ₁ᵀΣ⁻¹μ₁ - μ₂ᵀΣ⁻¹μ₂) + ln(P(ω₁)/P(ω₂))

Decision: Choose ω₁ if g(x) > 0
"""

import numpy as np
import matplotlib.pyplot as plt

def print_matrix(name, M):
    """Pretty print a matrix with its name"""
    print(f"\n{name} =")
    if M.ndim == 1:
        print(f"  [{M[0]:.4f}, {M[1]:.4f}]ᵀ")
    else:
        for row in M:
            print(f"  [{row[0]:8.4f}, {row[1]:8.4f}]")

def bayesian_unequal_priors():
    print("=" * 70)
    print("Bayesian Classifier with Unequal Priors - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define the problem =====
    print("\n" + "="*70)
    print("STEP 1: Problem Definition")
    print("="*70)

    mu1 = np.array([1, 2])
    mu2 = np.array([6, 6])
    sigma_sq = 2
    Sigma = sigma_sq * np.eye(2)
    P_w1 = 0.8
    P_w2 = 0.2

    print(f"""
Given:
  Class 1 (ω₁): μ₁ = [1, 2]ᵀ
  Class 2 (ω₂): μ₂ = [6, 6]ᵀ
  Common covariance: Σ = {sigma_sq}I = [[{sigma_sq}, 0], [0, {sigma_sq}]]
  Prior probabilities: P(ω₁) = {P_w1}, P(ω₂) = {P_w2}

Find: Decision boundary and classify test points
""")

    # ===== Step 2: Review Bayesian Decision Theory =====
    print("\n" + "="*70)
    print("STEP 2: Bayesian Decision Theory Review")
    print("="*70)

    print("""
Bayes Decision Rule:
  Choose ω₁ if P(ω₁|x) > P(ω₂|x)

Using Bayes' theorem:
  P(ωᵢ|x) = P(x|ωᵢ)P(ωᵢ) / P(x)

This is equivalent to (taking log):
  Choose ω₁ if ln P(x|ω₁) + ln P(ω₁) > ln P(x|ω₂) + ln P(ω₂)

For Gaussian distributions with equal Σ:
  ln P(x|ωᵢ) = -½(x-μᵢ)ᵀΣ⁻¹(x-μᵢ) + constant

The discriminant function becomes:
  g(x) = wᵀx + w₀

where:
  w = Σ⁻¹(μ₁ - μ₂)
  w₀ = -½μ₁ᵀΣ⁻¹μ₁ + ½μ₂ᵀΣ⁻¹μ₂ + ln(P(ω₁)/P(ω₂))
""")

    # ===== Step 3: Calculate Σ⁻¹ =====
    print("\n" + "="*70)
    print("STEP 3: Calculate Σ⁻¹")
    print("="*70)

    Sigma_inv = np.linalg.inv(Sigma)

    print(f"\nΣ = {sigma_sq}I")
    print(f"Σ⁻¹ = (1/{sigma_sq})I")
    print_matrix("Σ⁻¹", Sigma_inv)

    # ===== Step 4: Calculate w =====
    print("\n" + "="*70)
    print("STEP 4: Calculate weight vector w = Σ⁻¹(μ₁ - μ₂)")
    print("="*70)

    mu_diff = mu1 - mu2
    w = Sigma_inv @ mu_diff

    print(f"\nμ₁ - μ₂ = [{mu1[0]}, {mu1[1]}] - [{mu2[0]}, {mu2[1]}]")
    print(f"        = [{mu_diff[0]}, {mu_diff[1]}]")

    print(f"\nw = Σ⁻¹(μ₁ - μ₂)")
    print(f"  = (1/{sigma_sq}) × [{mu_diff[0]}, {mu_diff[1]}]")
    print(f"  = [{w[0]}, {w[1]}]")

    # ===== Step 5: Calculate w₀ =====
    print("\n" + "="*70)
    print("STEP 5: Calculate bias term w₀")
    print("="*70)

    # w₀ = -½μ₁ᵀΣ⁻¹μ₁ + ½μ₂ᵀΣ⁻¹μ₂ + ln(P(ω₁)/P(ω₂))
    term1 = -0.5 * mu1 @ Sigma_inv @ mu1
    term2 = 0.5 * mu2 @ Sigma_inv @ mu2
    term3 = np.log(P_w1 / P_w2)
    w0 = term1 + term2 + term3

    print(f"\nw₀ = -½μ₁ᵀΣ⁻¹μ₁ + ½μ₂ᵀΣ⁻¹μ₂ + ln(P(ω₁)/P(ω₂))")

    print(f"\n  Term 1: -½μ₁ᵀΣ⁻¹μ₁")
    print(f"    μ₁ᵀΣ⁻¹μ₁ = [{mu1[0]}, {mu1[1]}] × (1/{sigma_sq})I × [{mu1[0]}, {mu1[1]}]ᵀ")
    print(f"             = (1/{sigma_sq}) × ({mu1[0]}² + {mu1[1]}²)")
    print(f"             = (1/{sigma_sq}) × {mu1[0]**2 + mu1[1]**2}")
    print(f"             = {mu1 @ Sigma_inv @ mu1}")
    print(f"    -½ × {mu1 @ Sigma_inv @ mu1} = {term1}")

    print(f"\n  Term 2: ½μ₂ᵀΣ⁻¹μ₂")
    print(f"    μ₂ᵀΣ⁻¹μ₂ = (1/{sigma_sq}) × ({mu2[0]}² + {mu2[1]}²)")
    print(f"             = (1/{sigma_sq}) × {mu2[0]**2 + mu2[1]**2}")
    print(f"             = {mu2 @ Sigma_inv @ mu2}")
    print(f"    ½ × {mu2 @ Sigma_inv @ mu2} = {term2}")

    print(f"\n  Term 3: ln(P(ω₁)/P(ω₂))")
    print(f"    = ln({P_w1}/{P_w2})")
    print(f"    = ln({P_w1/P_w2})")
    print(f"    = {term3:.4f}")

    print(f"\n  w₀ = {term1} + {term2} + {term3:.4f}")
    print(f"     = {w0:.4f}")

    # ===== Step 6: Decision boundary equation =====
    print("\n" + "="*70)
    print("STEP 6: Decision Boundary Equation")
    print("="*70)

    print(f"""
Discriminant function:
  g(x) = wᵀx + w₀
       = {w[0]}x₁ + {w[1]}x₂ + {w0:.4f}

Decision boundary (g(x) = 0):
  {w[0]}x₁ + {w[1]}x₂ + {w0:.4f} = 0

Solving for x₂:
  x₂ = -({w[0]}/{w[1]})x₁ - ({w0:.4f}/{w[1]})
  x₂ = {-w[0]/w[1]:.4f}x₁ + {-w0/w[1]:.4f}

Decision rule:
  Choose ω₁ if g(x) > 0 (above the boundary)
  Choose ω₂ if g(x) < 0 (below the boundary)
""")

    # ===== Step 7: Compare with equal priors =====
    print("\n" + "="*70)
    print("STEP 7: Compare with Equal Priors Case")
    print("="*70)

    # Equal priors case
    w0_equal = -0.5 * mu1 @ Sigma_inv @ mu1 + 0.5 * mu2 @ Sigma_inv @ mu2 + np.log(0.5/0.5)

    print(f"""
With equal priors P(ω₁) = P(ω₂) = 0.5:
  ln(P(ω₁)/P(ω₂)) = ln(1) = 0
  w₀ = {term1} + {term2} + 0 = {w0_equal:.4f}

  Decision boundary: {w[0]}x₁ + {w[1]}x₂ + {w0_equal:.4f} = 0
  x₂ = {-w[0]/w[1]:.4f}x₁ + {-w0_equal/w[1]:.4f}

With unequal priors P(ω₁) = {P_w1}, P(ω₂) = {P_w2}:
  ln(P(ω₁)/P(ω₂)) = ln({P_w1/P_w2}) = {term3:.4f}
  w₀ = {w0:.4f}

  Decision boundary: {w[0]}x₁ + {w[1]}x₂ + {w0:.4f} = 0
  x₂ = {-w[0]/w[1]:.4f}x₁ + {-w0/w[1]:.4f}

Effect of unequal priors:
  The boundary shifts by {term3:.4f}/{abs(w[1]):.4f} = {term3/abs(w[1]):.4f} units
  Direction: toward class with LOWER prior (ω₂)
  Reason: Higher prior for ω₁ means we need less evidence to choose it
""")

    # ===== Step 8: Classify test points =====
    print("\n" + "="*70)
    print("STEP 8: Classify Test Points")
    print("="*70)

    test_points = [
        np.array([3, 3]),
        np.array([4, 4]),
        np.array([3.5, 4.5]),
        np.array([2, 3])
    ]

    print("\nTest points classification:")
    print("-" * 50)

    for x in test_points:
        g = w @ x + w0
        decision = "ω₁" if g > 0 else "ω₂"
        print(f"\nx = ({x[0]}, {x[1]})")
        print(f"  g(x) = {w[0]}×{x[0]} + {w[1]}×{x[1]} + {w0:.4f}")
        print(f"       = {w[0]*x[0]} + {w[1]*x[1]} + {w0:.4f}")
        print(f"       = {g:.4f}")
        print(f"  Decision: {decision} (g(x) {'>' if g > 0 else '<'} 0)")

    # ===== Step 9: Midpoint analysis =====
    print("\n" + "="*70)
    print("STEP 9: Midpoint Analysis")
    print("="*70)

    midpoint = (mu1 + mu2) / 2

    print(f"""
Midpoint between class means:
  M = (μ₁ + μ₂)/2 = ([{mu1[0]}, {mu1[1]}] + [{mu2[0]}, {mu2[1]}])/2
    = [{midpoint[0]}, {midpoint[1]}]

For EQUAL priors:
  Decision boundary passes through midpoint
  g(M) = {w @ midpoint + w0_equal:.4f} (≈ 0)

For UNEQUAL priors (P(ω₁) = {P_w1}, P(ω₂) = {P_w2}):
  g(M) = {w[0]}×{midpoint[0]} + {w[1]}×{midpoint[1]} + {w0:.4f}
       = {w @ midpoint + w0:.4f}

  Since g(M) > 0, the midpoint is classified as ω₁
  The boundary has shifted AWAY from μ₁ (toward μ₂)
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
Problem:
  μ₁ = [{mu1[0]}, {mu1[1]}]ᵀ, μ₂ = [{mu2[0]}, {mu2[1]}]ᵀ
  Σ = {sigma_sq}I
  P(ω₁) = {P_w1}, P(ω₂) = {P_w2}

Solution:
  w = [{w[0]}, {w[1]}]ᵀ
  w₀ = {w0:.4f}

  Discriminant function: g(x) = {w[0]}x₁ + {w[1]}x₂ + {w0:.4f}

  Decision boundary: {w[0]}x₁ + {w[1]}x₂ + {w0:.4f} = 0
                 or: x₂ = {-w[0]/w[1]:.4f}x₁ + {-w0/w[1]:.4f}

  Decision rule: Choose ω₁ if g(x) > 0, else choose ω₂

Key insight:
  Higher prior probability for ω₁ shifts the boundary toward ω₂,
  expanding the decision region for ω₁.
""")

    # ===== Visualization =====
    print("\n" + "="*70)
    print("VISUALIZATION (saved as bayesian_unequal_priors_plot.png)")
    print("="*70)

    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot class means
        ax.scatter(*mu1, s=200, c='blue', marker='o', label=f'μ₁ = ({mu1[0]}, {mu1[1]})', zorder=5)
        ax.scatter(*mu2, s=200, c='red', marker='o', label=f'μ₂ = ({mu2[0]}, {mu2[1]})', zorder=5)

        # Plot decision boundaries
        x_range = np.linspace(-2, 10, 100)

        # Unequal priors boundary
        y_unequal = (-w[0] * x_range - w0) / w[1]
        ax.plot(x_range, y_unequal, 'g-', linewidth=2,
                label=f'Unequal priors: P(ω₁)={P_w1}')

        # Equal priors boundary
        y_equal = (-w[0] * x_range - w0_equal) / w[1]
        ax.plot(x_range, y_equal, 'k--', linewidth=2,
                label='Equal priors: P(ω₁)=0.5')

        # Plot test points
        for x in test_points:
            g = w @ x + w0
            color = 'blue' if g > 0 else 'red'
            ax.scatter(*x, s=100, c=color, marker='x', zorder=5)
            ax.annotate(f'({x[0]},{x[1]})', (x[0]+0.1, x[1]+0.1))

        # Plot midpoint
        ax.scatter(*midpoint, s=100, c='purple', marker='s', label='Midpoint', zorder=5)

        ax.set_xlim(-2, 10)
        ax.set_ylim(-1, 10)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title('Bayesian Classifier: Effect of Unequal Priors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.savefig('/Users/tunchanokpunmeros/Desktop/pattern/bayesian_unequal_priors_plot.png',
                    dpi=150, bbox_inches='tight')
        print("Plot saved successfully!")
    except Exception as e:
        print(f"Could not save plot: {e}")

    return w, w0


if __name__ == "__main__":
    bayesian_unequal_priors()
