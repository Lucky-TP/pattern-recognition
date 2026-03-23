"""
Bayesian Classifier with Unequal Variances - Quadratic Boundary
================================================================
Pattern Recognition Exam Practice

Problem:
- 1D Gaussian classes with SAME mean but DIFFERENT variances
- μ₁ = 0, σ₁² = 1
- μ₂ = 0, σ₂² = 4
- Equal priors: P(ω₁) = P(ω₂) = 0.5
- Results in QUADRATIC decision boundary with 2 decision points

Key Insight:
- Same mean doesn't mean classes are identical!
- Variance difference creates two decision regions
- This is a classic exam problem testing understanding of Gaussian properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bayesian_quadratic_boundary():
    print("=" * 70)
    print("Bayesian Classifier: Quadratic Boundary (Unequal Variances)")
    print("=" * 70)

    # ===== Step 1: Problem Definition =====
    print("\n" + "="*70)
    print("STEP 1: Problem Definition")
    print("="*70)

    mu1 = 0
    sigma1_sq = 1
    sigma1 = np.sqrt(sigma1_sq)

    mu2 = 0
    sigma2_sq = 4
    sigma2 = np.sqrt(sigma2_sq)

    P_w1 = 0.5
    P_w2 = 0.5

    print(f"""
Given two 1D Gaussian classes:

Class ω₁:
  Mean: μ₁ = {mu1}
  Variance: σ₁² = {sigma1_sq}
  Standard deviation: σ₁ = {sigma1}

Class ω₂:
  Mean: μ₂ = {mu2}
  Variance: σ₂² = {sigma2_sq}
  Standard deviation: σ₂ = {sigma2}

Priors:
  P(ω₁) = {P_w1}
  P(ω₂) = {P_w2}

Note: Both classes have the SAME mean (μ₁ = μ₂ = 0) but DIFFERENT variances!
""")

    # ===== Step 2: Likelihood Functions =====
    print("\n" + "="*70)
    print("STEP 2: Likelihood Functions (Gaussian PDFs)")
    print("="*70)

    print(f"""
Gaussian PDF formula:
  P(x|ωᵢ) = (1/√(2πσᵢ²)) × exp(-(x-μᵢ)²/(2σᵢ²))

For our classes:

P(x|ω₁) = (1/√(2π×{sigma1_sq})) × exp(-x²/(2×{sigma1_sq}))
        = (1/√{2*np.pi*sigma1_sq:.4f}) × exp(-x²/{2*sigma1_sq})
        = {1/np.sqrt(2*np.pi*sigma1_sq):.4f} × exp(-x²/{2*sigma1_sq})

P(x|ω₂) = (1/√(2π×{sigma2_sq})) × exp(-x²/(2×{sigma2_sq}))
        = (1/√{2*np.pi*sigma2_sq:.4f}) × exp(-x²/{2*sigma2_sq})
        = {1/np.sqrt(2*np.pi*sigma2_sq):.4f} × exp(-x²/{2*sigma2_sq})
""")

    # ===== Step 3: Derive Decision Boundary =====
    print("\n" + "="*70)
    print("STEP 3: Derive Decision Boundary (where P(ω₁|x) = P(ω₂|x))")
    print("="*70)

    print("""
Decision boundary occurs where:
  P(x|ω₁)P(ω₁) = P(x|ω₂)P(ω₂)

With equal priors (P(ω₁) = P(ω₂)), this simplifies to:
  P(x|ω₁) = P(x|ω₂)

Taking the log of both sides:
  ln P(x|ω₁) = ln P(x|ω₂)

For Gaussians:
  -½ln(2πσ₁²) - (x-μ₁)²/(2σ₁²) = -½ln(2πσ₂²) - (x-μ₂)²/(2σ₂²)

Since μ₁ = μ₂ = 0:
  -½ln(2πσ₁²) - x²/(2σ₁²) = -½ln(2πσ₂²) - x²/(2σ₂²)
""")

    print(f"""
Substituting our values (σ₁² = {sigma1_sq}, σ₂² = {sigma2_sq}):

  -½ln(2π×{sigma1_sq}) - x²/{2*sigma1_sq} = -½ln(2π×{sigma2_sq}) - x²/{2*sigma2_sq}

Simplifying (the 2π cancels):
  -½ln({sigma1_sq}) - x²/{2*sigma1_sq} = -½ln({sigma2_sq}) - x²/{2*sigma2_sq}
""")

    # ===== Step 4: Solve the Quadratic Equation =====
    print("\n" + "="*70)
    print("STEP 4: Solve the Quadratic Equation")
    print("="*70)

    print("""
Rearranging (moving x² terms to left, constants to right):
  x²/(2σ₂²) - x²/(2σ₁²) = ½ln(σ₁²) - ½ln(σ₂²)

  x²/(2σ₂²) - x²/(2σ₁²) = ½ln(σ₁²/σ₂²)

Factor out x²:
  x² × (1/(2σ₂²) - 1/(2σ₁²)) = ½ln(σ₁²/σ₂²)

  x² × (σ₁² - σ₂²)/(2σ₁²σ₂²) = ½ln(σ₁²/σ₂²)
""")

    # Calculate using the CORRECT formula
    # x² × (σ₁² - σ₂²)/(2σ₁²σ₂²) = ½ln(σ₁²/σ₂²)
    # x² = ½ln(σ₁²/σ₂²) × 2σ₁²σ₂² / (σ₁² - σ₂²)
    # x² = ln(σ₁²/σ₂²) × σ₁²σ₂² / (σ₁² - σ₂²)

    log_ratio = np.log(sigma1_sq / sigma2_sq)  # ln(1/4) = -ln(4)
    numerator_coef = sigma1_sq * sigma2_sq  # 1 × 4 = 4
    denominator_coef = sigma1_sq - sigma2_sq  # 1 - 4 = -3

    x_sq = log_ratio * numerator_coef / denominator_coef

    print(f"""
Substituting σ₁² = {sigma1_sq}, σ₂² = {sigma2_sq}:

  x² × ({sigma1_sq} - {sigma2_sq})/(2 × {sigma1_sq} × {sigma2_sq}) = ½ln({sigma1_sq}/{sigma2_sq})

  x² × ({denominator_coef})/{2 * numerator_coef} = ½ × ln({sigma1_sq/sigma2_sq})

  x² × ({denominator_coef / (2 * numerator_coef):.4f}) = ½ × ({log_ratio:.4f})

  x² × ({denominator_coef / (2 * numerator_coef):.4f}) = {0.5 * log_ratio:.4f}

Solving for x²:
  x² = {0.5 * log_ratio:.4f} / {denominator_coef / (2 * numerator_coef):.4f}
     = {0.5 * log_ratio:.4f} × {(2 * numerator_coef) / denominator_coef:.4f}
     = {x_sq:.4f}
""")

    if x_sq < 0:
        print("ERROR: x² is negative. Let me recalculate...")
        print("\nActually, let's be more careful with the algebra:")

    # More careful derivation
    print("""
--- CAREFUL DERIVATION ---

Starting from:
  -½ln(σ₁²) - x²/(2σ₁²) = -½ln(σ₂²) - x²/(2σ₂²)

Move x² terms to left side, constant terms to right side:
  -x²/(2σ₁²) + x²/(2σ₂²) = -½ln(σ₂²) + ½ln(σ₁²)

  x² × (-1/(2σ₁²) + 1/(2σ₂²)) = ½(ln(σ₁²) - ln(σ₂²))

  x² × (1/(2σ₂²) - 1/(2σ₁²)) = ½ln(σ₁²/σ₂²)

  x² × (σ₁² - σ₂²)/(2σ₁²σ₂²) = ½ln(σ₁²/σ₂²)
""")

    print(f"""
With σ₁² = {sigma1_sq}, σ₂² = {sigma2_sq}:
  σ₁² - σ₂² = {sigma1_sq} - {sigma2_sq} = {sigma1_sq - sigma2_sq}
  2σ₁²σ₂² = 2 × {sigma1_sq} × {sigma2_sq} = {2 * sigma1_sq * sigma2_sq}
  ln(σ₁²/σ₂²) = ln({sigma1_sq}/{sigma2_sq}) = ln({sigma1_sq/sigma2_sq}) = {log_ratio:.4f}

So:
  x² × ({sigma1_sq - sigma2_sq}/{2 * sigma1_sq * sigma2_sq}) = ½ × {log_ratio:.4f}
  x² × ({(sigma1_sq - sigma2_sq)/(2 * sigma1_sq * sigma2_sq):.4f}) = {0.5 * log_ratio:.4f}

Since left side coefficient is negative and right side is negative:
  x² = {0.5 * log_ratio:.4f} / {(sigma1_sq - sigma2_sq)/(2 * sigma1_sq * sigma2_sq):.4f}
     = {x_sq:.4f}
""")

    x1 = np.sqrt(x_sq)
    x2 = -np.sqrt(x_sq)

    print(f"""
Taking square root:
  x = ±√{x_sq:.4f}
  x = ±{np.sqrt(x_sq):.4f}

Decision points:
  x₁ = +{x1:.4f}
  x₂ = {x2:.4f}
""")

    # ===== Step 5: Interpret the Decision Regions =====
    print("\n" + "="*70)
    print("STEP 5: Interpret the Decision Regions")
    print("="*70)

    # Check which class dominates in different regions
    test_points = [-3, 0, 3]

    print("\nTesting which class is more likely in each region:")
    print("-" * 50)

    for x in test_points:
        p1 = norm.pdf(x, mu1, sigma1)
        p2 = norm.pdf(x, mu2, sigma2)
        ratio = p1 / p2
        decision = "ω₁" if ratio > 1 else "ω₂"

        print(f"\nAt x = {x}:")
        print(f"  P(x|ω₁) = {p1:.6f}")
        print(f"  P(x|ω₂) = {p2:.6f}")
        print(f"  Likelihood ratio P(x|ω₁)/P(x|ω₂) = {ratio:.4f}")
        print(f"  Decision: {decision}")

    print(f"""

Decision Regions:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ω₂         ║       ω₁       ║         ω₂                      │
│              ║                ║                                  │
│ ←───────────x₂═══════════════x₁───────────→                     │
│           {x2:.2f}          0          {x1:.2f}                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Interpretation:
• For |x| < {x1:.4f}: Choose ω₁ (narrow distribution dominates near center)
• For |x| > {x1:.4f}: Choose ω₂ (wide distribution dominates in tails)

Why does ω₁ win near the center?
  - ω₁ has smaller variance (σ₁² = {sigma1_sq}), so it's "taller" at the center
  - Near x = 0, P(x|ω₁) > P(x|ω₂)

Why does ω₂ win in the tails?
  - ω₂ has larger variance (σ₂² = {sigma2_sq}), so its tails decay slower
  - Far from center, P(x|ω₂) > P(x|ω₁)
""")

    # ===== Step 6: Verify with specific calculations =====
    print("\n" + "="*70)
    print("STEP 6: Verification at Decision Points")
    print("="*70)

    for decision_x in [x1, x2]:
        p1 = norm.pdf(decision_x, mu1, sigma1)
        p2 = norm.pdf(decision_x, mu2, sigma2)

        print(f"\nAt x = {decision_x:.4f}:")
        print(f"  P(x|ω₁) = (1/√(2π×{sigma1_sq})) × exp(-{decision_x:.4f}²/{2*sigma1_sq})")
        print(f"          = {1/np.sqrt(2*np.pi*sigma1_sq):.4f} × exp({-decision_x**2/(2*sigma1_sq):.4f})")
        print(f"          = {p1:.6f}")
        print(f"  P(x|ω₂) = (1/√(2π×{sigma2_sq})) × exp(-{decision_x:.4f}²/{2*sigma2_sq})")
        print(f"          = {1/np.sqrt(2*np.pi*sigma2_sq):.4f} × exp({-decision_x**2/(2*sigma2_sq):.4f})")
        print(f"          = {p2:.6f}")
        print(f"  Ratio P(x|ω₁)/P(x|ω₂) = {p1/p2:.6f} ≈ 1.0 ✓")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
Problem:
  ω₁: N(μ₁={mu1}, σ₁²={sigma1_sq})
  ω₂: N(μ₂={mu2}, σ₂²={sigma2_sq})
  Equal priors: P(ω₁) = P(ω₂) = 0.5

Decision Points:
  x₁ = {x1:.4f}
  x₂ = {x2:.4f}

Decision Rule:
  Choose ω₁ if {x2:.4f} < x < {x1:.4f}
  Choose ω₂ otherwise (if x < {x2:.4f} or x > {x1:.4f})

Key Insights:
1. Same mean ≠ same class! Variance matters.
2. Unequal variances → Quadratic boundary
3. The "tighter" distribution (smaller σ²) wins near the mean
4. The "wider" distribution (larger σ²) wins in the tails
5. This creates TWO decision points (quadratic has 2 roots)
""")

    # ===== General Formula =====
    print("\n" + "="*70)
    print("GENERAL FORMULA (for exam reference)")
    print("="*70)

    print("""
For 1D Gaussian with same mean μ but different variances:
  ω₁: N(μ, σ₁²)
  ω₂: N(μ, σ₂²)

Decision boundary equation (equal priors):
  x² × (σ₁² - σ₂²)/(2σ₁²σ₂²) = ½ln(σ₁²/σ₂²)

Solving for x²:
  x² = σ₁²σ₂² × ln(σ₁²/σ₂²) / (σ₁² - σ₂²)

Note: If σ₁ < σ₂:
  - ln(σ₁²/σ₂²) < 0 (negative)
  - σ₁² - σ₂² < 0 (negative)
  - x² = (negative)/(negative) = positive ✓

Example verification with σ₁² = 1, σ₂² = 4:
  x² = 1×4 × ln(1/4) / (1-4)
     = 4 × (-1.386) / (-3)
     = -5.545 / (-3)
     = 1.848

  x = ±√1.848 = ±1.36 ✓
""")

    # ===== Visualization =====
    print("\n" + "="*70)
    print("VISUALIZATION (saved as bayesian_quadratic_plot.png)")
    print("="*70)

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: PDFs
        x_range = np.linspace(-5, 5, 500)
        pdf1 = norm.pdf(x_range, mu1, sigma1)
        pdf2 = norm.pdf(x_range, mu2, sigma2)

        ax1.plot(x_range, pdf1, 'b-', linewidth=2, label=f'ω₁: N({mu1}, {sigma1_sq})')
        ax1.plot(x_range, pdf2, 'r-', linewidth=2, label=f'ω₂: N({mu2}, {sigma2_sq})')

        # Mark decision points
        ax1.axvline(x=x1, color='g', linestyle='--', linewidth=2, label=f'Decision points: x = ±{x1:.2f}')
        ax1.axvline(x=x2, color='g', linestyle='--', linewidth=2)

        # Shade decision regions
        ax1.fill_between(x_range, 0, pdf1, where=(x_range > x2) & (x_range < x1),
                         alpha=0.3, color='blue', label='Choose ω₁')
        ax1.fill_between(x_range, 0, pdf2, where=(x_range < x2) | (x_range > x1),
                         alpha=0.3, color='red', label='Choose ω₂')

        ax1.set_xlabel('x')
        ax1.set_ylabel('P(x|ωᵢ)')
        ax1.set_title('Gaussian PDFs with Same Mean but Different Variances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Likelihood ratio
        likelihood_ratio = pdf1 / pdf2
        ax2.plot(x_range, likelihood_ratio, 'g-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', linewidth=1, label='Decision threshold = 1')
        ax2.axvline(x=x1, color='r', linestyle=':', linewidth=2)
        ax2.axvline(x=x2, color='r', linestyle=':', linewidth=2)

        ax2.set_xlabel('x')
        ax2.set_ylabel('P(x|ω₁) / P(x|ω₂)')
        ax2.set_title('Likelihood Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 3)

        plt.tight_layout()
        plt.savefig('/Users/tunchanokpunmeros/Desktop/pattern/bayesian_quadratic_plot.png',
                    dpi=150, bbox_inches='tight')
        print("Plot saved successfully!")
    except Exception as e:
        print(f"Could not save plot: {e}")

    return x1, x2


if __name__ == "__main__":
    bayesian_quadratic_boundary()
