"""
Maximum Likelihood (ML) vs Maximum A Posteriori (MAP) Estimation
================================================================
Pattern Recognition Exam Practice

Problem:
- Data D = {4, 6, 8} from Gaussian with known σ² = 4
- Prior: μ ~ N(m₀=10, σ₀²=4)
- Compare μ_ML vs μ_MAP

Key Formulas:
    μ_ML = (1/N) Σ xᵢ = sample mean

    μ_MAP = (N·σ₀²·x̄ + σ²·m₀) / (N·σ₀² + σ²)
          = weighted average of sample mean and prior mean
"""

import numpy as np

def ml_vs_map_comparison():
    print("=" * 70)
    print("ML vs MAP Estimation - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Problem Setup =====
    print("\n" + "="*70)
    print("STEP 1: Problem Setup")
    print("="*70)

    # Data
    D = np.array([4, 6, 8])
    N = len(D)

    # Known likelihood variance
    sigma_sq = 4  # Variance of the data distribution

    # Prior parameters
    m0 = 10       # Prior mean
    sigma0_sq = 4  # Prior variance

    print(f"""
Given:
  Data: D = {{{D[0]}, {D[1]}, {D[2]}}}
  Number of samples: N = {N}

Likelihood (data generating distribution):
  x ~ N(μ, σ²) where σ² = {sigma_sq} (known)
  Unknown parameter: μ

Prior belief about μ:
  μ ~ N(m₀, σ₀²)
  m₀ = {m0} (prior mean - our initial guess for μ)
  σ₀² = {sigma0_sq} (prior variance - uncertainty in our guess)
""")

    # ===== Step 2: Maximum Likelihood Estimation =====
    print("\n" + "="*70)
    print("STEP 2: Maximum Likelihood (ML) Estimation")
    print("="*70)

    print("""
ML Philosophy:
  "Find the parameter value that makes the observed data most likely"
  Does NOT use any prior information about μ

Derivation:
  Likelihood: L(μ) = P(D|μ) = ∏ᵢ P(xᵢ|μ)

  For Gaussian: P(xᵢ|μ) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))

  Log-likelihood: ln L(μ) = -N/2 × ln(2πσ²) - (1/2σ²) Σ(xᵢ-μ)²

  Taking derivative and setting to zero:
    d/dμ [ln L(μ)] = (1/σ²) Σ(xᵢ - μ) = 0

  Solving:
    Σxᵢ - Nμ = 0
    μ_ML = (1/N) Σxᵢ = sample mean
""")

    x_bar = np.mean(D)
    mu_ML = x_bar

    print(f"""
Calculation:
  μ_ML = (1/N) Σ xᵢ
       = (1/{N}) × ({D[0]} + {D[1]} + {D[2]})
       = (1/{N}) × {sum(D)}
       = {mu_ML}

  ∴ μ_ML = {mu_ML}
""")

    # ===== Step 3: Maximum A Posteriori (MAP) Estimation =====
    print("\n" + "="*70)
    print("STEP 3: Maximum A Posteriori (MAP) Estimation")
    print("="*70)

    print("""
MAP Philosophy:
  "Find the parameter value that is most probable given data AND prior"
  Uses Bayes' theorem: P(μ|D) ∝ P(D|μ) × P(μ)

Derivation:
  Posterior ∝ Likelihood × Prior

  ln P(μ|D) = ln P(D|μ) + ln P(μ) + const

  For Gaussian likelihood and Gaussian prior:
    ln P(D|μ) = -(1/2σ²) Σ(xᵢ-μ)²  + const
    ln P(μ)   = -(1/2σ₀²)(μ-m₀)²   + const

  Taking derivative and setting to zero:
    d/dμ [ln P(μ|D)] = (1/σ²) Σ(xᵢ-μ) - (1/σ₀²)(μ-m₀) = 0
""")

    print(f"""
Solving for μ_MAP:
    (1/σ²) Σ(xᵢ-μ) = (1/σ₀²)(μ-m₀)

    (1/σ²)[Σxᵢ - Nμ] = (1/σ₀²)[μ - m₀]

    (Σxᵢ)/σ² - Nμ/σ² = μ/σ₀² - m₀/σ₀²

    (Σxᵢ)/σ² + m₀/σ₀² = μ/σ₀² + Nμ/σ²

    (Σxᵢ)/σ² + m₀/σ₀² = μ × (1/σ₀² + N/σ²)

    μ_MAP = [(Σxᵢ)/σ² + m₀/σ₀²] / [1/σ₀² + N/σ²]

Multiplying numerator and denominator by σ²σ₀²:

    μ_MAP = [σ₀²Σxᵢ + σ²m₀] / [σ² + Nσ₀²]

Or equivalently (using x̄ = Σxᵢ/N):

    μ_MAP = [Nσ₀²x̄ + σ²m₀] / [Nσ₀² + σ²]
""")

    # Calculate μ_MAP
    numerator = N * sigma0_sq * x_bar + sigma_sq * m0
    denominator = N * sigma0_sq + sigma_sq
    mu_MAP = numerator / denominator

    print(f"""
Substituting our values:
  N = {N}
  σ² = {sigma_sq}
  σ₀² = {sigma0_sq}
  x̄ = {x_bar}
  m₀ = {m0}

  μ_MAP = [N×σ₀²×x̄ + σ²×m₀] / [N×σ₀² + σ²]
        = [{N}×{sigma0_sq}×{x_bar} + {sigma_sq}×{m0}] / [{N}×{sigma0_sq} + {sigma_sq}]
        = [{N * sigma0_sq * x_bar} + {sigma_sq * m0}] / [{N * sigma0_sq} + {sigma_sq}]
        = {numerator} / {denominator}
        = {mu_MAP}

  ∴ μ_MAP = {mu_MAP}
""")

    # ===== Step 4: Compare ML and MAP =====
    print("\n" + "="*70)
    print("STEP 4: Comparison of ML and MAP Estimates")
    print("="*70)

    print(f"""
Results:
┌──────────────────────────────────────────────────────────┐
│  Estimator    │    Value    │    Formula                 │
├──────────────────────────────────────────────────────────┤
│  μ_ML         │    {mu_ML:.4f}    │    Sample mean            │
│  μ_MAP        │    {mu_MAP:.4f}    │    Weighted average       │
│  Prior mean   │   {m0:.4f}    │    m₀ (initial guess)     │
└──────────────────────────────────────────────────────────┘

Key observations:
1. μ_ML = {mu_ML} (purely based on data)
2. μ_MAP = {mu_MAP} (pulled toward prior mean m₀ = {m0})
3. Prior mean m₀ = {m0}

The MAP estimate is a WEIGHTED AVERAGE:
  μ_MAP = w₁ × x̄ + w₂ × m₀

where:
  w₁ = Nσ₀² / (Nσ₀² + σ²) = {N*sigma0_sq} / {denominator} = {N*sigma0_sq/denominator:.4f}
  w₂ = σ² / (Nσ₀² + σ²) = {sigma_sq} / {denominator} = {sigma_sq/denominator:.4f}

Check: w₁ + w₂ = {N*sigma0_sq/denominator:.4f} + {sigma_sq/denominator:.4f} = {(N*sigma0_sq + sigma_sq)/denominator:.4f} ✓

Verification:
  μ_MAP = {N*sigma0_sq/denominator:.4f} × {x_bar} + {sigma_sq/denominator:.4f} × {m0}
        = {(N*sigma0_sq/denominator) * x_bar:.4f} + {(sigma_sq/denominator) * m0:.4f}
        = {mu_MAP:.4f} ✓
""")

    # ===== Step 5: Interpretation =====
    print("\n" + "="*70)
    print("STEP 5: Interpretation and Insights")
    print("="*70)

    print(f"""
Why is μ_MAP pulled toward the prior?

1. Data says: μ ≈ {x_bar} (sample mean)
2. Prior says: μ ≈ {m0} (prior belief)
3. MAP combines both: μ_MAP = {mu_MAP} (between data and prior)

The "pull" toward prior depends on:
  • Number of samples (N): More data → trust data more
  • Data variance (σ²): Lower variance → data more precise → trust data more
  • Prior variance (σ₀²): Lower variance → prior more certain → trust prior more

Effect of changing parameters:
""")

    # Show effect of different N values
    print("\nEffect of sample size N (keeping same sample mean x̄ = 6):")
    print("-" * 60)
    for N_test in [1, 3, 10, 100, 1000]:
        mu_MAP_test = (N_test * sigma0_sq * x_bar + sigma_sq * m0) / (N_test * sigma0_sq + sigma_sq)
        data_weight = N_test * sigma0_sq / (N_test * sigma0_sq + sigma_sq)
        print(f"  N = {N_test:4d}: μ_MAP = {mu_MAP_test:.4f}, data weight = {data_weight:.4f}")

    print(f"\n  As N → ∞: μ_MAP → x̄ = {x_bar} (MAP approaches ML)")

    # Show effect of different prior variances
    print(f"\nEffect of prior uncertainty σ₀² (N = {N}, σ² = {sigma_sq}):")
    print("-" * 60)
    for sigma0_sq_test in [0.1, 1, 4, 16, 100]:
        mu_MAP_test = (N * sigma0_sq_test * x_bar + sigma_sq * m0) / (N * sigma0_sq_test + sigma_sq)
        prior_weight = sigma_sq / (N * sigma0_sq_test + sigma_sq)
        print(f"  σ₀² = {sigma0_sq_test:5.1f}: μ_MAP = {mu_MAP_test:.4f}, prior weight = {prior_weight:.4f}")

    print(f"\n  As σ₀² → ∞: μ_MAP → x̄ = {x_bar} (uncertain prior → trust data)")
    print(f"  As σ₀² → 0: μ_MAP → m₀ = {m0} (certain prior → trust prior)")

    # ===== Step 6: When to use ML vs MAP =====
    print("\n" + "="*70)
    print("STEP 6: When to Use ML vs MAP")
    print("="*70)

    print("""
Use ML when:
  • No prior information is available
  • Large amount of data (N is large)
  • Want unbiased estimate
  • Prior might be wrong or misleading

Use MAP when:
  • Have reliable prior information
  • Limited data (small N)
  • Want to regularize against overfitting
  • Prior domain knowledge is valuable

Special cases:
  • If prior variance σ₀² → ∞: MAP → ML (uninformative prior)
  • If N → ∞: MAP → ML (data dominates)
  • If σ₀² → 0: MAP → m₀ (dogmatic prior)
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
Problem:
  Data: D = {{{D[0]}, {D[1]}, {D[2]}}} (N = {N})
  Likelihood variance: σ² = {sigma_sq}
  Prior: μ ~ N(m₀ = {m0}, σ₀² = {sigma0_sq})

Results:
  μ_ML  = x̄ = {mu_ML}
  μ_MAP = {mu_MAP}

Key formulas:
  μ_ML = (1/N) Σxᵢ = x̄

  μ_MAP = (Nσ₀²x̄ + σ²m₀) / (Nσ₀² + σ²)
        = w₁ × x̄ + w₂ × m₀
  where w₁ = Nσ₀²/(Nσ₀² + σ²), w₂ = σ²/(Nσ₀² + σ²)

Interpretation:
  • μ_ML ignores prior, based purely on data
  • μ_MAP is a weighted average of data and prior
  • μ_MAP is "pulled" toward prior mean m₀
  • Amount of pull depends on relative precisions (1/variance)
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common exam variations:
1. Different data values → Just calculate new x̄
2. Different N → Affects how much MAP is pulled toward prior
3. Different prior parameters → Changes the "pull" effect
4. Asked to derive MAP formula → Use Bayes' rule + take derivative

Common mistakes to avoid:
1. Forgetting that σ² is the DATA variance, not the PRIOR variance
2. Confusing m₀ (prior mean of μ) with μ itself
3. Using wrong formula weights
4. Not showing intermediate steps

Quick sanity checks:
✓ μ_MAP should be between x̄ and m₀
✓ With large N, μ_MAP ≈ x̄
✓ With small σ₀², μ_MAP ≈ m₀
""")

    return mu_ML, mu_MAP


if __name__ == "__main__":
    ml_vs_map_comparison()
