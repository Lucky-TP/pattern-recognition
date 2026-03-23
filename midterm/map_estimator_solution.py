"""
Problem 2: Maximum A Posteriori (MAP) Estimator for Mean

Given:
- Training data D = {x₁, x₂, ..., xₙ} from Gaussian distribution
- Known covariance Σ, unknown mean μ
- Prior: μ ~ N(m₀, Σ₀)
- Likelihood: xᵢ|μ ~ N(μ, Σ)

Find: MAP estimator for μ
"""

import numpy as np
from scipy.linalg import inv


def main():
    print("=" * 70)
    print("Problem 2: MAP Estimator for Gaussian Mean")
    print("=" * 70)

    print("\n--- Problem Setup ---")
    print("\nGiven:")
    print("  • Training data: D = {x₁, x₂, ..., xₙ}")
    print("  • Likelihood: xᵢ|μ ~ N(μ, Σ)  for i = 1, ..., N")
    print("  • Prior: μ ~ N(m₀, Σ₀)")
    print("\nFind: MAP estimator for μ")

    print("\n" + "=" * 70)
    print("DERIVATION")
    print("=" * 70)

    print("\n--- Step 1: Write the Posterior ---")
    print("\nBy Bayes' theorem:")
    print("  P(μ|D) ∝ P(D|μ) P(μ)")
    print("\nwhere:")
    print("  • P(D|μ) = ∏ᵢ P(xᵢ|μ) is the likelihood")
    print("  • P(μ) is the prior")

    print("\n--- Step 2: Log Posterior ---")
    print("\nFor MAP estimation, we maximize log P(μ|D):")
    print("  log P(μ|D) = log P(D|μ) + log P(μ) + const")

    print("\n--- Step 3: Expand the Likelihood ---")
    print("\nLikelihood of data given μ:")
    print("  P(D|μ) = ∏ᵢ₌₁ᴺ P(xᵢ|μ)")
    print("         = ∏ᵢ₌₁ᴺ (2π)^(-d/2)|Σ|^(-1/2) exp(-½(xᵢ-μ)ᵀΣ⁻¹(xᵢ-μ))")

    print("\nLog-likelihood:")
    print("  log P(D|μ) = -N·d/2·log(2π) - N/2·log|Σ| - ½ Σᵢ (xᵢ-μ)ᵀΣ⁻¹(xᵢ-μ)")

    print("\nExpanding the quadratic term:")
    print("  Σᵢ (xᵢ-μ)ᵀΣ⁻¹(xᵢ-μ) = Σᵢ [xᵢᵀΣ⁻¹xᵢ - 2xᵢᵀΣ⁻¹μ + μᵀΣ⁻¹μ]")
    print("                       = Σᵢ xᵢᵀΣ⁻¹xᵢ - 2(Σᵢ xᵢ)ᵀΣ⁻¹μ + NμᵀΣ⁻¹μ")

    print("\nLet x̄ = (1/N)Σᵢ xᵢ be the sample mean, then Σᵢ xᵢ = N·x̄:")
    print("  = Σᵢ xᵢᵀΣ⁻¹xᵢ - 2N·x̄ᵀΣ⁻¹μ + NμᵀΣ⁻¹μ")

    print("\n--- Step 4: Expand the Prior ---")
    print("\nPrior:")
    print("  P(μ) = (2π)^(-d/2)|Σ₀|^(-1/2) exp(-½(μ-m₀)ᵀΣ₀⁻¹(μ-m₀))")

    print("\nLog-prior:")
    print("  log P(μ) = -d/2·log(2π) - ½·log|Σ₀| - ½(μ-m₀)ᵀΣ₀⁻¹(μ-m₀)")

    print("\nExpanding:")
    print("  (μ-m₀)ᵀΣ₀⁻¹(μ-m₀) = μᵀΣ₀⁻¹μ - 2m₀ᵀΣ₀⁻¹μ + m₀ᵀΣ₀⁻¹m₀")

    print("\n--- Step 5: Combine and Take Derivative ---")
    print("\nLog-posterior (keeping only μ-dependent terms):")
    print("  log P(μ|D) ∝ -½[NμᵀΣ⁻¹μ - 2N·x̄ᵀΣ⁻¹μ + μᵀΣ₀⁻¹μ - 2m₀ᵀΣ₀⁻¹μ]")
    print("             = -½[μᵀ(NΣ⁻¹ + Σ₀⁻¹)μ - 2(NΣ⁻¹x̄ + Σ₀⁻¹m₀)ᵀμ]")

    print("\nThis is a quadratic form in μ. Taking derivative and setting to zero:")
    print("  ∂/∂μ log P(μ|D) = -(NΣ⁻¹ + Σ₀⁻¹)μ + (NΣ⁻¹x̄ + Σ₀⁻¹m₀) = 0")

    print("\nSolving for μ:")
    print("  (NΣ⁻¹ + Σ₀⁻¹)μ = NΣ⁻¹x̄ + Σ₀⁻¹m₀")

    print("\n" + "=" * 70)
    print("FINAL RESULT: MAP ESTIMATOR")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                                                                 │")
    print("│   μ_MAP = (NΣ⁻¹ + Σ₀⁻¹)⁻¹ (NΣ⁻¹x̄ + Σ₀⁻¹m₀)                    │")
    print("│                                                                 │")
    print("│   where x̄ = (1/N) Σᵢ₌₁ᴺ xᵢ  (sample mean)                      │")
    print("│                                                                 │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n--- Alternative Form ---")
    print("\nDefine:")
    print("  • Σ_post = (NΣ⁻¹ + Σ₀⁻¹)⁻¹  (posterior covariance)")
    print("\nThen:")
    print("  μ_MAP = Σ_post (NΣ⁻¹x̄ + Σ₀⁻¹m₀)")
    print("       = Σ_post NΣ⁻¹x̄ + Σ_post Σ₀⁻¹m₀")

    print("\n--- Interpretation ---")
    print("\nThe MAP estimate is a weighted combination of:")
    print("  1. The sample mean x̄ (from data)")
    print("  2. The prior mean m₀")

    print("\nWeights depend on:")
    print("  • N: number of samples (more data → more weight on x̄)")
    print("  • Σ: data covariance (smaller → more confident in data)")
    print("  • Σ₀: prior covariance (smaller → more confident in prior)")

    print("\n--- Special Cases ---")

    print("\n1. As N → ∞ (lots of data):")
    print("   μ_MAP → x̄ (approaches MLE)")

    print("\n2. As N → 0 (no data):")
    print("   μ_MAP → m₀ (falls back to prior)")

    print("\n3. For scalar case (1D) with σ² and σ₀²:")
    print("   μ_MAP = (N/σ² + 1/σ₀²)⁻¹ (N·x̄/σ² + m₀/σ₀²)")
    print("         = (Nσ₀² · x̄ + σ² · m₀) / (Nσ₀² + σ²)")

    # Numerical example
    print("\n" + "=" * 70)
    print("NUMERICAL EXAMPLE")
    print("=" * 70)

    # Example parameters
    d = 2  # dimension
    N = 5  # number of samples
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    Sigma_0 = np.array([[2, 0], [0, 2]])
    m_0 = np.array([0, 0])

    # Generate some example data
    np.random.seed(42)
    true_mu = np.array([3, 2])
    data = np.random.multivariate_normal(true_mu, Sigma, N)

    print(f"\nExample with d={d} dimensions, N={N} samples:")
    print(f"\nData covariance Σ:")
    print(Sigma)
    print(f"\nPrior mean m₀ = {m_0}")
    print(f"Prior covariance Σ₀:")
    print(Sigma_0)

    print(f"\nSample data:")
    for i, x in enumerate(data):
        print(f"  x{i + 1} = [{x[0]:.4f}, {x[1]:.4f}]")

    x_bar = np.mean(data, axis=0)
    print(f"\nSample mean x̄ = [{x_bar[0]:.4f}, {x_bar[1]:.4f}]")

    # Calculate MAP estimate
    Sigma_inv = inv(Sigma)
    Sigma_0_inv = inv(Sigma_0)

    Sigma_post_inv = N * Sigma_inv + Sigma_0_inv
    Sigma_post = inv(Sigma_post_inv)

    mu_MAP = Sigma_post @ (N * Sigma_inv @ x_bar + Sigma_0_inv @ m_0)

    print(f"\nMAP Estimate:")
    print(f"  μ_MAP = [{mu_MAP[0]:.4f}, {mu_MAP[1]:.4f}]")

    # Compare with MLE
    mu_MLE = x_bar
    print(f"\nFor comparison:")
    print(f"  MLE (just sample mean) = [{mu_MLE[0]:.4f}, {mu_MLE[1]:.4f}]")
    print(f"  Prior mean m₀ = {m_0}")
    print(f"  True μ (for this simulation) = {true_mu}")

    print("\nNote: MAP estimate lies between MLE and prior mean, ")
    print("pulled toward the prior mean by the prior distribution.")


if __name__ == "__main__":
    main()
