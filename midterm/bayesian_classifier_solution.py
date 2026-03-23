"""
Problem 1: Bayesian Classifier

Two-class classification problem with 2D Gaussian distributions:
- μ₁ = [1, 1]^T, μ₂ = [1.5, 1.5]^T
- σ₁² = σ₂² = 0.2
- P(ω₁) = P(ω₂) = 0.5

Part (a): Design decision rule for Minimum Error Probability (Bayesian Classifier)
Part (b): Design decision rule for Minimum Average Risk with given Loss Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def main():
    print("=" * 70)
    print("Problem 1: Bayesian Classifier")
    print("=" * 70)

    # Given parameters
    mu1 = np.array([1, 1])
    mu2 = np.array([1.5, 1.5])
    sigma_sq = 0.2  # σ² = 0.2 for both classes
    P_w1 = 0.5
    P_w2 = 0.5

    # Covariance matrix (isotropic)
    Sigma = sigma_sq * np.eye(2)

    print("\nGiven Parameters:")
    print("-" * 50)
    print(f"μ₁ = [1, 1]^T")
    print(f"μ₂ = [1.5, 1.5]^T")
    print(f"σ₁² = σ₂² = {sigma_sq}")
    print(f"P(ω₁) = P(ω₂) = {P_w1}")
    print(f"\nCovariance Matrix Σ = {sigma_sq} * I₂ = [[{sigma_sq}, 0], [0, {sigma_sq}]]")

    # Part (a): Minimum Error Probability
    print("\n" + "=" * 70)
    print("Part (a): Minimum Error Probability (Bayes Decision Rule)")
    print("=" * 70)

    print("\n--- Derivation ---")
    print("\nBayes Decision Rule: Choose ω₁ if P(ω₁|x) > P(ω₂|x)")
    print("\nUsing Bayes' theorem:")
    print("  P(ωᵢ|x) = P(x|ωᵢ) P(ωᵢ) / P(x)")
    print("\nSince P(x) is the same for both classes:")
    print("  Choose ω₁ if P(x|ω₁) P(ω₁) > P(x|ω₂) P(ω₂)")

    print("\nSince P(ω₁) = P(ω₂) = 0.5 (equal priors):")
    print("  Choose ω₁ if P(x|ω₁) > P(x|ω₂)")

    print("\nTaking log (likelihood ratio test):")
    print("  Choose ω₁ if log P(x|ω₁) > log P(x|ω₂)")

    print("\nFor Gaussian with equal covariance:")
    print("  log P(x|ωᵢ) = -1/(2σ²) (x-μᵢ)^T(x-μᵢ) + const")

    print("\nThe decision boundary is where P(x|ω₁) = P(x|ω₂):")
    print("  (x-μ₁)^T(x-μ₁) = (x-μ₂)^T(x-μ₂)")

    print("\nExpanding:")
    print("  x^Tx - 2μ₁^Tx + μ₁^Tμ₁ = x^Tx - 2μ₂^Tx + μ₂^Tμ₂")
    print("  -2μ₁^Tx + μ₁^Tμ₁ = -2μ₂^Tx + μ₂^Tμ₂")
    print("  2(μ₂ - μ₁)^Tx = μ₂^Tμ₂ - μ₁^Tμ₁")

    # Calculate terms
    mu_diff = mu2 - mu1
    mu1_sq = np.dot(mu1, mu1)
    mu2_sq = np.dot(mu2, mu2)

    print(f"\nCalculation:")
    print(f"  μ₂ - μ₁ = [{mu_diff[0]}, {mu_diff[1]}]^T")
    print(f"  μ₁^Tμ₁ = 1² + 1² = {mu1_sq}")
    print(f"  μ₂^Tμ₂ = 1.5² + 1.5² = {mu2_sq}")
    print(f"  μ₂^Tμ₂ - μ₁^Tμ₁ = {mu2_sq} - {mu1_sq} = {mu2_sq - mu1_sq}")

    print("\nDecision Boundary Equation:")
    print(f"  2 * [{mu_diff[0]}, {mu_diff[1]}] · [x₁, x₂]^T = {mu2_sq - mu1_sq}")
    print(f"  2 * ({mu_diff[0]}x₁ + {mu_diff[1]}x₂) = {mu2_sq - mu1_sq}")
    print(f"  {2 * mu_diff[0]}x₁ + {2 * mu_diff[1]}x₂ = {mu2_sq - mu1_sq}")
    print(f"  x₁ + x₂ = {(mu2_sq - mu1_sq) / (2 * mu_diff[0])}")

    threshold = (mu2_sq - mu1_sq) / (2 * mu_diff[0])
    print(f"\n*** Decision Boundary: x₁ + x₂ = {threshold} ***")
    print(f"    Or equivalently: x₂ = {threshold} - x₁")

    print("\n*** Decision Rule for Minimum Error: ***")
    print(f"    Choose ω₁ if x₁ + x₂ < {threshold}")
    print(f"    Choose ω₂ if x₁ + x₂ > {threshold}")

    # Part (b): Minimum Average Risk
    print("\n" + "=" * 70)
    print("Part (b): Minimum Average Risk")
    print("=" * 70)

    # Loss matrix
    lambda_11, lambda_12 = 0, 1
    lambda_21, lambda_22 = 0.5, 0

    print("\nLoss Matrix Λ:")
    print("       Decide ω₁  Decide ω₂")
    print(f"True ω₁    {lambda_11}         {lambda_12}")
    print(f"True ω₂    {lambda_21}       {lambda_22}")

    print("\nWhere:")
    print("  λ₁₁ = loss of deciding ω₁ when true class is ω₁ = 0 (correct)")
    print("  λ₁₂ = loss of deciding ω₂ when true class is ω₁ = 1 (miss ω₁)")
    print("  λ₂₁ = loss of deciding ω₁ when true class is ω₂ = 0.5 (false alarm)")
    print("  λ₂₂ = loss of deciding ω₂ when true class is ω₂ = 0 (correct)")

    print("\n--- Derivation ---")
    print("\nFor Minimum Risk, choose ω₁ if R(ω₁|x) < R(ω₂|x)")
    print("\nConditional Risk:")
    print("  R(ω₁|x) = λ₁₁ P(ω₁|x) + λ₂₁ P(ω₂|x)")
    print("  R(ω₂|x) = λ₁₂ P(ω₁|x) + λ₂₂ P(ω₂|x)")

    print(f"\nSubstituting values:")
    print(f"  R(ω₁|x) = {lambda_11} · P(ω₁|x) + {lambda_21} · P(ω₂|x) = {lambda_21} · P(ω₂|x)")
    print(f"  R(ω₂|x) = {lambda_12} · P(ω₁|x) + {lambda_22} · P(ω₂|x) = P(ω₁|x)")

    print("\nChoose ω₁ if R(ω₁|x) < R(ω₂|x):")
    print("  0.5 · P(ω₂|x) < P(ω₁|x)")
    print("  P(ω₁|x) > 0.5 · P(ω₂|x)")
    print("  P(ω₁|x) / P(ω₂|x) > 0.5")

    print("\nUsing Bayes' theorem:")
    print("  [P(x|ω₁) P(ω₁)] / [P(x|ω₂) P(ω₂)] > 0.5")

    print("\nSince P(ω₁) = P(ω₂) = 0.5:")
    print("  P(x|ω₁) / P(x|ω₂) > 0.5")

    print("\nTaking log (likelihood ratio test):")
    print("  log P(x|ω₁) - log P(x|ω₂) > log(0.5)")
    print(f"  log P(x|ω₁) - log P(x|ω₂) > {np.log(0.5):.4f}")

    print("\nFor Gaussian:")
    print("  -1/(2σ²)(x-μ₁)^T(x-μ₁) + 1/(2σ²)(x-μ₂)^T(x-μ₂) > log(0.5)")
    print("  1/(2σ²)[(x-μ₂)^T(x-μ₂) - (x-μ₁)^T(x-μ₁)] > log(0.5)")

    print("\nSimplifying (as before):")
    print("  1/(2σ²)[2(μ₁-μ₂)^Tx + μ₂^Tμ₂ - μ₁^Tμ₁] > log(0.5)")
    print("  1/(2σ²)[-2(μ₂-μ₁)^Tx + μ₂^Tμ₂ - μ₁^Tμ₁] > log(0.5)")

    # For the boundary
    print(f"\n  With σ² = {sigma_sq}:")
    print(f"  1/(2·{sigma_sq})[-2·{mu_diff[0]}·(x₁+x₂) + {mu2_sq - mu1_sq}] > {np.log(0.5):.4f}")
    print(f"  {1 / (2 * sigma_sq)}[-{2 * mu_diff[0]}(x₁+x₂) + {mu2_sq - mu1_sq}] > {np.log(0.5):.4f}")

    coef = 1 / (2 * sigma_sq)
    print(f"  -{coef * 2 * mu_diff[0]}(x₁+x₂) + {coef * (mu2_sq - mu1_sq)} > {np.log(0.5):.4f}")
    print(f"  -{coef * 2 * mu_diff[0]}(x₁+x₂) > {np.log(0.5):.4f} - {coef * (mu2_sq - mu1_sq)}")
    print(f"  (x₁+x₂) < [{coef * (mu2_sq - mu1_sq)} - {np.log(0.5):.4f}] / {coef * 2 * mu_diff[0]}")

    threshold_risk = (coef * (mu2_sq - mu1_sq) - np.log(0.5)) / (coef * 2 * mu_diff[0])
    print(f"  (x₁+x₂) < {threshold_risk:.4f}")

    print(f"\n*** Decision Boundary for Minimum Risk: x₁ + x₂ = {threshold_risk:.4f} ***")

    print("\n*** Decision Rule for Minimum Average Risk: ***")
    print(f"    Choose ω₁ if x₁ + x₂ < {threshold_risk:.4f}")
    print(f"    Choose ω₂ if x₁ + x₂ > {threshold_risk:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPart (a) - Minimum Error Probability:")
    print(f"  Decision Boundary: x₁ + x₂ = {threshold}")
    print(f"  Decision Rule: Choose ω₁ if x₁ + x₂ < {threshold}, else choose ω₂")

    print(f"\nPart (b) - Minimum Average Risk:")
    print(f"  Decision Boundary: x₁ + x₂ = {threshold_risk:.4f}")
    print(f"  Decision Rule: Choose ω₁ if x₁ + x₂ < {threshold_risk:.4f}, else choose ω₂")

    print(f"\nNote: The threshold shifted from {threshold} to {threshold_risk:.4f}")
    print("This is because the loss matrix penalizes false alarms (λ₂₁=0.5) less than")
    print("missing ω₁ (λ₁₂=1), so the boundary moves toward ω₂ to reduce misses of ω₁.")

    # Plot visualization
    plot_decision_boundaries(mu1, mu2, sigma_sq, threshold, threshold_risk)


def plot_decision_boundaries(mu1, mu2, sigma_sq, threshold_error, threshold_risk):
    """Plot the two decision boundaries and class distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create grid for contour plot
    x = np.linspace(-0.5, 3, 100)
    y = np.linspace(-0.5, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Create distributions
    Sigma = sigma_sq * np.eye(2)
    rv1 = multivariate_normal(mu1, Sigma)
    rv2 = multivariate_normal(mu2, Sigma)

    # Plot contours
    Z1 = rv1.pdf(pos)
    Z2 = rv2.pdf(pos)
    ax.contour(X, Y, Z1, levels=5, colors='blue', alpha=0.5)
    ax.contour(X, Y, Z2, levels=5, colors='red', alpha=0.5)

    # Plot decision boundaries
    x_line = np.linspace(-0.5, 3, 100)

    # Minimum error boundary: x1 + x2 = 2.5
    y_error = threshold_error - x_line
    ax.plot(x_line, y_error, 'g-', linewidth=2, label=f'Min Error: x₁+x₂={threshold_error}')

    # Minimum risk boundary
    y_risk = threshold_risk - x_line
    ax.plot(x_line, y_risk, 'm--', linewidth=2, label=f'Min Risk: x₁+x₂={threshold_risk:.3f}')

    # Plot means
    ax.plot(mu1[0], mu1[1], 'bo', markersize=10, label='μ₁ = (1, 1)')
    ax.plot(mu2[0], mu2[1], 'ro', markersize=10, label='μ₂ = (1.5, 1.5)')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Bayesian Classifier Decision Boundaries')
    ax.legend()
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add region labels
    ax.text(0.3, 0.3, 'ω₁ region', fontsize=12, color='blue')
    ax.text(2.2, 2.2, 'ω₂ region', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig('/Users/tunchanokpunmeros/Desktop/pattern/bayesian_classifier_plot.png', dpi=150)
    print("\nPlot saved to: bayesian_classifier_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
