"""
EM Algorithm for Gaussian Mixture Models - Step-by-Step Solution
=================================================================
Pattern Recognition Exam Practice

Algorithm for K-component GMM:
1. Initialize: means μ_k, covariances Σ_k, mixing coefficients π_k
2. E-step: Compute responsibilities γ(z_nk) = P(z_n=k | x_n)
3. M-step: Update parameters using responsibilities
4. Repeat until convergence

Key Formulas:
    E-step: γ(z_nk) = π_k × N(x_n | μ_k, Σ_k) / Σ_j π_j × N(x_n | μ_j, Σ_j)

    M-step:
        N_k = Σ_n γ(z_nk)
        μ_k = (1/N_k) × Σ_n γ(z_nk) × x_n
        Σ_k = (1/N_k) × Σ_n γ(z_nk) × (x_n - μ_k)(x_n - μ_k)^T
        π_k = N_k / N
"""

import numpy as np


def gaussian_pdf(x, mu, sigma):
    """Calculate Gaussian probability density."""
    d = len(x)
    diff = x - mu
    exponent = -0.5 * np.dot(diff, diff) / (sigma ** 2)
    norm = 1 / (np.sqrt(2 * np.pi) ** d * sigma ** d)
    return norm * np.exp(exponent)


def em_algorithm():
    print("=" * 70)
    print("EM Algorithm for Gaussian Mixture Models - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Data")
    print("="*70)

    # 1D data for simplicity
    X = np.array([1.0, 1.5, 2.0, 5.5, 6.0, 6.5, 7.0])

    N = len(X)

    print("\n1D Data points:")
    print("-" * 30)
    for i, x in enumerate(X):
        print(f"  x_{i+1} = {x:.1f}")

    print(f"\nNumber of samples: N = {N}")
    print("Number of components: K = 2")

    # ===== Step 2: Initialize Parameters =====
    print("\n" + "="*70)
    print("STEP 2: Initialize GMM Parameters")
    print("="*70)

    K = 2  # Number of Gaussian components

    # Initial parameters
    mu = np.array([1.5, 6.0])      # Means
    sigma = np.array([1.0, 1.0])   # Standard deviations (using shared variance for simplicity)
    pi = np.array([0.5, 0.5])      # Mixing coefficients

    print("\nInitial parameters:")
    print(f"  μ₁ = {mu[0]:.2f}")
    print(f"  μ₂ = {mu[1]:.2f}")
    print(f"  σ₁ = {sigma[0]:.2f}")
    print(f"  σ₂ = {sigma[1]:.2f}")
    print(f"  π₁ = {pi[0]:.2f}")
    print(f"  π₂ = {pi[1]:.2f}")

    # ===== EM Iterations =====
    print("\n" + "="*70)
    print("EM ITERATIONS")
    print("="*70)

    max_iterations = 5
    gamma = np.zeros((N, K))  # Initialize responsibilities

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        print(f"\nCurrent parameters:")
        print(f"  μ = [{mu[0]:.4f}, {mu[1]:.4f}]")
        print(f"  σ = [{sigma[0]:.4f}, {sigma[1]:.4f}]")
        print(f"  π = [{pi[0]:.4f}, {pi[1]:.4f}]")

        # ===== E-step =====
        print(f"\n--- E-step: Compute Responsibilities ---")

        print(f"\nFormula: γ(z_nk) = π_k × N(x_n | μ_k, σ_k) / Σ_j π_j × N(x_n | μ_j, σ_j)")

        # Calculate responsibilities
        gamma = np.zeros((N, K))

        print("\nCalculating Gaussian densities and responsibilities:")
        print("-" * 80)
        print("  n      x_n     N(x|μ₁,σ₁)   N(x|μ₂,σ₂)    π₁×N₁     π₂×N₂      γ₁      γ₂")
        print("-" * 80)

        for n, x in enumerate(X):
            # Calculate Gaussian densities
            p1 = gaussian_pdf(np.array([x]), np.array([mu[0]]), sigma[0])
            p2 = gaussian_pdf(np.array([x]), np.array([mu[1]]), sigma[1])

            # Weighted by mixing coefficient
            wp1 = pi[0] * p1
            wp2 = pi[1] * p2
            total = wp1 + wp2

            # Responsibilities
            gamma[n, 0] = wp1 / total
            gamma[n, 1] = wp2 / total

            print(f"  {n+1}     {x:.1f}     {p1:.6f}    {p2:.6f}    {wp1:.6f}   {wp2:.6f}   {gamma[n,0]:.4f}  {gamma[n,1]:.4f}")

        # ===== M-step =====
        print(f"\n--- M-step: Update Parameters ---")

        # Calculate N_k (effective number of points in each cluster)
        N_k = np.sum(gamma, axis=0)
        print(f"\nEffective counts: N₁ = {N_k[0]:.4f}, N₂ = {N_k[1]:.4f}")

        # Update means
        print(f"\nUpdating means:")
        new_mu = np.zeros(K)
        for k in range(K):
            new_mu[k] = np.sum(gamma[:, k] * X) / N_k[k]
            print(f"  μ_{k+1} = (1/N_{k+1}) × Σ γ_n{k+1} × x_n")
            weighted_sum = np.sum(gamma[:, k] * X)
            print(f"      = (1/{N_k[k]:.4f}) × {weighted_sum:.4f}")
            print(f"      = {new_mu[k]:.4f}")

        # Update variances (using 1D)
        print(f"\nUpdating variances:")
        new_sigma = np.zeros(K)
        for k in range(K):
            new_sigma[k] = np.sqrt(np.sum(gamma[:, k] * (X - new_mu[k])**2) / N_k[k])
            print(f"  σ_{k+1} = sqrt((1/N_{k+1}) × Σ γ_n{k+1} × (x_n - μ_{k+1})²)")
            var_sum = np.sum(gamma[:, k] * (X - new_mu[k])**2)
            print(f"      = sqrt((1/{N_k[k]:.4f}) × {var_sum:.4f})")
            print(f"      = {new_sigma[k]:.4f}")

        # Update mixing coefficients
        print(f"\nUpdating mixing coefficients:")
        new_pi = N_k / N
        for k in range(K):
            print(f"  π_{k+1} = N_{k+1} / N = {N_k[k]:.4f} / {N} = {new_pi[k]:.4f}")

        # Check convergence
        mu_change = np.max(np.abs(new_mu - mu))
        print(f"\n--- Convergence Check ---")
        print(f"Maximum change in means: {mu_change:.6f}")

        if mu_change < 1e-4:
            print("*** CONVERGED! ***")
            mu, sigma, pi = new_mu, new_sigma, new_pi
            break

        # Update parameters
        mu, sigma, pi = new_mu, new_sigma, new_pi

    # ===== Final Results =====
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\nFinal GMM parameters:")
    print(f"  Component 1: μ₁ = {mu[0]:.4f}, σ₁ = {sigma[0]:.4f}, π₁ = {pi[0]:.4f}")
    print(f"  Component 2: μ₂ = {mu[1]:.4f}, σ₂ = {sigma[1]:.4f}, π₂ = {pi[1]:.4f}")

    print(f"\nFinal responsibilities:")
    print("-" * 50)
    print("  x_n     γ₁ (P(cluster 1))    γ₂ (P(cluster 2))    Assigned")
    print("-" * 50)
    for n, x in enumerate(X):
        assignment = 1 if gamma[n, 0] > gamma[n, 1] else 2
        print(f"  {x:.1f}        {gamma[n,0]:.4f}               {gamma[n,1]:.4f}             Cluster {assignment}")

    # Calculate log-likelihood
    print(f"\n--- Log-Likelihood ---")
    log_likelihood = 0
    for x in X:
        p = pi[0] * gaussian_pdf(np.array([x]), np.array([mu[0]]), sigma[0]) + \
            pi[1] * gaussian_pdf(np.array([x]), np.array([mu[1]]), sigma[1])
        log_likelihood += np.log(p)

    print(f"Log-likelihood: {log_likelihood:.4f}")
    print("(Higher is better; EM maximizes this)")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: EM Algorithm for GMM")
    print("="*70)

    print(f"""
Data: {N} 1D points
Components: K = 2

Final Parameters:
  Component 1: μ₁ = {mu[0]:.4f}, σ₁ = {sigma[0]:.4f}, π₁ = {pi[0]:.4f}
  Component 2: μ₂ = {mu[1]:.4f}, σ₂ = {sigma[1]:.4f}, π₂ = {pi[1]:.4f}

EM Algorithm Steps:
1. Initialize: μ_k, σ_k, π_k
2. E-step: Compute responsibilities γ(z_nk)
         = P(component k generated point n)
3. M-step: Update parameters using soft assignments
4. Repeat until convergence

Key Properties:
• Soft assignment (probabilistic clustering)
• Maximizes log-likelihood
• Can get stuck in local maxima
• Sensitive to initialization
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform E-step and M-step calculations
2. Calculate responsibilities from given parameters
3. Update means/variances using responsibilities
4. Compare EM with K-means

Important Formulas:
• Responsibility: γ_nk = π_k × N(x_n|μ_k,Σ_k) / Σ_j π_j × N(x_n|μ_j,Σ_j)
• N_k = Σ_n γ_nk (effective count for cluster k)
• New mean: μ_k = (1/N_k) × Σ_n γ_nk × x_n
• New variance: Σ_k = (1/N_k) × Σ_n γ_nk × (x_n-μ_k)(x_n-μ_k)^T
• New π_k = N_k / N

Key Differences from K-means:
• K-means: hard assignment (point belongs to ONE cluster)
• EM/GMM: soft assignment (point has probability for EACH cluster)
• EM maximizes likelihood; K-means minimizes WCSS
""")

    return mu, sigma, pi, gamma


if __name__ == "__main__":
    em_algorithm()
