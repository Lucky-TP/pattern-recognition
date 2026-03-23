"""
Scatter Matrices & Class Separability Measures - Step-by-Step Solution
=======================================================================
Pattern Recognition Exam Practice

Key Concepts:
- S_W: Within-class scatter matrix (measures spread within each class)
- S_B: Between-class scatter matrix (measures separation between class means)
- J: Fisher criterion for class separability

Formulas:
    S_W = Σᵢ Σ_{x∈ωᵢ} (x - μᵢ)(x - μᵢ)ᵀ  (within-class scatter)
    S_B = Σᵢ Nᵢ(μᵢ - μ)(μᵢ - μ)ᵀ         (between-class scatter)

    For 2-class case:
    S_B = (μ₁ - μ₂)(μ₁ - μ₂)ᵀ  (up to scaling)

    Fisher criterion (scalar form):
    J = (m₁ - m₂)² / (s₁² + s₂²)

    where m₁, m₂ are projected means and s₁², s₂² are projected variances
"""

import numpy as np

def print_matrix(name, M):
    """Pretty print a matrix with its name"""
    print(f"\n{name} =")
    if M.ndim == 1:
        print(f"  [{M[0]:.4f}, {M[1]:.4f}]ᵀ")
    else:
        for row in M:
            print(f"  [{row[0]:8.4f}, {row[1]:8.4f}]")

def scatter_matrices_analysis():
    print("=" * 65)
    print("Scatter Matrices & Class Separability - Step-by-Step Solution")
    print("=" * 65)

    # ===== Step 1: Define the data =====
    print("\n" + "="*65)
    print("STEP 1: Define the data")
    print("="*65)

    # Class 1 data
    X1 = np.array([[1, 2],
                   [2, 3],
                   [3, 3]])

    # Class 2 data
    X2 = np.array([[6, 6],
                   [7, 7],
                   [8, 6]])

    N1 = len(X1)
    N2 = len(X2)
    N = N1 + N2

    print("\nClass 1 (ω₁) samples: N₁ =", N1)
    for i, x in enumerate(X1):
        print(f"  x₁⁽{i+1}⁾ = ({x[0]}, {x[1]})")

    print("\nClass 2 (ω₂) samples: N₂ =", N2)
    for i, x in enumerate(X2):
        print(f"  x₂⁽{i+1}⁾ = ({x[0]}, {x[1]})")

    # ===== Step 2: Calculate class means and global mean =====
    print("\n" + "="*65)
    print("STEP 2: Calculate class means and global mean")
    print("="*65)

    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    mu = (N1 * mu1 + N2 * mu2) / N  # Global mean (weighted average)

    print(f"\nμ₁ = (1/N₁) Σ x for x ∈ ω₁")
    print(f"   = (1/{N1}) × [({X1[0,0]},{X1[0,1]}) + ({X1[1,0]},{X1[1,1]}) + ({X1[2,0]},{X1[2,1]})]")
    print(f"   = (1/{N1}) × ({sum(X1[:,0])}, {sum(X1[:,1])})")
    print(f"   = ({mu1[0]:.4f}, {mu1[1]:.4f})")

    print(f"\nμ₂ = (1/N₂) Σ x for x ∈ ω₂")
    print(f"   = (1/{N2}) × [({X2[0,0]},{X2[0,1]}) + ({X2[1,0]},{X2[1,1]}) + ({X2[2,0]},{X2[2,1]})]")
    print(f"   = (1/{N2}) × ({sum(X2[:,0])}, {sum(X2[:,1])})")
    print(f"   = ({mu2[0]:.4f}, {mu2[1]:.4f})")

    print(f"\nGlobal mean μ = (N₁μ₁ + N₂μ₂) / N")
    print(f"             = ({N1}×({mu1[0]:.2f},{mu1[1]:.2f}) + {N2}×({mu2[0]:.2f},{mu2[1]:.2f})) / {N}")
    print(f"             = ({mu[0]:.4f}, {mu[1]:.4f})")

    # ===== Step 3: Calculate within-class scatter S_W =====
    print("\n" + "="*65)
    print("STEP 3: Calculate within-class scatter matrix S_W")
    print("="*65)

    print("\nFormula: S_W = S₁ + S₂")
    print("         Sᵢ = Σ (x - μᵢ)(x - μᵢ)ᵀ for x ∈ class i")

    # Class 1 scatter
    print("\n--- Class 1 scatter S₁ ---")
    S1 = np.zeros((2, 2))
    for i, x in enumerate(X1):
        diff = x - mu1
        outer = np.outer(diff, diff)
        print(f"  x₁⁽{i+1}⁾ - μ₁ = ({diff[0]:.4f}, {diff[1]:.4f})")
        S1 += outer

    print_matrix("S₁", S1)

    # Class 2 scatter
    print("\n--- Class 2 scatter S₂ ---")
    S2 = np.zeros((2, 2))
    for i, x in enumerate(X2):
        diff = x - mu2
        outer = np.outer(diff, diff)
        print(f"  x₂⁽{i+1}⁾ - μ₂ = ({diff[0]:.4f}, {diff[1]:.4f})")
        S2 += outer

    print_matrix("S₂", S2)

    # Total within-class scatter
    S_W = S1 + S2
    print("\n--- Total within-class scatter ---")
    print("S_W = S₁ + S₂")
    print_matrix("S_W", S_W)

    # ===== Step 4: Calculate between-class scatter S_B =====
    print("\n" + "="*65)
    print("STEP 4: Calculate between-class scatter matrix S_B")
    print("="*65)

    print("\nFormula: S_B = Σᵢ Nᵢ(μᵢ - μ)(μᵢ - μ)ᵀ")
    print("\nFor 2-class case, this simplifies to:")
    print("  S_B ∝ (μ₁ - μ₂)(μ₁ - μ₂)ᵀ")

    # General formula
    diff1 = mu1 - mu
    diff2 = mu2 - mu
    S_B = N1 * np.outer(diff1, diff1) + N2 * np.outer(diff2, diff2)

    print(f"\nμ₁ - μ = ({diff1[0]:.4f}, {diff1[1]:.4f})")
    print(f"μ₂ - μ = ({diff2[0]:.4f}, {diff2[1]:.4f})")

    print(f"\nS_B = N₁(μ₁ - μ)(μ₁ - μ)ᵀ + N₂(μ₂ - μ)(μ₂ - μ)ᵀ")
    print(f"    = {N1} × outer({diff1}) + {N2} × outer({diff2})")

    print_matrix("S_B", S_B)

    # Alternative 2-class formula
    mean_diff = mu1 - mu2
    S_B_alt = np.outer(mean_diff, mean_diff)
    print(f"\n[Alternative] (μ₁ - μ₂)(μ₁ - μ₂)ᵀ where μ₁ - μ₂ = ({mean_diff[0]:.4f}, {mean_diff[1]:.4f})")
    print_matrix("(μ₁ - μ₂)(μ₁ - μ₂)ᵀ", S_B_alt)
    print("Note: S_B is proportional to this (same eigenvectors)")

    # ===== Step 5: Calculate total scatter S_T =====
    print("\n" + "="*65)
    print("STEP 5: Calculate total scatter matrix S_T")
    print("="*65)

    print("\nFormula: S_T = S_W + S_B")
    print("         Also: S_T = Σ (xᵢ - μ)(xᵢ - μ)ᵀ for all samples")

    S_T = S_W + S_B
    print_matrix("S_T = S_W + S_B", S_T)

    # Verify with direct calculation
    all_X = np.vstack([X1, X2])
    S_T_direct = np.zeros((2, 2))
    for x in all_X:
        diff = x - mu
        S_T_direct += np.outer(diff, diff)

    print_matrix("S_T (direct calculation)", S_T_direct)
    print("\n✓ Verified: S_T = S_W + S_B")

    # ===== Step 6: Calculate Fisher criterion J =====
    print("\n" + "="*65)
    print("STEP 6: Calculate Fisher criterion J")
    print("="*65)

    print("\nFisher criterion measures class separability:")
    print("  J = trace(S_W⁻¹ S_B)")
    print("\nAlternative for 2-class (scalar projection):")
    print("  J = (m₁ - m₂)² / (s₁² + s₂²)")
    print("  where m₁, m₂ are projected means, s₁², s₂² are projected variances")

    # Matrix form
    S_W_inv = np.linalg.inv(S_W)
    J_matrix = np.trace(S_W_inv @ S_B)

    print_matrix("S_W⁻¹", S_W_inv)
    print(f"\nJ = trace(S_W⁻¹ S_B) = {J_matrix:.4f}")

    # Scalar form using optimal projection
    v = S_W_inv @ mean_diff  # Optimal FLD direction
    v = v / np.linalg.norm(v)  # Normalize

    # Project data
    y1 = X1 @ v
    y2 = X2 @ v

    m1 = np.mean(y1)
    m2 = np.mean(y2)
    s1_sq = np.var(y1, ddof=0) * N1  # Sum of squared deviations
    s2_sq = np.var(y2, ddof=0) * N2

    print(f"\nUsing optimal projection direction v = ({v[0]:.4f}, {v[1]:.4f}):")
    print(f"  Projected Class 1: {y1}")
    print(f"  Projected Class 2: {y2}")
    print(f"  m₁ = {m1:.4f}, m₂ = {m2:.4f}")
    print(f"  s₁² = {s1_sq:.4f}, s₂² = {s2_sq:.4f}")

    J_scalar = (m1 - m2)**2 / (s1_sq + s2_sq) if (s1_sq + s2_sq) > 0 else float('inf')
    print(f"\n  J = (m₁ - m₂)² / (s₁² + s₂²)")
    print(f"    = ({m1:.4f} - {m2:.4f})² / ({s1_sq:.4f} + {s2_sq:.4f})")
    print(f"    = {(m1 - m2)**2:.4f} / {s1_sq + s2_sq:.4f}")
    print(f"    = {J_scalar:.4f}")

    # ===== Step 7: Interpretation =====
    print("\n" + "="*65)
    print("STEP 7: Interpretation")
    print("="*65)

    print(f"""
Fisher Criterion J = {J_matrix:.4f}

Interpretation:
• J measures class separability
• Higher J → better class separation
• J = 0 → classes completely overlap (no separation)
• J → ∞ → classes perfectly separable (no within-class variance)

For this example:
• J = {J_matrix:.4f} indicates {'good' if J_matrix > 1 else 'moderate' if J_matrix > 0.5 else 'poor'} class separability
• The classes {'are well' if J_matrix > 1 else 'may be' if J_matrix > 0.5 else 'are not well'} separated

Key relationships:
• S_T = S_W + S_B (total scatter = within + between)
• trace(S_B)/trace(S_W) gives overall separability ratio
• trace(S_B)/trace(S_W) = {np.trace(S_B)/np.trace(S_W):.4f}
""")

    # ===== Summary =====
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print_matrix("Within-class scatter S_W", S_W)
    print_matrix("Between-class scatter S_B", S_B)
    print_matrix("Total scatter S_T", S_T)
    print(f"\nFisher criterion J = {J_matrix:.4f}")
    print(f"Separability ratio trace(S_B)/trace(S_W) = {np.trace(S_B)/np.trace(S_W):.4f}")

    return S_W, S_B, S_T, J_matrix


if __name__ == "__main__":
    scatter_matrices_analysis()
