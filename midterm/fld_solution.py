"""
Fisher Linear Discriminant (FLD) - Step-by-Step Solution
=========================================================
Pattern Recognition Exam Practice

Problem:
- Class 1: (1,2), (2,1), (3,3) → μ₁ = (2, 2)
- Class 2: (6,5), (7,6), (8,5) → μ₂ = (7, 5.33)
- Find optimal projection vector v = S_W⁻¹(μ₁ - μ₂)

Key Formula:
    v = S_W⁻¹(μ₁ - μ₂)

where S_W = S₁ + S₂ (within-class scatter matrix)
      Sᵢ = Σ(x - μᵢ)(x - μᵢ)ᵀ for all x in class i
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

def fisher_linear_discriminant():
    print("=" * 60)
    print("Fisher Linear Discriminant (FLD) - Step-by-Step Solution")
    print("=" * 60)

    # ===== Step 1: Define the data =====
    print("\n" + "="*60)
    print("STEP 1: Define the data")
    print("="*60)

    # Class 1 data (not collinear - forms a triangle)
    X1 = np.array([[0, 1],
                   [2, 1],
                   [1, 0],
                   [1,2]])

    # Class 2 data (not collinear - forms a triangle)
    X2 = np.array([[4, 1],
                   [6, 1],
                   [5, 0],
                   [5,2]])

    print("\nClass 1 (ω₁) samples:")
    print(f"  x₁⁽¹⁾ = (0, 1)")
    print(f"  x₂⁽¹⁾ = (2, 1)")
    print(f"  x₃⁽¹⁾ = (1, 0)")
    print(f"  x4⁽¹⁾ = (1, 2)")

    print("\nClass 2 (ω₂) samples:")
    print(f"  x₁⁽²⁾ = (4, 1)")
    print(f"  x₂⁽²⁾ = (6, 1)")
    print(f"  x₃⁽²⁾ = (5, 0)")
    print(f"  x4⁽²⁾ = (5, 2)")

    # ===== Step 2: Calculate class means =====
    print("\n" + "="*60)
    print("STEP 2: Calculate class means")
    print("="*60)

    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    print("\nμ₁ = (1/N₁) Σ xᵢ for xᵢ ∈ Class 1")
    print(f"   = (1/4) × [(0,1) + (2,1) + (1,0), (1,2)]")
    print(f"   = (1/4) × (4, 4)")
    print(f"   = ({mu1[0]:.4f}, {mu1[1]:.4f})")

    print("\nμ₂ = (1/N₂) Σ xᵢ for xᵢ ∈ Class 2")
    print(f"   = (1/4) × [(4,1) + (6,1) + (5,0), (5,2)]")
    print(f"   = (1/4) × (20, 4)")
    print(f"   = ({mu2[0]:.4f}, {mu2[1]:.4f})")

    # ===== Step 3: Calculate within-class scatter for each class =====
    print("\n" + "="*60)
    print("STEP 3: Calculate within-class scatter matrices S₁ and S₂")
    print("="*60)

    print("\nFormula: Sᵢ = Σ (x - μᵢ)(x - μᵢ)ᵀ for all x in class i")

    # Class 1 scatter
    print("\n--- Class 1 scatter S₁ ---")
    S1 = np.zeros((2, 2))
    for i, x in enumerate(X1):
        diff = x - mu1
        outer = np.outer(diff, diff)
        print(f"\nSample x₁⁽{i+1}⁾ = {tuple(x)}:")
        print(f"  x - μ₁ = ({x[0]}, {x[1]}) - ({mu1[0]}, {mu1[1]}) = ({diff[0]}, {diff[1]})")
        print(f"  (x - μ₁)(x - μ₁)ᵀ = [{diff[0]:.1f}] × [{diff[0]:.1f}, {diff[1]:.1f}]")
        print(f"                       [{diff[1]:.1f}]")
        print(f"                     = [[{outer[0,0]:.2f}, {outer[0,1]:.2f}],")
        print(f"                        [{outer[1,0]:.2f}, {outer[1,1]:.2f}]]")
        S1 += outer

    print_matrix("S₁ (sum of outer products for Class 1)", S1)

    # Class 2 scatter
    print("\n--- Class 2 scatter S₂ ---")
    S2 = np.zeros((2, 2))
    for i, x in enumerate(X2):
        diff = x - mu2
        outer = np.outer(diff, diff)
        print(f"\nSample x₂⁽{i+1}⁾ = {tuple(x)}:")
        print(f"  x - μ₂ = ({x[0]}, {x[1]}) - ({mu2[0]}, {mu2[1]}) = ({diff[0]}, {diff[1]})")
        print(f"  (x - μ₂)(x - μ₂)ᵀ = [{diff[0]:.1f}] × [{diff[0]:.1f}, {diff[1]:.1f}]")
        print(f"                       [{diff[1]:.1f}]")
        print(f"                     = [[{outer[0,0]:.2f}, {outer[0,1]:.2f}],")
        print(f"                        [{outer[1,0]:.2f}, {outer[1,1]:.2f}]]")
        S2 += outer

    print_matrix("S₂ (sum of outer products for Class 2)", S2)

    # ===== Step 4: Calculate total within-class scatter S_W =====
    print("\n" + "="*60)
    print("STEP 4: Calculate total within-class scatter S_W = S₁ + S₂")
    print("="*60)

    S_W = S1 + S2

    print(f"\nS_W = S₁ + S₂")
    print(f"    = [[{S1[0,0]:.2f}, {S1[0,1]:.2f}]  +  [[{S2[0,0]:.2f}, {S2[0,1]:.2f}]")
    print(f"       [{S1[1,0]:.2f}, {S1[1,1]:.2f}]]     [{S2[1,0]:.2f}, {S2[1,1]:.2f}]]")

    print_matrix("S_W", S_W)

    # ===== Step 5: Calculate inverse of S_W =====
    print("\n" + "="*60)
    print("STEP 5: Calculate S_W⁻¹")
    print("="*60)

    det_SW = np.linalg.det(S_W)
    print(f"\nFor 2×2 matrix [[a, b], [c, d]]:")
    print(f"  det = ad - bc")
    print(f"  inverse = (1/det) × [[d, -b], [-c, a]]")

    print(f"\ndet(S_W) = ({S_W[0,0]})({S_W[1,1]}) - ({S_W[0,1]})({S_W[1,0]})")
    print(f"        = {S_W[0,0]*S_W[1,1]} - {S_W[0,1]*S_W[1,0]}")
    print(f"        = {det_SW}")

    S_W_inv = np.linalg.inv(S_W)

    print(f"\nS_W⁻¹ = (1/{det_SW}) × [[{S_W[1,1]:.2f}, {-S_W[0,1]:.2f}],")
    print(f"                         [{-S_W[1,0]:.2f}, {S_W[0,0]:.2f}]]")

    print_matrix("S_W⁻¹", S_W_inv)

    # ===== Step 6: Calculate mean difference =====
    print("\n" + "="*60)
    print("STEP 6: Calculate mean difference (μ₁ - μ₂)")
    print("="*60)

    mean_diff = mu1 - mu2

    print(f"\nμ₁ - μ₂ = ({mu1[0]}, {mu1[1]}) - ({mu2[0]}, {mu2[1]})")
    print(f"        = ({mean_diff[0]}, {mean_diff[1]})")

    # ===== Step 7: Calculate optimal projection vector =====
    print("\n" + "="*60)
    print("STEP 7: Calculate optimal projection vector v = S_W⁻¹(μ₁ - μ₂)")
    print("="*60)

    v = S_W_inv @ mean_diff

    print(f"\nv = S_W⁻¹ × (μ₁ - μ₂)")
    print(f"  = [[{S_W_inv[0,0]:.4f}, {S_W_inv[0,1]:.4f}]  ×  [{mean_diff[0]:.1f}]")
    print(f"     [{S_W_inv[1,0]:.4f}, {S_W_inv[1,1]:.4f}]]    [{mean_diff[1]:.1f}]")

    print(f"\nv[0] = ({S_W_inv[0,0]:.4f})×({mean_diff[0]:.1f}) + ({S_W_inv[0,1]:.4f})×({mean_diff[1]:.1f})")
    print(f"     = {S_W_inv[0,0]*mean_diff[0]:.4f} + {S_W_inv[0,1]*mean_diff[1]:.4f} = {v[0]:.4f}")

    print(f"\nv[1] = ({S_W_inv[1,0]:.4f})×({mean_diff[0]:.1f}) + ({S_W_inv[1,1]:.4f})×({mean_diff[1]:.1f})")
    print(f"     = {S_W_inv[1,0]*mean_diff[0]:.4f} + {S_W_inv[1,1]*mean_diff[1]:.4f} = {v[1]:.4f}")

    print_matrix("v (optimal projection vector)", v)

    # Normalize for interpretation
    v_norm = v / np.linalg.norm(v)
    print(f"\nNormalized: v̂ = v/||v|| = ({v_norm[0]:.4f}, {v_norm[1]:.4f})")

    # ===== Step 8: Project data and verify separation =====
    print("\n" + "="*60)
    print("STEP 8: Project data onto v and verify separation")
    print("="*60)

    print("\nProjected values y = vᵀx:")

    print("\nClass 1 projections:")
    y1 = []
    for i, x in enumerate(X1):
        y = np.dot(v, x)
        y1.append(y)
        print(f"  y₁⁽{i+1}⁾ = vᵀ × x₁⁽{i+1}⁾ = ({v[0]:.4f})×{x[0]} + ({v[1]:.4f})×{x[1]} = {y:.4f}")

    print("\nClass 2 projections:")
    y2 = []
    for i, x in enumerate(X2):
        y = np.dot(v, x)
        y2.append(y)
        print(f"  y₂⁽{i+1}⁾ = vᵀ × x₂⁽{i+1}⁾ = ({v[0]:.4f})×{x[0]} + ({v[1]:.4f})×{x[1]} = {y:.4f}")

    print(f"\nProjected means:")
    print(f"  m₁ = mean of Class 1 projections = {np.mean(y1):.4f}")
    print(f"  m₂ = mean of Class 2 projections = {np.mean(y2):.4f}")
    print(f"  |m₁ - m₂| = {abs(np.mean(y1) - np.mean(y2)):.4f}")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n• Class 1 mean: μ₁ = ({mu1[0]}, {mu1[1]})")
    print(f"• Class 2 mean: μ₂ = ({mu2[0]}, {mu2[1]})")
    print_matrix("• Within-class scatter S_W", S_W)
    print_matrix("• Optimal projection vector v", v)
    print(f"\n• Direction of v: The projection maximizes class separation")
    print(f"• Note: v can be scaled by any constant (only direction matters)")

    return v, S_W, mu1, mu2


if __name__ == "__main__":
    fisher_linear_discriminant()
