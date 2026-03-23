"""
Matrix Calculations for Pattern Recognition - Step-by-Step Solutions
=====================================================================
Essential matrix operations for exams with detailed hand calculations

Topics covered:
1. Matrix multiplication
2. Transpose
3. Determinant (2x2 and 3x3)
4. Inverse (2x2 and 3x3)
5. Eigenvalues and Eigenvectors
6. Outer product
7. Quadratic form (xᵀAx)
"""

import numpy as np

def print_matrix(name, M, decimals=4):
    """Pretty print a matrix"""
    print(f"\n{name} =")
    if isinstance(M, (int, float, np.floating)):
        print(f"  {M:.{decimals}f}")
    elif M.ndim == 1:
        print(f"  [{', '.join([f'{x:.{decimals}f}' for x in M])}]ᵀ")
    else:
        for row in M:
            print(f"  [{', '.join([f'{x:8.{decimals}f}' for x in row])}]")

def section_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 1. MATRIX MULTIPLICATION
# =============================================================================
def matrix_multiplication():
    section_header("1. MATRIX MULTIPLICATION")

    print("""
Formula: (AB)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

Rule: A(m×n) × B(n×p) = C(m×p)
      Number of columns in A must equal number of rows in B
""")

    A = np.array([[1, 0]])
    B = np.array([[0, 2, 1 ,1],
                  [1, 1, 0, 2]])

    print_matrix("A", A)
    print_matrix("B", B)

    print("\nStep-by-step calculation of C = A × B:")
    print("-" * 50)

    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            terms = []
            for k in range(2):
                terms.append(f"({A[i,k]}×{B[k,j]})")
            C[i,j] = A[i,:] @ B[:,j]
            print(f"  C[{i+1},{j+1}] = {' + '.join(terms)} = {C[i,j]:.0f}")

    print_matrix("C = AB", C, 0)

    # Matrix-vector multiplication
    print("\n--- Matrix-Vector Multiplication ---")
    x = np.array([2, 3])
    print_matrix("x", x)

    print("\nCalculation of Ax:")
    result = A @ x
    for i in range(2):
        terms = [f"({A[i,k]}×{x[k]})" for k in range(2)]
        print(f"  (Ax)[{i+1}] = {' + '.join(terms)} = {result[i]:.0f}")

    print_matrix("Ax", result, 0)


# =============================================================================
# 2. TRANSPOSE
# =============================================================================
def matrix_transpose():
    section_header("2. MATRIX TRANSPOSE")

    print("""
Formula: (Aᵀ)ᵢⱼ = Aⱼᵢ

Properties:
  • (Aᵀ)ᵀ = A
  • (A + B)ᵀ = Aᵀ + Bᵀ
  • (AB)ᵀ = BᵀAᵀ  ← Note the order reversal!
  • (cA)ᵀ = cAᵀ
""")

    A = np.array([[-0.79], [0.89]])

    print_matrix("A (2×3)", A)
    print_matrix("Aᵀ (3×2)", A.T)

    print("\nVisualization:")
    print("  A:          Aᵀ:")
    print("  [1 2 3]     [1 4]")
    print("  [4 5 6]     [2 5]")
    print("              [3 6]")

    # Vector transpose
    print("\n--- Vector Transpose ---")
    v = np.array([1, 2, 3])
    print(f"\nColumn vector v = [1, 2, 3]ᵀ")
    print(f"Row vector vᵀ = [1, 2, 3]")
    print(f"\nvᵀv (dot product) = 1² + 2² + 3² = {np.dot(v, v)}")


# =============================================================================
# 3. DETERMINANT
# =============================================================================
def matrix_determinant():
    section_header("3. DETERMINANT")

    # 2x2 Determinant
    print("\n--- 2×2 Determinant ---")
    print("""
Formula: det([[a, b], [c, d]]) = ad - bc
""")

    A = np.array([[3, 2],
                  [1, 4]])

    print_matrix("A", A)

    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    print(f"\ndet(A) = ({A[0,0]})({A[1,1]}) - ({A[0,1]})({A[1,0]})")
    print(f"       = {A[0,0]*A[1,1]} - {A[0,1]*A[1,0]}")
    print(f"       = {det_A}")

    # 3x3 Determinant
    print("\n--- 3×3 Determinant (Cofactor Expansion) ---")
    print("""
Formula (expansion along first row):
  det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)
""")

    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]])

    print_matrix("B", B)

    # Cofactor expansion along first row
    c11 = B[1,1]*B[2,2] - B[1,2]*B[2,1]
    c12 = B[1,0]*B[2,2] - B[1,2]*B[2,0]
    c13 = B[1,0]*B[2,1] - B[1,1]*B[2,0]

    print(f"\nCofactor C₁₁ = ({B[1,1]})({B[2,2]}) - ({B[1,2]})({B[2,1]}) = {c11}")
    print(f"Cofactor C₁₂ = ({B[1,0]})({B[2,2]}) - ({B[1,2]})({B[2,0]}) = {c12}")
    print(f"Cofactor C₁₃ = ({B[1,0]})({B[2,1]}) - ({B[1,1]})({B[2,0]}) = {c13}")

    det_B = B[0,0]*c11 - B[0,1]*c12 + B[0,2]*c13
    print(f"\ndet(B) = ({B[0,0]})({c11}) - ({B[0,1]})({c12}) + ({B[0,2]})({c13})")
    print(f"       = {B[0,0]*c11} - {B[0,1]*c12} + {B[0,2]*c13}")
    print(f"       = {det_B}")

    print(f"\nVerification with numpy: det(B) = {np.linalg.det(B):.4f}")


# =============================================================================
# 4. MATRIX INVERSE
# =============================================================================
def matrix_inverse():
    section_header("4. MATRIX INVERSE")

    # 2x2 Inverse
    print("\n--- 2×2 Inverse ---")
    print("""
Formula: A⁻¹ = (1/det(A)) × [[d, -b], [-c, a]]

For A = [[a, b], [c, d]]:
  A⁻¹ = (1/(ad-bc)) × [[d, -b], [-c, a]]
""")

    A = np.array([[4, 3],
                  [2, 1]])

    print_matrix("A", A)

    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    print(f"\nStep 1: Calculate det(A) = ({A[0,0]})({A[1,1]}) - ({A[0,1]})({A[1,0]}) = {det_A}")

    adj_A = np.array([[A[1,1], -A[0,1]],
                      [-A[1,0], A[0,0]]])
    print(f"\nStep 2: Form adjugate matrix:")
    print(f"  adj(A) = [[{A[1,1]}, {-A[0,1]}], [{-A[1,0]}, {A[0,0]}]]")
    print_matrix("adj(A)", adj_A)

    A_inv = adj_A / det_A
    print(f"\nStep 3: A⁻¹ = (1/{det_A}) × adj(A)")
    print_matrix("A⁻¹", A_inv)

    print("\nVerification: A × A⁻¹ should equal I")
    print_matrix("A × A⁻¹", A @ A_inv)

    # Special case: Diagonal matrix
    print("\n--- Special Case: Diagonal Matrix ---")
    D = np.array([[2, 0],
                  [0, 5]])
    print_matrix("D", D)
    print(f"\nFor diagonal matrix: D⁻¹[i,i] = 1/D[i,i]")
    D_inv = np.array([[1/2, 0],
                      [0, 1/5]])
    print_matrix("D⁻¹", D_inv)

    # Special case: σ²I
    print("\n--- Special Case: σ²I (Scaled Identity) ---")
    sigma_sq = 4
    print(f"\nIf Σ = σ²I = {sigma_sq}I")
    print(f"Then Σ⁻¹ = (1/σ²)I = (1/{sigma_sq})I = {1/sigma_sq}I")


# =============================================================================
# 5. EIGENVALUES AND EIGENVECTORS
# =============================================================================
def eigenvalues_eigenvectors():
    section_header("5. EIGENVALUES AND EIGENVECTORS")

    print("""
Definition: Av = λv
  • λ is an eigenvalue
  • v is the corresponding eigenvector

To find eigenvalues: solve det(A - λI) = 0
To find eigenvectors: solve (A - λI)v = 0 for each λ
""")

    A = np.array([[4, 2],
                  [1, 3]])

    print_matrix("A", A)

    print("\n--- Step 1: Find Eigenvalues ---")
    print("\nForm characteristic equation: det(A - λI) = 0")
    print(f"\nA - λI = [[{A[0,0]}-λ, {A[0,1]}], [{A[1,0]}, {A[1,1]}-λ]]")

    print(f"\ndet(A - λI) = ({A[0,0]}-λ)({A[1,1]}-λ) - ({A[0,1]})({A[1,0]})")
    print(f"            = λ² - ({A[0,0]}+{A[1,1]})λ + ({A[0,0]}×{A[1,1]} - {A[0,1]}×{A[1,0]})")

    trace_A = A[0,0] + A[1,1]
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    print(f"            = λ² - {trace_A}λ + {det_A}")

    print(f"\nUsing quadratic formula: λ = ({trace_A} ± √({trace_A}² - 4×{det_A})) / 2")
    discriminant = trace_A**2 - 4*det_A
    print(f"                        = ({trace_A} ± √{discriminant}) / 2")
    print(f"                        = ({trace_A} ± {np.sqrt(discriminant):.4f}) / 2")

    lambda1 = (trace_A + np.sqrt(discriminant)) / 2
    lambda2 = (trace_A - np.sqrt(discriminant)) / 2
    print(f"\nλ₁ = {lambda1:.4f}")
    print(f"λ₂ = {lambda2:.4f}")

    print("\n--- Step 2: Find Eigenvectors ---")

    for i, lam in enumerate([lambda1, lambda2], 1):
        print(f"\nFor λ{i} = {lam:.4f}:")
        A_minus_lambda_I = A - lam * np.eye(2)
        print(f"  (A - λ{i}I) = [[{A_minus_lambda_I[0,0]:.4f}, {A_minus_lambda_I[0,1]:.4f}],")
        print(f"                [{A_minus_lambda_I[1,0]:.4f}, {A_minus_lambda_I[1,1]:.4f}]]")

        print(f"\n  Solve (A - λ{i}I)v = 0:")
        print(f"  {A_minus_lambda_I[0,0]:.4f}v₁ + {A_minus_lambda_I[0,1]:.4f}v₂ = 0")

        if abs(A_minus_lambda_I[0,1]) > 1e-10:
            ratio = -A_minus_lambda_I[0,0] / A_minus_lambda_I[0,1]
            print(f"  v₂ = {ratio:.4f}v₁")
            print(f"  Let v₁ = 1, then v₂ = {ratio:.4f}")
            v = np.array([1, ratio])
        else:
            v = np.array([0, 1])

        v_normalized = v / np.linalg.norm(v)
        print(f"  v{i} = [{v[0]:.4f}, {v[1]:.4f}]ᵀ")
        print(f"  Normalized: v̂{i} = [{v_normalized[0]:.4f}, {v_normalized[1]:.4f}]ᵀ")

    # Verify with numpy
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\n--- Verification with numpy ---")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):")
    print_matrix("V", eigenvectors)


# =============================================================================
# 6. OUTER PRODUCT
# =============================================================================
def outer_product():
    section_header("6. OUTER PRODUCT")

    print("""
Definition: For vectors u and v, the outer product uvᵀ is a matrix

Formula: (uvᵀ)ᵢⱼ = uᵢ × vⱼ

Key usage in Pattern Recognition:
  • Scatter matrix: S = Σ(xᵢ - μ)(xᵢ - μ)ᵀ
  • Covariance matrix calculation
""")

    u = np.array([1, 2, 3])
    v = np.array([4, 5])

    print_matrix("u (3×1)", u)
    print_matrix("v (2×1)", v)

    print("\nOuter product uvᵀ (3×2 matrix):")
    print("-" * 40)

    outer = np.outer(u, v)
    for i in range(len(u)):
        row_calc = [f"({u[i]}×{v[j]})" for j in range(len(v))]
        row_vals = [f"{u[i]*v[j]}" for j in range(len(v))]
        print(f"  Row {i+1}: {row_calc} = [{', '.join(row_vals)}]")

    print_matrix("uvᵀ", outer, 0)

    # Scatter matrix example
    print("\n--- Application: Scatter Matrix ---")
    x = np.array([3, 4])
    mu = np.array([2, 2])
    diff = x - mu

    print_matrix("x", x)
    print_matrix("μ", mu)
    print_matrix("x - μ", diff)

    scatter_contrib = np.outer(diff, diff)
    print("\n(x - μ)(x - μ)ᵀ:")
    print(f"  = [{diff[0]}] × [{diff[0]}, {diff[1]}]")
    print(f"    [{diff[1]}]")
    print_matrix("(x - μ)(x - μ)ᵀ", scatter_contrib, 0)


# =============================================================================
# 7. QUADRATIC FORM
# =============================================================================
def quadratic_form():
    section_header("7. QUADRATIC FORM (xᵀAx)")

    print("""
Definition: xᵀAx is a scalar (single number)

Formula for 2D: If x = [x₁, x₂]ᵀ and A = [[a, b], [c, d]]
  xᵀAx = ax₁² + (b+c)x₁x₂ + dx₂²

Key usage in Pattern Recognition:
  • Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
  • Discriminant functions
""")

    x = np.array([2, 3])
    A = np.array([[4, 1],
                  [1, 2]])

    print_matrix("x", x)
    print_matrix("A", A)

    print("\n--- Method 1: Step by step ---")

    # Step 1: Ax
    Ax = A @ x
    print(f"\nStep 1: Calculate Ax")
    print(f"  Ax = [[{A[0,0]}, {A[0,1]}]  × [{x[0]}]")
    print(f"       [{A[1,0]}, {A[1,1]}]]   [{x[1]}]")
    print(f"\n  (Ax)[1] = {A[0,0]}×{x[0]} + {A[0,1]}×{x[1]} = {Ax[0]}")
    print(f"  (Ax)[2] = {A[1,0]}×{x[0]} + {A[1,1]}×{x[1]} = {Ax[1]}")
    print_matrix("Ax", Ax, 0)

    # Step 2: xᵀ(Ax)
    result = x @ Ax
    print(f"\nStep 2: Calculate xᵀ(Ax)")
    print(f"  xᵀ(Ax) = [{x[0]}, {x[1]}] × [{Ax[0]}]")
    print(f"                         [{Ax[1]}]")
    print(f"         = {x[0]}×{Ax[0]} + {x[1]}×{Ax[1]}")
    print(f"         = {x[0]*Ax[0]} + {x[1]*Ax[1]}")
    print(f"         = {result}")

    print("\n--- Method 2: Direct expansion ---")
    print(f"\nxᵀAx = {A[0,0]}x₁² + ({A[0,1]}+{A[1,0]})x₁x₂ + {A[1,1]}x₂²")
    print(f"     = {A[0,0]}({x[0]})² + {A[0,1]+A[1,0]}({x[0]})({x[1]}) + {A[1,1]}({x[1]})²")
    print(f"     = {A[0,0]*x[0]**2} + {(A[0,1]+A[1,0])*x[0]*x[1]} + {A[1,1]*x[1]**2}")
    print(f"     = {result}")

    # Mahalanobis distance example
    print("\n--- Application: Mahalanobis Distance ---")
    mu = np.array([1, 1])
    Sigma = np.array([[2, 0],
                      [0, 2]])
    Sigma_inv = np.linalg.inv(Sigma)

    print_matrix("x", x)
    print_matrix("μ", mu)
    print_matrix("Σ", Sigma)
    print_matrix("Σ⁻¹", Sigma_inv)

    diff = x - mu
    print_matrix("x - μ", diff)

    mahal_sq = diff @ Sigma_inv @ diff
    print(f"\nMahalanobis distance² = (x-μ)ᵀΣ⁻¹(x-μ)")
    print(f"                      = [{diff[0]}, {diff[1]}] × (1/2)I × [{diff[0]}, {diff[1]}]ᵀ")
    print(f"                      = (1/2) × ({diff[0]}² + {diff[1]}²)")
    print(f"                      = (1/2) × {diff[0]**2 + diff[1]**2}")
    print(f"                      = {mahal_sq}")


# =============================================================================
# 8. SUMMARY CHEAT SHEET
# =============================================================================
def cheat_sheet():
    section_header("MATRIX CALCULATION CHEAT SHEET")

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    2×2 MATRIX FORMULAS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For A = [[a, b],    Determinant: det(A) = ad - bc                  │
│           [c, d]]                                                    │
│                                                                      │
│                      Inverse: A⁻¹ = (1/det(A)) × [[d, -b],          │
│                                                   [-c, a]]          │
│                                                                      │
│                      Trace: tr(A) = a + d                           │
│                                                                      │
│  Eigenvalues: λ = (tr(A) ± √(tr(A)² - 4·det(A))) / 2               │
│                                                                      │
│  Characteristic equation: λ² - tr(A)·λ + det(A) = 0                │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    SPECIAL MATRICES                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Identity I:     I⁻¹ = I,  det(I) = 1                               │
│                                                                      │
│  Diagonal D:     D⁻¹[i,i] = 1/D[i,i],  det(D) = ∏ D[i,i]           │
│                                                                      │
│  σ²I:            (σ²I)⁻¹ = (1/σ²)I                                  │
│                                                                      │
│  Symmetric A:    A = Aᵀ,  eigenvalues are real                      │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    KEY PROPERTIES                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  (AB)ᵀ = BᵀAᵀ              (ABC)⁻¹ = C⁻¹B⁻¹A⁻¹                     │
│                                                                      │
│  det(AB) = det(A)·det(B)   det(A⁻¹) = 1/det(A)                     │
│                                                                      │
│  det(cA) = cⁿ·det(A)       for n×n matrix                          │
│                                                                      │
│  tr(AB) = tr(BA)           tr(A+B) = tr(A) + tr(B)                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    PATTERN RECOGNITION FORMULAS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Scatter matrix: S = Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ                          │
│                                                                      │
│  Covariance: Σ = (1/N) Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ                        │
│                                                                      │
│  Mahalanobis: d² = (x - μ)ᵀ Σ⁻¹ (x - μ)                            │
│                                                                      │
│  FLD: v = S_W⁻¹(μ₁ - μ₂)                                            │
│                                                                      │
│  PCA: Find eigenvectors of S = Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MATRIX CALCULATIONS FOR PATTERN RECOGNITION")
    print("Step-by-Step Solutions for Exam Preparation")
    print("=" * 70)

    # matrix_multiplication()
    matrix_transpose()
    # matrix_determinant()
    # matrix_inverse()
    # eigenvalues_eigenvectors()
    # outer_product()
    # quadratic_form()
    # cheat_sheet()

    print("\n" + "=" * 70)
    print("END OF MATRIX CALCULATIONS GUIDE")
    print("=" * 70)


if __name__ == "__main__":
    main()
