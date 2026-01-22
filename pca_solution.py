"""
Problem 4: Principal Component Analysis (PCA)

Given 2D dataset D = {(1,1), (2,3), (3,2), (4,4)}
Find:
1. Direction of the first principal component (PC1)
2. Project all data points onto 1D space in the direction of PC1
"""

import numpy as np

def main():
    print("=" * 60)
    print("Problem 4: Principal Component Analysis (PCA)")
    print("=" * 60)

    # Step 1: Define the data
    data = np.array([
        [1, 1],
        [2, 3],
        [3, 2],
        [4, 4]
    ])

    print("\nStep 1: Original Data")
    print("-" * 40)
    print("D = {(1,1), (2,3), (3,2), (4,4)}")
    print("\nData matrix X:")
    print(data)

    # Step 2: Calculate the mean
    mean = np.mean(data, axis=0)
    print("\nStep 2: Calculate Mean")
    print("-" * 40)
    print(f"Mean μ = [{mean[0]}, {mean[1]}]")
    print(f"μ_x = (1+2+3+4)/4 = {mean[0]}")
    print(f"μ_y = (1+3+2+4)/4 = {mean[1]}")

    # Step 3: Center the data (subtract mean)
    centered_data = data - mean
    print("\nStep 3: Center the Data (X - μ)")
    print("-" * 40)
    print("Centered data:")
    for i, (orig, cent) in enumerate(zip(data, centered_data)):
        print(f"  ({orig[0]}, {orig[1]}) - ({mean[0]}, {mean[1]}) = ({cent[0]}, {cent[1]})")

    print("\nCentered data matrix:")
    print(centered_data)

    # Step 4: Calculate covariance matrix
    n = len(data)
    cov_matrix = (centered_data.T @ centered_data) / (n - 1)  # Sample covariance

    print("\nStep 4: Calculate Covariance Matrix")
    print("-" * 40)
    print("Covariance matrix Σ = (1/(n-1)) * X_centered^T * X_centered")
    print(f"where n = {n}")
    print("\nManual calculation:")
    print(f"Σ_11 = Var(x) = Σ(x_i - μ_x)² / (n-1)")
    var_x = np.sum(centered_data[:, 0] ** 2) / (n - 1)
    var_y = np.sum(centered_data[:, 1] ** 2) / (n - 1)
    cov_xy = np.sum(centered_data[:, 0] * centered_data[:, 1]) / (n - 1)
    print(f"     = [(-1.5)² + (-0.5)² + (0.5)² + (1.5)²] / 3")
    print(f"     = [{centered_data[0, 0] ** 2} + {centered_data[1, 0] ** 2} + {centered_data[2, 0] ** 2} + {centered_data[3, 0] ** 2}] / 3")
    print(f"     = {np.sum(centered_data[:, 0] ** 2)} / 3 = {var_x:.4f}")

    print(f"\nΣ_22 = Var(y) = Σ(y_i - μ_y)² / (n-1)")
    print(f"     = [(-1.5)² + (0.5)² + (-0.5)² + (1.5)²] / 3")
    print(f"     = {np.sum(centered_data[:, 1] ** 2)} / 3 = {var_y:.4f}")

    print(f"\nΣ_12 = Σ_21 = Cov(x,y) = Σ(x_i - μ_x)(y_i - μ_y) / (n-1)")
    print(f"     = [(-1.5)(-1.5) + (-0.5)(0.5) + (0.5)(-0.5) + (1.5)(1.5)] / 3")
    products = centered_data[:, 0] * centered_data[:, 1]
    print(f"     = [{products[0]} + {products[1]} + {products[2]} + {products[3]}] / 3")
    print(f"     = {np.sum(products)} / 3 = {cov_xy:.4f}")

    print("\nCovariance Matrix Σ:")
    print(f"Σ = [[{cov_matrix[0, 0]:.4f}, {cov_matrix[0, 1]:.4f}],")
    print(f"     [{cov_matrix[1, 0]:.4f}, {cov_matrix[1, 1]:.4f}]]")

    # Step 5: Calculate eigenvalues and eigenvectors
    print("\nStep 5: Calculate Eigenvalues and Eigenvectors")
    print("-" * 40)
    print("Solve: det(Σ - λI) = 0")
    print(f"\ndet([[{cov_matrix[0, 0]:.4f} - λ, {cov_matrix[0, 1]:.4f}],")
    print(f"      [{cov_matrix[1, 0]:.4f}, {cov_matrix[1, 1]:.4f} - λ]]) = 0")

    a = cov_matrix[0, 0]
    b = cov_matrix[0, 1]
    c = cov_matrix[1, 0]
    d = cov_matrix[1, 1]

    print(f"\n(a - λ)(d - λ) - bc = 0")
    print(f"({a:.4f} - λ)({d:.4f} - λ) - ({b:.4f})({c:.4f}) = 0")
    print(f"λ² - (a+d)λ + (ad - bc) = 0")
    print(f"λ² - {a + d:.4f}λ + {a * d - b * c:.4f} = 0")

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nUsing quadratic formula:")
    print(f"λ = [(a+d) ± √((a+d)² - 4(ad-bc))] / 2")
    discriminant = (a + d) ** 2 - 4 * (a * d - b * c)
    print(f"λ = [{a + d:.4f} ± √{discriminant:.4f}] / 2")
    print(f"λ = [{a + d:.4f} ± {np.sqrt(discriminant):.4f}] / 2")

    print(f"\nEigenvalues:")
    print(f"  λ₁ = {eigenvalues[0]:.4f} (largest)")
    print(f"  λ₂ = {eigenvalues[1]:.4f}")

    print(f"\nEigenvectors (columns):")
    print(f"  v₁ (for λ₁) = [{eigenvectors[0, 0]:.4f}, {eigenvectors[1, 0]:.4f}]^T")
    print(f"  v₂ (for λ₂) = [{eigenvectors[0, 1]:.4f}, {eigenvectors[1, 1]:.4f}]^T")

    # Step 6: PC1 direction
    pc1 = eigenvectors[:, 0]
    print("\nStep 6: First Principal Component (PC1)")
    print("-" * 40)
    print(f"PC1 direction (eigenvector for largest eigenvalue λ₁):")
    print(f"  PC1 = [{pc1[0]:.4f}, {pc1[1]:.4f}]^T")

    # Normalize to unit vector (should already be normalized)
    pc1_normalized = pc1 / np.linalg.norm(pc1)
    print(f"\nNormalized PC1: [{pc1_normalized[0]:.4f}, {pc1_normalized[1]:.4f}]^T")
    print(f"  (magnitude = {np.linalg.norm(pc1_normalized):.4f})")

    # Step 7: Project data onto PC1
    print("\nStep 7: Project Data onto PC1")
    print("-" * 40)
    print("Projection: z_i = (x_i - μ)^T · PC1")
    print("\nProjections of each point:")

    projections = centered_data @ pc1
    for i, (orig, cent, proj) in enumerate(zip(data, centered_data, projections)):
        print(f"  Point ({orig[0]}, {orig[1]}): z_{i + 1} = ({cent[0]}, {cent[1]})^T · ({pc1[0]:.4f}, {pc1[1]:.4f})^T = {proj:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n1. PC1 Direction: [{pc1[0]:.4f}, {pc1[1]:.4f}]^T")
    print(f"   (or equivalently: [{-pc1[0]:.4f}, {-pc1[1]:.4f}]^T)")
    print(f"\n2. Variance explained by PC1: {eigenvalues[0] / sum(eigenvalues) * 100:.2f}%")
    print(f"\n3. Projected 1D coordinates:")
    for i, (orig, proj) in enumerate(zip(data, projections)):
        print(f"   ({orig[0]}, {orig[1]}) → {proj:.4f}")


if __name__ == "__main__":
    main()
