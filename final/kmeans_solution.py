"""
K-Means Clustering - Step-by-Step Solution
==========================================
Pattern Recognition Exam Practice

Algorithm:
1. Initialize k centroids randomly (or using specific method)
2. Assignment Step: Assign each point to nearest centroid
3. Update Step: Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence (centroids don't change)

Key Formulas:
    Assignment: c_i = argmin_j ||x_i - μ_j||²
    Update: μ_j = (1/n_j) × Σ x_i  (for all x_i assigned to cluster j)

Convergence: K-means always converges (may be local minimum)
"""

import numpy as np


def kmeans_clustering():
    print("=" * 70)
    print("K-Means Clustering - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Data Points")
    print("="*70)

    # 2D data points
    X = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [7, 5],
        [8, 5],
        [7, 6],
        [8, 6],
    ])

    print("\nData Points:")
    print("-" * 40)
    for i, x in enumerate(X):
        print(f"  x_{i+1} = ({x[0]}, {x[1]})")

    print(f"\nTotal points: {len(X)}")

    # ===== Step 2: Initialize Centroids =====
    print("\n" + "="*70)
    print("STEP 2: Initialize Centroids (k=2)")
    print("="*70)

    k = 2  # Number of clusters

    # Manual initialization (for demonstration)
    centroids = np.array([
        [1.0, 1.0],  # μ₁ (initial)
        [7.0, 5.0],  # μ₂ (initial)
    ])

    print(f"\nNumber of clusters: k = {k}")
    print("\nInitial centroids:")
    for i, c in enumerate(centroids):
        print(f"  μ_{i+1} = ({c[0]}, {c[1]})")

    # ===== K-Means Iterations =====
    print("\n" + "="*70)
    print("K-MEANS ITERATIONS")
    print("="*70)

    max_iterations = 10
    iteration = 0
    assignments = np.zeros(len(X), dtype=int)  # Initialize assignments

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}")
        print(f"{'='*70}")

        # ===== Assignment Step =====
        print("\n--- Assignment Step ---")
        print(f"Assign each point to nearest centroid")
        print(f"\nCurrent centroids:")
        for i, c in enumerate(centroids):
            print(f"  μ_{i+1} = ({c[0]:.2f}, {c[1]:.2f})")

        print(f"\nDistance calculations (Euclidean):")
        print("-" * 70)
        print("  Point    (x1, x2)    d(x, μ₁)    d(x, μ₂)    Assigned Cluster")
        print("-" * 70)

        assignments = []
        for i, x in enumerate(X):
            # Calculate distances to all centroids
            distances = [np.sqrt(np.sum((x - c)**2)) for c in centroids]

            # Assign to nearest centroid
            cluster = np.argmin(distances)
            assignments.append(cluster)

            print(f"  x_{i+1}     ({x[0]}, {x[1]})      {distances[0]:.4f}      {distances[1]:.4f}           C_{cluster+1}")

        assignments = np.array(assignments)

        # ===== Update Step =====
        print("\n--- Update Step ---")
        print("Recalculate centroids as mean of assigned points")

        new_centroids = np.zeros_like(centroids)

        for j in range(k):
            cluster_points = X[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j]

            print(f"\nCluster C_{j+1} points: {[tuple(int(v) for v in x) for x in cluster_points]}")
            if len(cluster_points) > 0:
                print(f"  New μ_{j+1} = mean of cluster points")
                print(f"        = (1/{len(cluster_points)}) × Σ x_i")
                sum_x = np.sum(cluster_points[:, 0])
                sum_y = np.sum(cluster_points[:, 1])
                print(f"        = (1/{len(cluster_points)}) × ({sum_x}, {sum_y})")
                print(f"        = ({new_centroids[j][0]:.2f}, {new_centroids[j][1]:.2f})")
            else:
                print(f"  No points assigned, keeping centroid: ({new_centroids[j][0]:.2f}, {new_centroids[j][1]:.2f})")

        # Check convergence
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids)**2))

        print(f"\n--- Convergence Check ---")
        print(f"Centroid shift: {centroid_shift:.6f}")

        if centroid_shift < 1e-6:
            print("\n*** CONVERGED! Centroids no longer change. ***")
            centroids = new_centroids
            break

        print("Not converged, continuing...")
        centroids = new_centroids

    # ===== Final Results =====
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\nFinal centroids after {iteration} iterations:")
    for i, c in enumerate(centroids):
        print(f"  μ_{i+1} = ({c[0]:.2f}, {c[1]:.2f})")

    print(f"\nFinal cluster assignments:")
    for j in range(k):
        cluster_points = X[assignments == j]
        print(f"\nCluster C_{j+1}:")
        for x in cluster_points:
            print(f"  ({x[0]}, {x[1]})")

    # Calculate within-cluster sum of squares (WCSS)
    print("\n--- Within-Cluster Sum of Squares (WCSS) ---")
    wcss = 0
    for j in range(k):
        cluster_points = X[assignments == j]
        cluster_wcss = np.sum((cluster_points - centroids[j])**2)
        wcss += cluster_wcss
        print(f"  Cluster C_{j+1}: WCSS = {cluster_wcss:.4f}")
    print(f"  Total WCSS: {wcss:.4f}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: K-Means Algorithm")
    print("="*70)

    print(f"""
Data: {len(X)} points in 2D
Number of clusters: k = {k}

Final Centroids:
  μ₁ = ({centroids[0][0]:.2f}, {centroids[0][1]:.2f})
  μ₂ = ({centroids[1][0]:.2f}, {centroids[1][1]:.2f})

Convergence: {iteration} iterations

Algorithm Steps:
1. Initialize k centroids
2. Assignment: Assign each point to nearest centroid
3. Update: Recalculate centroids as cluster means
4. Repeat until convergence

Key Properties:
• Guaranteed to converge (but may be local minimum)
• Final result depends on initialization
• K-means++ improves initialization
• Complexity: O(n·k·d·i) where i=iterations, d=dimensions
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform k-means iterations step by step
2. Calculate WCSS (Within-Cluster Sum of Squares)
3. Determine optimal k using elbow method
4. Compare k-means with other clustering methods

Important Points:
• k-means minimizes WCSS (variance within clusters)
• Initialization affects final result
• Sensitive to outliers
• Assumes spherical clusters of similar size

Quick Checks:
• Each point assigned to exactly one cluster
• Centroid is the mean of cluster points
• Converged when centroids don't change

Elbow Method:
• Plot WCSS vs k
• Choose k where the curve "bends" (elbow point)
""")

    return centroids, assignments


if __name__ == "__main__":
    kmeans_clustering()
