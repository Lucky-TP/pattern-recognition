"""
Hierarchical Clustering - Step-by-Step Solution
===============================================
Pattern Recognition Exam Practice

Algorithm (Agglomerative / Bottom-Up):
1. Start with each point as its own cluster
2. Compute distance matrix between all clusters
3. Merge the two closest clusters
4. Update distance matrix
5. Repeat until one cluster remains

Linkage Methods:
- Single Linkage: min distance between points in clusters
- Complete Linkage: max distance between points in clusters
- Average Linkage: average distance between points in clusters
"""

import numpy as np


def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def hierarchical_clustering():
    print("=" * 70)
    print("Hierarchical Clustering - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Data Points")
    print("="*70)

    # 2D data points
    X = np.array([
        [1, 1],   # A
        [1, 2],   # B
        [2, 2],   # C
        [5, 5],   # D
        [6, 5],   # E
        [6, 6],   # F
    ])

    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    n = len(X)

    print("\nData Points:")
    print("-" * 40)
    for i, (x, label) in enumerate(zip(X, labels)):
        print(f"  {label} = ({x[0]}, {x[1]})")

    # ===== Step 2: Compute Initial Distance Matrix =====
    print("\n" + "="*70)
    print("STEP 2: Compute Initial Distance Matrix")
    print("="*70)

    print("\nEuclidean distance formula:")
    print("  d(x, y) = sqrt((x1-y1)² + (x2-y2)²)")

    # Compute distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = euclidean_distance(X[i], X[j])

    print("\nInitial Distance Matrix:")
    print("-" * 60)
    header = "       " + "   ".join([f"{l:6s}" for l in labels])
    print(header)
    print("-" * 60)
    for i, label in enumerate(labels):
        row = f"  {label}   "
        for j in range(n):
            row += f"{dist_matrix[i,j]:6.2f}"
        print(row)

    # ===== Step 3: Agglomerative Clustering (Single Linkage) =====
    print("\n" + "="*70)
    print("STEP 3: Agglomerative Clustering (Single Linkage)")
    print("="*70)

    print("""
Single Linkage: Distance between clusters = minimum distance
               between any pair of points from each cluster

d(C_i, C_j) = min{d(x, y) : x ∈ C_i, y ∈ C_j}
""")

    # Track clusters
    clusters = [[i] for i in range(n)]
    cluster_labels = [labels[i] for i in range(n)]
    merge_history = []

    step = 0
    while len(clusters) > 1:
        step += 1
        print(f"\n{'='*60}")
        print(f"MERGE STEP {step}")
        print(f"{'='*60}")

        # Show current clusters
        print(f"\nCurrent clusters:")
        for i, (cluster, cl) in enumerate(zip(clusters, cluster_labels)):
            points = [labels[p] for p in cluster]
            print(f"  C_{i+1} = {{{', '.join(points)}}}")

        # Find minimum distance
        min_dist = float('inf')
        merge_i, merge_j = 0, 0

        print(f"\nFinding minimum distance between clusters:")
        print("-" * 50)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Single linkage: minimum distance
                min_pair_dist = float('inf')
                for p1 in clusters[i]:
                    for p2 in clusters[j]:
                        d = euclidean_distance(X[p1], X[p2])
                        if d < min_pair_dist:
                            min_pair_dist = d

                ci_name = cluster_labels[i]
                cj_name = cluster_labels[j]
                print(f"  d({ci_name}, {cj_name}) = {min_pair_dist:.2f}")

                if min_pair_dist < min_dist:
                    min_dist = min_pair_dist
                    merge_i, merge_j = i, j

        # Merge clusters
        ci_name = cluster_labels[merge_i]
        cj_name = cluster_labels[merge_j]

        print(f"\n*** Minimum distance: d({ci_name}, {cj_name}) = {min_dist:.2f} ***")
        print(f"*** Merging clusters {ci_name} and {cj_name} ***")

        merge_history.append((ci_name, cj_name, min_dist))

        # Update clusters
        new_cluster = clusters[merge_i] + clusters[merge_j]
        new_label = f"({ci_name}+{cj_name})"

        # Remove old clusters and add merged
        new_clusters = []
        new_cluster_labels = []
        for k in range(len(clusters)):
            if k != merge_i and k != merge_j:
                new_clusters.append(clusters[k])
                new_cluster_labels.append(cluster_labels[k])

        new_clusters.append(new_cluster)
        new_cluster_labels.append(new_label)

        clusters = new_clusters
        cluster_labels = new_cluster_labels

    # ===== Step 4: Dendrogram =====
    print("\n" + "="*70)
    print("STEP 4: Dendrogram Visualization")
    print("="*70)

    print("\nMerge History (for dendrogram):")
    print("-" * 50)
    print("  Step    Clusters Merged        Distance")
    print("-" * 50)
    for i, (c1, c2, d) in enumerate(merge_history, 1):
        print(f"   {i}       {c1} + {c2}          {d:.2f}")

    print("""
Dendrogram (visual representation):

Distance
   |
   |              ___________ (D+E+F)
   |             |           |
   |        _____|           |________ (A+B+C)
   |       |     |           |
   |   ____|     |___________|
   |  |    |     |     |     |
   |  |    |  ___|  ___|  ___|
   |  |    |  |  |  |  |  |  |
   |__|____|__|__|__|__|__|__|____
        A  B     C     D  E  F
""")

    # ===== Step 5: Complete Linkage Comparison =====
    print("\n" + "="*70)
    print("STEP 5: Comparison of Linkage Methods")
    print("="*70)

    print("""
Linkage Methods Comparison:

1. Single Linkage (MIN):
   d(C_i, C_j) = min{d(x, y) : x ∈ C_i, y ∈ C_j}
   - Produces elongated clusters
   - Sensitive to noise and outliers
   - Can cause "chaining" effect

2. Complete Linkage (MAX):
   d(C_i, C_j) = max{d(x, y) : x ∈ C_i, y ∈ C_j}
   - Produces compact, spherical clusters
   - Less sensitive to noise
   - Tends to break large clusters

3. Average Linkage (UPGMA):
   d(C_i, C_j) = (1/|C_i||C_j|) × Σ d(x, y)
   - Compromise between single and complete
   - Produces balanced clusters

4. Ward's Method:
   - Minimizes within-cluster variance
   - Tends to produce equal-sized clusters
""")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: Hierarchical Clustering")
    print("="*70)

    print(f"""
Data: {n} points in 2D
Method: Agglomerative (bottom-up)
Linkage: Single linkage

Merge Sequence:
""")

    for i, (c1, c2, d) in enumerate(merge_history, 1):
        print(f"  {i}. {c1} + {c2} at distance {d:.2f}")

    print(f"""
Final Result: One cluster containing all points

Advantages:
• No need to specify number of clusters beforehand
• Dendrogram shows hierarchical structure
• Can cut dendrogram at any level for different k

Disadvantages:
• O(n³) time complexity
• Cannot undo merges
• Sensitive to noise (especially single linkage)
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Perform agglomerative clustering step by step
2. Calculate distances using different linkage methods
3. Draw dendrogram from merge history
4. Compare with k-means

Important Points:
• Start: n clusters, End: 1 cluster
• Single linkage uses minimum distance
• Complete linkage uses maximum distance
• Dendrogram shows merge distances on y-axis

Quick Checks:
• Number of merges = n - 1
• Each merge combines exactly 2 clusters
• Distances in dendrogram increase (for single/complete)
• Cutting dendrogram at height h gives clusters
""")

    return merge_history


if __name__ == "__main__":
    hierarchical_clustering()
