"""
k-Nearest Neighbors (k-NN) Classifier - Step-by-Step Solution
==============================================================
Pattern Recognition Exam Practice

Algorithm:
1. Calculate distance from query point to all training points
2. Select k nearest neighbors
3. Majority vote among k neighbors for classification

Key Formula (Euclidean Distance):
    d(x, x_i) = sqrt(sum((x_j - x_ij)^2))

Important Notes:
- k-NN is a lazy learner (no training phase)
- Choice of k affects decision boundary smoothness
- Odd k preferred for binary classification (avoid ties)
"""

import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_classifier():
    print("=" * 70)
    print("k-Nearest Neighbors (k-NN) Classifier - Step-by-Step Solution")
    print("=" * 70)

    # ===== Step 1: Define Training Data =====
    print("\n" + "="*70)
    print("STEP 1: Define Training Data")
    print("="*70)

    # Training data: [feature1, feature2, class]
    X_train = np.array([
        [1, 1],   # Class 1
        [1, 2],   # Class 1
        [2, 1],   # Class 1
        [2, 2],   # Class 1
        [6, 5],   # Class 2
        [7, 5],   # Class 2
        [6, 6],   # Class 2
        [7, 6],   # Class 2
        [4, 3],   # Class 3
        [5, 3],   # Class 3
        [4, 4],   # Class 3
        [5, 4],   # Class 3
    ])

    y_train = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    print("\nTraining Data:")
    print("-" * 40)
    print("  Point    Features (x1, x2)    Class")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"  x_{i+1:2d}       ({x[0]}, {x[1]})           ω_{y}")

    # ===== Step 2: Define Query Point =====
    print("\n" + "="*70)
    print("STEP 2: Define Query Point (Test Point)")
    print("="*70)

    query_point = np.array([3, 3])
    print(f"\nQuery point: x_query = ({query_point[0]}, {query_point[1]})")
    print("Task: Classify this point using k-NN")

    # ===== Step 3: Calculate Distances =====
    print("\n" + "="*70)
    print("STEP 3: Calculate Distances to All Training Points")
    print("="*70)

    print("\nEuclidean Distance Formula:")
    print("  d(x, x_i) = sqrt((x1 - x_i1)² + (x2 - x_i2)²)")
    print("            = sqrt((x1 - x_i1)² + (x2 - x_i2)²)")

    distances = []
    print("\nCalculating distances:")
    print("-" * 60)
    print("  Point    Features    Class    Distance Calculation")
    print("-" * 60)

    for i, (x, y) in enumerate(zip(X_train, y_train)):
        diff = query_point - x
        dist = euclidean_distance(query_point, x)
        distances.append((dist, y, i+1, x))

        print(f"  x_{i+1:2d}     ({x[0]}, {x[1]})      ω_{y}     "
              f"d = sqrt(({query_point[0]}-{x[0]})² + ({query_point[1]}-{x[1]})²)")
        print(f"                                    = sqrt({diff[0]}² + {diff[1]}²)")
        print(f"                                    = sqrt({diff[0]**2} + {diff[1]**2})")
        print(f"                                    = sqrt({diff[0]**2 + diff[1]**2})")
        print(f"                                    = {dist:.4f}")
        print()

    # ===== Step 4: Sort by Distance =====
    print("\n" + "="*70)
    print("STEP 4: Sort Points by Distance (Ascending)")
    print("="*70)

    distances.sort(key=lambda x: x[0])

    print("\nSorted distances (nearest to farthest):")
    print("-" * 50)
    print("  Rank    Point    Class    Distance")
    print("-" * 50)
    for rank, (dist, y, idx, x) in enumerate(distances, 1):
        print(f"  {rank:2d}       x_{idx:2d}      ω_{y}       {dist:.4f}")

    # ===== Step 5: k=1 Classification =====
    print("\n" + "="*70)
    print("STEP 5: Classification with k=1")
    print("="*70)

    k = 1
    neighbors = distances[:k]
    neighbor_classes = [n[1] for n in neighbors]

    print(f"\nFor k = {k}:")
    print(f"  Nearest neighbor: Point x_{neighbors[0][2]} (Class ω_{neighbors[0][1]})")
    print(f"  Distance: {neighbors[0][0]:.4f}")

    # Majority vote
    class_counts = Counter(neighbor_classes)
    predicted_class = class_counts.most_common(1)[0][0]

    vote_dict = {int(cls): int(cnt) for cls, cnt in class_counts.items()}
    print(f"\n  Vote count: {vote_dict}")
    print(f"\n  *** Prediction: Class ω_{int(predicted_class)} ***")

    # ===== Step 6: k=3 Classification =====
    print("\n" + "="*70)
    print("STEP 6: Classification with k=3")
    print("="*70)

    k = 3
    neighbors = distances[:k]
    neighbor_classes = [n[1] for n in neighbors]

    print(f"\nFor k = {k}:")
    print(f"  3 Nearest neighbors:")
    for i, (dist, y, idx, x) in enumerate(neighbors, 1):
        print(f"    {i}. Point x_{idx} (Class ω_{y}), Distance = {dist:.4f}")

    # Majority vote
    class_counts = Counter(neighbor_classes)
    vote_dict = {int(cls): int(cnt) for cls, cnt in class_counts.items()}
    print(f"\n  Vote count: {vote_dict}")

    predicted_class = class_counts.most_common(1)[0][0]

    print(f"\n  *** Prediction: Class ω_{int(predicted_class)} ***")
    print(f"      (Class ω_{int(predicted_class)} has {class_counts[predicted_class]} votes, "
          f"which is the majority)")

    # ===== Step 7: k=5 Classification =====
    print("\n" + "="*70)
    print("STEP 7: Classification with k=5")
    print("="*70)

    k = 5
    neighbors = distances[:k]
    neighbor_classes = [n[1] for n in neighbors]

    print(f"\nFor k = {k}:")
    print(f"  5 Nearest neighbors:")
    for i, (dist, y, idx, x) in enumerate(neighbors, 1):
        print(f"    {i}. Point x_{idx} (Class ω_{y}), Distance = {dist:.4f}")

    # Majority vote
    class_counts = Counter(neighbor_classes)
    vote_dict = {int(cls): int(cnt) for cls, cnt in class_counts.items()}
    print(f"\n  Vote count: {vote_dict}")

    predicted_class = class_counts.most_common(1)[0][0]

    print(f"\n  *** Prediction: Class ω_{int(predicted_class)} ***")

    # ===== Step 8: Effect of k =====
    print("\n" + "="*70)
    print("STEP 8: Effect of Different k Values")
    print("="*70)

    print("\nClassifications for different k values:")
    print("-" * 50)
    print("  k    Nearest Classes                Prediction")
    print("-" * 50)

    for k in [1, 3, 5, 7, 9]:
        neighbors = distances[:k]
        neighbor_classes = [n[1] for n in neighbors]
        class_counts = Counter(neighbor_classes)
        predicted = class_counts.most_common(1)[0][0]

        classes_str = ", ".join([f"ω_{c}" for c in neighbor_classes])
        print(f"  {k}    [{classes_str}]    ω_{predicted}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: k-NN Algorithm")
    print("="*70)

    print(f"""
Query Point: ({query_point[0]}, {query_point[1]})

Algorithm Steps:
1. Calculate distance from query to all training points
2. Sort by distance (ascending)
3. Select k nearest neighbors
4. Majority vote among neighbors

Key Observations:
• Small k (e.g., k=1): More sensitive to noise, complex boundary
• Large k (e.g., k=9): Smoother boundary, but may miss local patterns
• Odd k preferred for binary classification (avoid ties)

Distance Metrics (alternatives to Euclidean):
• Manhattan: d = |x1-x2| + |y1-y2|
• Minkowski: d = (|x1-x2|^p + |y1-y2|^p)^(1/p)
• Cosine: d = 1 - (x·y)/(||x|| ||y||)
""")

    # ===== Exam Tips =====
    print("\n" + "="*70)
    print("EXAM TIPS")
    print("="*70)

    print("""
Common Exam Questions:
1. Calculate distances and classify a new point
2. Draw decision boundary for different k values
3. Compare k-NN with other classifiers
4. Discuss effect of k on bias-variance tradeoff

Key Points to Remember:
• k-NN has NO training phase (lazy learning)
• Larger k → smoother decision boundary, higher bias, lower variance
• Smaller k → more complex boundary, lower bias, higher variance
• Computational cost: O(nd) for each query (n=samples, d=dimensions)
• Normalize features if they have different scales!

Quick Calculation Check:
• Always show distance calculations step by step
• Make sure to sort distances correctly
• Count votes carefully, especially for ties
""")

    return distances


if __name__ == "__main__":
    knn_classifier()
