"""
Sequential Forward Selection (SFS) - Step-by-Step Solution
===========================================================
Pattern Recognition Exam Practice

Algorithm:
1. Start with empty feature set
2. For each remaining feature, evaluate J(current + feature)
3. Add the feature that maximizes J
4. Repeat until desired number of features

Key Points:
- SFS is a greedy algorithm (locally optimal choices)
- May not find globally optimal feature subset
- Computationally efficient: O(d²) vs O(2^d) for exhaustive search
"""

import numpy as np
from itertools import combinations

def sfs_solution():
    print("=" * 65)
    print("Sequential Forward Selection (SFS) - Step-by-Step Solution")
    print("=" * 65)

    # ===== Problem Setup =====
    print("\n" + "="*65)
    print("PROBLEM SETUP")
    print("="*65)

    print("""
Given: J scores (Fisher criterion) for feature subsets
Task: Select the best 2 features using SFS algorithm

Available features: f₁, f₂, f₃, f₄
""")

    # Define J scores for individual features
    J_single = {
        'f1': 0.8,
        'f2': 0.6,
        'f3': 0.9,
        'f4': 0.5
    }

    # Define J scores for feature pairs
    J_pairs = {
        ('f1', 'f2'): 1.2,
        ('f1', 'f3'): 1.1,
        ('f1', 'f4'): 1.5,
        ('f2', 'f3'): 1.8,
        ('f2', 'f4'): 1.0,
        ('f3', 'f4'): 1.3
    }

    print("J scores for individual features:")
    for f, j in sorted(J_single.items()):
        print(f"  J({f}) = {j}")

    print("\nJ scores for feature pairs:")
    for (f1, f2), j in sorted(J_pairs.items()):
        print(f"  J({f1}, {f2}) = {j}")

    # ===== Step 1: Select first feature =====
    print("\n" + "="*65)
    print("STEP 1: Select first feature (greedy - pick best single feature)")
    print("="*65)

    print("\nEvaluate each feature individually:")
    for f, j in sorted(J_single.items(), key=lambda x: -x[1]):
        print(f"  J({f}) = {j}")

    best_first = max(J_single.items(), key=lambda x: x[1])
    selected = [best_first[0]]

    print(f"\nBest single feature: {best_first[0]} with J = {best_first[1]}")
    print(f"\n→ Selected features so far: {{{selected[0]}}}")

    # ===== Step 2: Select second feature =====
    print("\n" + "="*65)
    print("STEP 2: Select second feature (add to existing set)")
    print("="*65)

    print(f"\nCurrent set: {{{selected[0]}}}")
    print("Evaluate adding each remaining feature:")

    remaining = [f for f in J_single.keys() if f not in selected]
    candidates = []

    for f in remaining:
        pair = tuple(sorted([selected[0], f]))
        j = J_pairs[pair]
        candidates.append((f, j))
        print(f"  J({{{selected[0]}, {f}}}) = J{pair} = {j}")

    best_second = max(candidates, key=lambda x: x[1])
    selected.append(best_second[0])

    print(f"\nBest addition: {best_second[0]} with J = {best_second[1]}")
    print(f"\n→ Selected features: {{{selected[0]}, {selected[1]}}}")

    # ===== Verify SFS result =====
    print("\n" + "="*65)
    print("STEP 3: Verify SFS Result")
    print("="*65)

    final_pair = tuple(sorted(selected))
    sfs_j = J_pairs[final_pair]

    print(f"\nSFS selected: {{{selected[0]}, {selected[1]}}}")
    print(f"J score: {sfs_j}")

    # ===== Compare with optimal =====
    print("\n" + "="*65)
    print("STEP 4: Compare with Exhaustive Search (Optimal)")
    print("="*65)

    print("\nAll possible pairs and their J scores:")
    for (f1, f2), j in sorted(J_pairs.items(), key=lambda x: -x[1]):
        marker = " ← SFS selected" if (f1, f2) == final_pair or (f2, f1) == final_pair else ""
        optimal_marker = " ★ OPTIMAL" if j == max(J_pairs.values()) else ""
        print(f"  J({f1}, {f2}) = {j}{marker}{optimal_marker}")

    optimal_pair = max(J_pairs.items(), key=lambda x: x[1])

    print(f"\nOptimal pair (exhaustive search): {optimal_pair[0]} with J = {optimal_pair[1]}")
    print(f"SFS selected pair: {final_pair} with J = {sfs_j}")

    if sfs_j == optimal_pair[1]:
        print("\n✓ SFS found the optimal solution!")
    else:
        print(f"\n✗ SFS is SUBOPTIMAL!")
        print(f"  SFS J = {sfs_j}")
        print(f"  Optimal J = {optimal_pair[1]}")
        print(f"  Difference = {optimal_pair[1] - sfs_j}")

    # ===== Why SFS can be suboptimal =====
    print("\n" + "="*65)
    print("ANALYSIS: Why SFS May Not Find Optimal Solution")
    print("="*65)

    print(f"""
In this example:
1. SFS first selected {selected[0]} because J({selected[0]}) = {J_single[selected[0]]} was highest
2. Then added {selected[1]} because J({{{selected[0]}, {selected[1]}}}) = {sfs_j} was best given {selected[0]}

However, the optimal pair is {optimal_pair[0]} with J = {optimal_pair[1]}

Key Insight:
- The best individual features don't always combine to form the best pair
- Features can be:
  • Redundant: individually good but provide similar information
  • Complementary: individually mediocre but excellent together

SFS Limitations:
1. Greedy approach - makes locally optimal choices
2. Cannot "undo" previous selections
3. Misses feature interactions discovered only in combination

When SFS works well:
- When good features remain good in combination
- When feature interactions are monotonic
- As an approximation when exhaustive search is infeasible
""")

    # ===== Extended Example: SFS to 3 features =====
    print("\n" + "="*65)
    print("EXTENDED EXAMPLE: Continue SFS to 3 features")
    print("="*65)

    # Define J scores for triplets
    J_triplets = {
        ('f1', 'f2', 'f3'): 2.1,
        ('f1', 'f2', 'f4'): 1.9,
        ('f1', 'f3', 'f4'): 2.0,
        ('f2', 'f3', 'f4'): 2.4
    }

    print(f"\nJ scores for feature triplets:")
    for features, j in sorted(J_triplets.items()):
        print(f"  J{features} = {j}")

    print(f"\nCurrent set: {{{selected[0]}, {selected[1]}}}")
    print("Evaluate adding each remaining feature:")

    remaining = [f for f in J_single.keys() if f not in selected]
    candidates = []

    for f in remaining:
        triplet = tuple(sorted(selected + [f]))
        j = J_triplets[triplet]
        candidates.append((f, j))
        print(f"  J({{{selected[0]}, {selected[1]}, {f}}}) = {j}")

    best_third = max(candidates, key=lambda x: x[1])
    selected.append(best_third[0])

    print(f"\nBest addition: {best_third[0]} with J = {best_third[1]}")
    print(f"\n→ Final SFS selection (3 features): {{{selected[0]}, {selected[1]}, {selected[2]}}}")

    optimal_triplet = max(J_triplets.items(), key=lambda x: x[1])
    print(f"\nOptimal triplet: {optimal_triplet[0]} with J = {optimal_triplet[1]}")

    # ===== Summary =====
    print("\n" + "="*65)
    print("SUMMARY: Sequential Forward Selection Algorithm")
    print("="*65)

    print("""
Algorithm Pseudocode:
    S = {}  # selected features
    for k = 1 to desired_features:
        best_feature = None
        best_J = -∞
        for f in remaining_features:
            J_new = evaluate(S ∪ {f})
            if J_new > best_J:
                best_J = J_new
                best_feature = f
        S = S ∪ {best_feature}
    return S

Complexity:
- Exhaustive search: O(C(d,k)) = O(d!/(k!(d-k)!))
- SFS: O(k × d) evaluations

Trade-off:
- SFS is fast but may miss optimal solution
- Use SFS when d is large and exhaustive search is infeasible
- For small d, exhaustive search guarantees optimal solution
""")

    print("\n" + "="*65)
    print("EXAM TIP")
    print("="*65)
    print("""
On exams, you may be asked to:
1. Perform SFS step by step (show your work!)
2. Compare SFS result with optimal (exhaustive search)
3. Explain why SFS might fail

Always:
- Show the J value for each candidate at each step
- Clearly state which feature you select and why
- If asked, verify by comparing with exhaustive search
""")


if __name__ == "__main__":
    sfs_solution()
