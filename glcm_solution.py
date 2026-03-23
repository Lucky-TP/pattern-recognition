"""
Problem 3: Gray-Level Co-occurrence Matrix (GLCM) Calculation
==============================================================
Pattern Recognition Exam Practice

Given a 5x5 grayscale image with intensity values in {0,1,2,3}:
I = [[0, 1, 2, 2, 3],
     [1, 2, 0, 2, 0],
     [3, 0, 3, 2, 1],
     [1, 2, 2, 2, 3],
     [0, 0, 1, 1, 2]]

Calculate GLCM for distance d=1 in 4 directions: 0°, 45°, 90°, 135°

IMPORTANT (from slides):
- 0° counts BOTH 0° (right) AND 180° (left) directions
- 45° counts BOTH 45° AND 225° directions
- 90° counts BOTH 90° (up) AND 270° (down) directions
- 135° counts BOTH 135° AND 315° directions

This makes the GLCM symmetric!

Direction offsets (row, col):
- 0°   (→): (0, +d)   and 180° (←): (0, -d)
- 45°  (↗): (-d, +d)  and 225° (↙): (+d, -d)
- 90°  (↑): (-d, 0)   and 270° (↓): (+d, 0)
- 135° (↖): (-d, -d)  and 315° (↘): (+d, +d)
"""

import numpy as np

def compute_glcm(image, d, angle, levels=4):
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM)

    Counts pairs in BOTH directions (θ and θ+180°) as per slides.
    This makes the resulting matrix symmetric.

    Parameters:
    - image: 2D numpy array (grayscale image)
    - d: distance
    - angle: direction in degrees (0, 45, 90, 135)
    - levels: number of gray levels

    Returns:
    - GLCM matrix (levels x levels)
    """
    rows, cols = image.shape
    glcm = np.zeros((levels, levels), dtype=int)

    # Define BOTH offsets for angle θ and θ+180°
    # Format: [(dr1, dc1), (dr2, dc2)] for θ and θ+180°

    if angle == 0:
        # 0° (right) and 180° (left)
        offsets = [(0, d), (0, -d)]
    elif angle == 45:
        # 45° (upper-right) and 225° (lower-left)
        offsets = [(-d, d), (d, -d)]
    elif angle == 90:
        # 90° (up) and 270° (down)
        offsets = [(-d, 0), (d, 0)]
    elif angle == 135:
        # 135° (upper-left) and 315° (lower-right)
        offsets = [(-d, -d), (d, d)]
    else:
        raise ValueError("Angle must be 0, 45, 90, or 135")

    # Count co-occurrences for both directions
    for dr, dc in offsets:
        for i in range(rows):
            for j in range(cols):
                ni, nj = i + dr, j + dc
                if 0 <= ni < rows and 0 <= nj < cols:
                    # Count pair (image[i,j], image[ni,nj])
                    glcm[image[i, j], image[ni, nj]] += 1

    return glcm


def compute_glcm_step_by_step(image, d, angle, levels=4):
    """
    Compute GLCM with detailed step-by-step output showing each pair found.
    """
    rows, cols = image.shape
    glcm = np.zeros((levels, levels), dtype=int)
    pairs_found = []

    if angle == 0:
        offsets = [(0, d), (0, -d)]
        dir_names = ["0° (→)", "180° (←)"]
    elif angle == 45:
        offsets = [(-d, d), (d, -d)]
        dir_names = ["45° (↗)", "225° (↙)"]
    elif angle == 90:
        offsets = [(-d, 0), (d, 0)]
        dir_names = ["90° (↑)", "270° (↓)"]
    elif angle == 135:
        offsets = [(-d, -d), (d, d)]
        dir_names = ["135° (↖)", "315° (↘)"]

    for idx, (dr, dc) in enumerate(offsets):
        print(f"\n  Direction {dir_names[idx]}: offset (Δrow={dr}, Δcol={dc})")
        print(f"  " + "-" * 50)

        dir_pairs = []
        for i in range(rows):
            for j in range(cols):
                ni, nj = i + dr, j + dc
                if 0 <= ni < rows and 0 <= nj < cols:
                    val1 = image[i, j]
                    val2 = image[ni, nj]
                    glcm[val1, val2] += 1
                    dir_pairs.append((val1, val2))
                    pairs_found.append((val1, val2))

        print(f"  Pairs found: {dir_pairs}")
        print(f"  Count: {len(dir_pairs)} pairs")

    return glcm, pairs_found


def main():
    # Define the image
    I = np.array([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 2, 0],
        [3, 0, 3, 2, 1],
        [1, 2, 2, 2, 3],
        [0, 0, 1, 1, 2]
    ])

    print("=" * 70)
    print("GLCM Calculation (Gray-Level Co-occurrence Matrix)")
    print("=" * 70)

    print("\nInput Image I (5×5):")
    print("     col0 col1 col2 col3 col4")
    for i in range(5):
        print(f"row{i}   {I[i, 0]}    {I[i, 1]}    {I[i, 2]}    {I[i, 3]}    {I[i, 4]}")

    print(f"\nGray levels: {{0, 1, 2, 3}}")
    print(f"Distance d = 1")

    print("\n" + "=" * 70)
    print("IMPORTANT: How GLCM counts pairs (from slides)")
    print("=" * 70)
    print("""
For each angle θ, we count pairs in BOTH θ AND θ+180° directions:

• 0°:   Count pairs going RIGHT (→) AND LEFT (←)
• 45°:  Count pairs going UPPER-RIGHT (↗) AND LOWER-LEFT (↙)
• 90°:  Count pairs going UP (↑) AND DOWN (↓)
• 135°: Count pairs going UPPER-LEFT (↖) AND LOWER-RIGHT (↘)

This makes the GLCM matrix SYMMETRIC (GLCM[i,j] contributes equally as GLCM[j,i])
""")

    # Calculate and display GLCM for each angle
    angles = [0, 45, 90, 135]

    for angle in angles:
        print("\n" + "=" * 70)
        print(f"GLCM for θ = {angle}° (counting both {angle}° and {angle+180}° directions)")
        print("=" * 70)

        glcm, pairs = compute_glcm_step_by_step(I, d=1, angle=angle, levels=4)

        print(f"\n  All pairs combined: {pairs}")
        print(f"  Total pairs: {len(pairs)}")

        print(f"\n  GLCM Matrix P({angle}°):")
        print("         j=0   j=1   j=2   j=3")
        for i in range(4):
            print(f"  i={i}    {glcm[i, 0]:3d}   {glcm[i, 1]:3d}   {glcm[i, 2]:3d}   {glcm[i, 3]:3d}")

        print(f"\n  Sum of all entries: {glcm.sum()}")

        # Verify symmetry
        is_symmetric = np.allclose(glcm, glcm.T)
        print(f"  Matrix is symmetric: {is_symmetric}")

    # Detailed walkthrough for 0°
    print("\n" + "=" * 70)
    print("DETAILED WALKTHROUGH: θ = 0° (Horizontal)")
    print("=" * 70)

    print("""
For 0°, we scan the image and find ALL horizontal adjacent pairs:

Image:
     col0 col1 col2 col3 col4
row0   0    1    2    2    3
row1   1    2    0    2    0
row2   3    0    3    2    1
row3   1    2    2    2    3
row4   0    0    1    1    2

Direction 0° (→): Look at (i,j) and (i,j+1)
Direction 180° (←): Look at (i,j) and (i,j-1)

Equivalently: For each horizontally adjacent pair, count BOTH (a,b) and (b,a)
""")

    print("\nRow-by-row pair counting:")
    print("-" * 50)

    all_pairs = []
    for row in range(5):
        row_pairs = []
        print(f"\nRow {row}: {list(I[row, :])}")
        for col in range(4):
            a, b = I[row, col], I[row, col+1]
            row_pairs.append((a, b))
            row_pairs.append((b, a))
            all_pairs.append((a, b))
            all_pairs.append((b, a))
            print(f"  Position ({row},{col})-({row},{col+1}): values ({a},{b}) → count ({a},{b}) and ({b},{a})")

    print(f"\nTotal pairs for 0°: {len(all_pairs)}")

    # Count occurrences
    print("\nCounting each pair type:")
    from collections import Counter
    pair_counts = Counter(all_pairs)
    for pair in sorted(pair_counts.keys()):
        print(f"  ({pair[0]},{pair[1]}): {pair_counts[pair]}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: All GLCM Matrices")
    print("=" * 70)

    for angle in angles:
        glcm = compute_glcm(I, d=1, angle=angle, levels=4)
        print(f"\nP({angle}°):")
        print("     j=0  j=1  j=2  j=3")
        for i in range(4):
            print(f"i={i}  {glcm[i, 0]:3d}  {glcm[i, 1]:3d}  {glcm[i, 2]:3d}  {glcm[i, 3]:3d}")


if __name__ == "__main__":
    main()
