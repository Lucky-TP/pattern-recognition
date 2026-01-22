"""
Problem 3: Gray-Level Co-occurrence Matrix (GLCM) Calculation

Given a 5x5 grayscale image with intensity values in {0,1,2,3}:
I = [[0, 1, 2, 2, 3],
     [1, 2, 0, 2, 0],
     [3, 0, 3, 2, 1],
     [1, 2, 2, 2, 3],
     [0, 0, 1, 1, 2]]

Calculate GLCM for distance d=1 in 4 directions: 0°, 45°, 90°, 135°
"""

import numpy as np

def compute_glcm(image, d, angle, levels=4):
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM)

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

    # Define offset based on angle
    # 0°: horizontal right (0, d)
    # 45°: diagonal upper-right (-d, d)
    # 90°: vertical up (-d, 0)
    # 135°: diagonal upper-left (-d, -d)

    if angle == 0:
        dr, dc = 0, d
    elif angle == 45:
        dr, dc = -d, d
    elif angle == 90:
        dr, dc = -d, 0
    elif angle == 135:
        dr, dc = -d, -d
    else:
        raise ValueError("Angle must be 0, 45, 90, or 135")

    # Count co-occurrences (symmetric - count both directions)
    for i in range(rows):
        for j in range(cols):
            # Check if neighbor is within bounds
            ni, nj = i + dr, j + dc
            if 0 <= ni < rows and 0 <= nj < cols:
                glcm[image[i, j], image[ni, nj]] += 1

            # Also count the reverse direction for symmetry
            ni, nj = i - dr, j - dc
            if 0 <= ni < rows and 0 <= nj < cols:
                glcm[image[i, j], image[ni, nj]] += 1

    return glcm


def main():
    # Define the image
    I = np.array([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 2, 0],
        [3, 0, 3, 2, 1],
        [1, 2, 2, 2, 3],
        [0, 0, 1, 1, 2]
    ])

    print("=" * 60)
    print("Problem 3: GLCM Calculation")
    print("=" * 60)
    print("\nInput Image I (5x5):")
    print(I)
    print(f"\nGray levels: {{0, 1, 2, 3}}")
    print(f"Distance d = 1")

    angles = [0, 45, 90, 135]

    for angle in angles:
        print(f"\n{'-' * 60}")
        print(f"GLCM for θ = {angle}°")
        print(f"{'-' * 60}")

        glcm = compute_glcm(I, d=1, angle=angle, levels=4)

        print("\nGLCM Matrix:")
        print("     j=0  j=1  j=2  j=3")
        for i in range(4):
            print(f"i={i}  {glcm[i, 0]:3d}  {glcm[i, 1]:3d}  {glcm[i, 2]:3d}  {glcm[i, 3]:3d}")

        print(f"\nTotal pairs: {glcm.sum()}")

    # Also show step-by-step for 0° direction
    print("\n" + "=" * 60)
    print("Step-by-step explanation for θ = 0° (horizontal)")
    print("=" * 60)
    print("\nFor θ = 0°, we look at pixel pairs (i,j) and (i, j+1)")
    print("Counting all horizontal adjacent pairs (both directions):\n")

    d = 1
    pairs_0 = []
    for i in range(5):
        for j in range(5 - d):
            val1 = I[i, j]
            val2 = I[i, j + d]
            pairs_0.append((val1, val2))
            pairs_0.append((val2, val1))  # symmetric

    print(f"All pairs (i, j) for 0°: {pairs_0}")
    print(f"Total: {len(pairs_0)} pairs")


if __name__ == "__main__":
    main()
