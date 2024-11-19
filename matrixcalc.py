import numpy as np

# Define the matrix
A = np.array([
    [3, 2, -1],
    [1, 6, 3],
    [2, -4, 0]
])

# Calculate the determinant of A
det_A = np.linalg.det(A)

# Calculate the inverse of A
if det_A != 0:
    A_inv = np.linalg.inv(A)
else:
    A_inv = None

det_A, A_inv
