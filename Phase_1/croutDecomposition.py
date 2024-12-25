import numpy as np

def lu_decomposition_crout(A):
    A = np.array(A, dtype=float)

    if A.ndim != 2 or A.size == 0:
        raise ValueError("Input must be a non-empty 2D matrix")

    n = A.shape[0]

    # Check matrix is square
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square")

    L = np.zeros_like(A, dtype=float)
    U = np.eye(n, dtype=float)

    for j in range(n):
        # Compute column j in L
        for i in range(j, n):
            L[i, j] = A[i, j] - np.sum(L[i, :j] * U[:j, j])

        if np.isclose(L[j, j], 0):
            raise ValueError("Matrix is singular and cannot be decomposed using LU decomposition")

        # Compute row j in U
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - np.sum(L[j, :j] * U[:j, i])) / L[j, j]

    return L, U


def forward_substitution(L, b):
    n = len(L)
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]
    return y

def backward_substitution(U, y):
    n = len(U)
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i]
    return x

def solve_system(A, b):
    L, U = lu_decomposition_crout(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x
