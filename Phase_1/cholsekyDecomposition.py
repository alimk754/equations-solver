import numpy as np


def cholesky_decomposition(A):
    A = np.array(A, dtype=float)

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")

    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")

    if not np.all(np.linalg.eigvals(A) >= 0):
        raise ValueError("Matrix must be positive definite")

    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum = np.dot(L[i, :i], L[i, :i])  # sum of pow(L[i, j], 2)
                L[i, i] = np.sqrt(A[i, i] - sum)
            else:
                sum = np.dot(L[i, :j], L[j, :j])  # sum of L[i, j] * L[j, j]
                L[i, j] = (A[i, j] - sum) / L[j, j]

    return L, L.T


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
        x[i] = (y[i] - np.sum(U[i, i + 1:] * x[i + 1:])) / U[i, i]
    return x


def solve_system(A, b):
    L, U = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x
