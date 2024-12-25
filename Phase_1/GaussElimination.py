import numpy as np


def scale(matrix):
    scaling_factors = np.max(np.abs(matrix), axis=1)
    return matrix / scaling_factors[:, np.newaxis]


def pivoting(matrix, start_row, use_scaling):
    copy_matrix = matrix.copy()
    if use_scaling:
        copy_matrix = scale(copy_matrix)
    max_idx = abs(copy_matrix[start_row:, start_row]).argmax() + start_row
    if copy_matrix[start_row, start_row] == 0:
        raise ValueError(f"Zero pivot encountered at row {start_row} while pivoting.")
    matrix[start_row], matrix[max_idx] = matrix[max_idx].copy(), matrix[start_row].copy()


def normalize_rows(Ab):
    n = len(Ab)
    for i in range(n):
        if Ab[i, i] == 0:
            raise ValueError(f"Zero pivot encountered at row {i}. Cannot normalize.")
        Ab[i, n] /= Ab[i, i]
        Ab[i, i] = 1
    return Ab


def forward_elimination_generator(A, b, use_scaling):
    Ab = np.column_stack((A, b))
    n = len(b)

    for pivot in range(n - 1):
        pivoting(Ab, pivot, use_scaling)
        for row in range(pivot + 1, n):
            factor = Ab[row, pivot] / Ab[pivot, pivot]
            Ab[row] -= factor * Ab[pivot]
        if np.all(Ab[pivot, :-1] == 0) and Ab[pivot, -1] != 0:
            raise ValueError(f"Inconsistent system: No solution at row {pivot}.")
        yield Ab


def backward_substitution(Ab):
    n = len(Ab)
    x = np.zeros(n)
    for i in reversed(range(n)):
        rhs = Ab[i, n]
        for j in range(i + 1, n):
            rhs -= x[j] * Ab[i, j]
        x[i] = rhs / Ab[i, i]
    return x


def backward_elimination(Ab):
    n = len(b)

    for pivot in reversed(range(1, n)):
        for row in reversed(range(pivot)):
            factor = Ab[row, pivot] / Ab[pivot, pivot]
            Ab[row] -= factor * Ab[pivot]
        yield Ab


def gauss_generator(A, b, use_scaling):
    for Ab in forward_elimination_generator(A, b, use_scaling):
        yield Ab

    x = backward_substitution(Ab)
    n = len(x)

    answer = np.zeros((n, n + 1))
    for i in range(n):
        answer[i, i] = 1
        answer[i, n] = x[i]
    yield answer


def gauss_gordan_generator(A, b, use_scaling):
    for Ab in forward_elimination_generator(A, b, use_scaling):
        yield Ab
    for Ab in backward_elimination(Ab):
        yield Ab
    yield normalize_rows(Ab)


def gauss(A, b, use_scaling):
    return list(gauss_generator(A, b, use_scaling))[-1][:, -1]


def gauss_gordon(A, b, use_scaling):
    return list(gauss_gordan_generator(A, b, use_scaling))[-1][:, -1]


# TESTs
a = np.array([
    [-4, 5, 8, -9],
    [4, 3, -7, 8],
    [6, 8, 7, 4],
    [1, -5, 8, 3]
], dtype=float)
b = np.array([10, 25, 6, 7], dtype=float)

Ab = np.column_stack((a, b))
solution = np.linalg.solve(a, b)
print(solution)
print(gauss(a, b, True))
print(gauss_gordon(a, b, True))
