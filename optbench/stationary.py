"""Hessian index, energy, and Kabsch RMSD for one stationary point.

A minimum is index 0. A saddle is index 1. Eigenvalues at or above
``-eigenvalue_tol`` are not negative, so a numerical zero mode is not
an extra imaginary frequency. The RMSD is the Kabsch RMSD after
translation and a proper rotation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Matrix = list[list[float]]
Point = Sequence[float]


def jacobi_eigenvalues(matrix: Sequence[Sequence[float]], sweeps: int = 80, tol: float = 1e-15):
    """Eigenvalues and column eigenvectors of a real symmetric matrix.

    The matrix is symmetrised on entry. Eigenvalues are ascending.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    current: Matrix = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            mean = 0.5 * (current[i][j] + current[j][i])
            current[i][j] = mean
            current[j][i] = mean
    vectors: Matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                element = current[p][q]
                off += element * element
                if abs(element) <= tol:
                    continue
                app = current[p][p]
                aqq = current[q][q]
                tau = (aqq - app) / (2.0 * element)
                tangent = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                for i in range(n):
                    if i == p or i == q:
                        continue
                    aip = current[i][p]
                    aiq = current[i][q]
                    current[i][p] = current[p][i] = cosine * aip - sine * aiq
                    current[i][q] = current[q][i] = sine * aip + cosine * aiq
                current[p][p] = cosine * cosine * app - 2.0 * sine * cosine * element + sine * sine * aqq
                current[q][q] = sine * sine * app + 2.0 * sine * cosine * element + cosine * cosine * aqq
                current[p][q] = current[q][p] = 0.0
                for i in range(n):
                    vip = vectors[i][p]
                    viq = vectors[i][q]
                    vectors[i][p] = cosine * vip - sine * viq
                    vectors[i][q] = sine * vip + cosine * viq
        if off <= tol * tol:
            break
    order = sorted(range(n), key=lambda index: current[index][index])
    values = [current[index][index] for index in order]
    ordered = [[vectors[row][index] for index in order] for row in range(n)]
    return values, ordered


def negative_index(eigenvalues: Sequence[float], tol: float = 1e-6) -> int:
    """Count eigenvalues below ``-tol``."""
    return sum(1 for value in eigenvalues if value < -tol)


def numerical_hessian(gradient, point: Sequence[float], step: float = 1e-5) -> Matrix:
    """Central-difference Hessian. The gradient is called ``2 * len(point)`` times."""
    size = len(point)
    hessian: Matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        plus = [float(value) for value in point]
        minus = [float(value) for value in point]
        plus[i] += step
        minus[i] -= step
        gp = gradient(plus)
        gm = gradient(minus)
        for j in range(size):
            hessian[j][i] = (float(gp[j]) - float(gm[j])) / (2.0 * step)
    for i in range(size):
        for j in range(i):
            mean = 0.5 * (hessian[i][j] + hessian[j][i])
            hessian[i][j] = mean
            hessian[j][i] = mean
    return hessian


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    cols = len(right[0])
    shared = len(right)
    return [
        [sum(left[i][k] * right[k][j] for k in range(shared)) for j in range(cols)]
        for i in range(rows)
    ]


def _transpose(matrix: Matrix) -> Matrix:
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def _matvec(matrix: Matrix, vector: Sequence[float]) -> list[float]:
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]


def _det3(matrix: Matrix) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _kabsch_rotation(moved: Sequence[Point], reference: Sequence[Point]) -> Matrix:
    """Proper rotation taking ``moved`` onto ``reference``. Both are centered."""
    covariance: Matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for left, right in zip(moved, reference, strict=True):
        for i in range(3):
            for j in range(3):
                covariance[i][j] += float(left[i]) * float(right[j])
    gram = _matmul(_transpose(covariance), covariance)
    _values, eigenvectors = jacobi_eigenvalues(gram)
    singular = [math.sqrt(max(gram_value, 0.0)) for gram_value in _values]
    columns: list[list[float] | None] = []
    for index in range(3):
        axis = [eigenvectors[row][index] for row in range(3)]
        image = _matvec(covariance, axis)
        if singular[index] > 1e-12:
            columns.append([value / singular[index] for value in image])
        else:
            columns.append(None)
    present = [column for column in columns if column is not None]
    if len(present) == 2:
        missing = columns.index(None)
        columns[missing] = _cross(present[0], present[1])
    elif len(present) == 1:
        seed = [1.0, 0.0, 0.0] if abs(present[0][0]) < 0.9 else [0.0, 1.0, 0.0]
        first = _cross(present[0], seed)
        scale = _norm(first) or 1.0
        first = [value / scale for value in first]
        second = _cross(present[0], first)
        blanks = [index for index, column in enumerate(columns) if column is None]
        columns[blanks[0]] = first
        columns[blanks[1]] = second
    elif len(present) == 0:
        columns = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    left_basis = [[columns[column][row] for column in range(3)] for row in range(3)]
    rotation = _matmul(eigenvectors, _transpose(left_basis))
    if _det3(rotation) < 0.0:
        smallest = min(range(3), key=lambda index: singular[index])
        eigenvectors = [row[:] for row in eigenvectors]
        for row in range(3):
            eigenvectors[row][smallest] *= -1.0
        rotation = _matmul(eigenvectors, _transpose(left_basis))
    return rotation


def kabsch_rmsd(moved: Sequence[Point], reference: Sequence[Point]) -> float:
    """RMSD after centering and a proper rotation.

    A single point is the Cartesian distance. Alignment would remove the
    only coordinate that point has.
    """
    if not moved or len(moved) != len(reference):
        raise ValueError("coordinate sets must have the same non-zero length")
    if any(len(point) != 3 for point in (*moved, *reference)):
        raise ValueError("each coordinate is x, y, z")
    count = len(moved)
    # One point has no internal geometry. Centering it would report a
    # distance of zero for every placement.
    if count == 1:
        return math.sqrt(sum((float(moved[0][axis]) - float(reference[0][axis])) ** 2 for axis in range(3)))
    left_center = [sum(float(point[axis]) for point in moved) / count for axis in range(3)]
    right_center = [sum(float(point[axis]) for point in reference) / count for axis in range(3)]
    left = [[float(point[axis]) - left_center[axis] for axis in range(3)] for point in moved]
    right = [[float(point[axis]) - right_center[axis] for axis in range(3)] for point in reference]
    rotation = _kabsch_rotation(left, right)
    error = 0.0
    for source, target in zip(left, right, strict=True):
        rotated = _matvec(rotation, source)
        for axis in range(3):
            delta = rotated[axis] - target[axis]
            error += delta * delta
    return math.sqrt(error / count)


def judge_stationary(
    *,
    kind: str,
    eigenvalues: Sequence[float],
    energy: float,
    coords: Sequence[Point],
    reference_energy: float,
    reference_coords: Sequence[Point],
    energy_tol: float,
    rmsd_tol: float,
    eigenvalue_tol: float = 1e-6,
) -> dict:
    """Accept a minimum at index 0 or a saddle at index 1, with energy and RMSD."""
    if kind == "saddle":
        expected = 1
    elif kind == "minimum":
        expected = 0
    else:
        raise ValueError("kind must be saddle or minimum")
    index = negative_index(eigenvalues, eigenvalue_tol)
    rmsd = kabsch_rmsd(coords, reference_coords)
    reasons: list[str] = []
    if index != expected:
        reasons.append("hessian_index")
    if abs(float(energy) - float(reference_energy)) > energy_tol:
        reasons.append("energy")
    if rmsd > rmsd_tol:
        reasons.append("rmsd")
    return {
        "ok": not reasons,
        "hessian_index": index,
        "expected_index": expected,
        "rmsd": rmsd,
        "energy_error": abs(float(energy) - float(reference_energy)),
        "reasons": reasons,
    }
