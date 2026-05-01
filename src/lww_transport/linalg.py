from __future__ import annotations

import numpy as np


def band_to_agb(a: np.ndarray, n: int, nx: int) -> np.ndarray:
    """Convert a dense matrix to general band storage."""
    m = n * nx
    ml = 2 * n
    mu = 2 * n
    agb = np.zeros((ml + mu + 1, m), dtype=float)
    for j in range(1, m + 1):
        k = mu + 1 - j
        for i in range(max(1, j - mu), min(m, j + ml) + 1):
            agb[k + i - 1, j - 1] = a[i - 1, j - 1]
    return agb


def agb_to_dense(agb: np.ndarray, n: int, nx: int) -> np.ndarray:
    """Convert general band storage to a dense matrix."""
    m = n * nx
    ml = 2 * n
    mu = 2 * n
    a = np.zeros((m, m), dtype=float)
    for j in range(1, m + 1):
        k = mu + 1 - j
        for i in range(max(1, j - mu), min(m, j + ml) + 1):
            a[i - 1, j - 1] = agb[k + i - 1, j - 1]
    return a


def tridag(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    Raises an explicit error if a zero pivot is encountered.
    """
    nx = len(r)
    u = np.zeros(nx, dtype=float)
    gam = np.zeros(nx, dtype=float)
    bet = b[0]
    if bet == 0:
        raise ZeroDivisionError("tridag failed: zero first pivot")
    u[0] = r[0] / bet
    for j in range(2, nx + 1):
        gam[j - 1] = c[j - 2] / bet
        bet = b[j - 1] - a[j - 1] * gam[j - 1]
        if bet == 0:
            raise ZeroDivisionError(f"tridag failed: zero pivot at row {j}")
        u[j - 1] = (r[j - 1] - a[j - 1] * u[j - 2]) / bet
    for j in range(nx - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u
