from __future__ import annotations

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised only when numba is missing.
    NUMBA_AVAILABLE = False
    njit = None
else:
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def fill_potential_banded(amx, rvs, sin_matrix, n, nx):
        center = 2 * n
        half = n // 2
        scale = 2.0 / n

        for ix in range(nx):
            base = ix * n
            for j0 in range(n - 1):
                val = 0.0
                for ip0 in range(half):
                    ip = ip0 + 1
                    right = ix + ip
                    if right >= nx:
                        right = nx - 1
                    left = ix - ip
                    if left < 0:
                        left = 0
                    val += sin_matrix[j0, ip0] * (rvs[right] - rvs[left])
                val *= scale

                if val != 0.0:
                    width = n - (j0 + 1)
                    upper_row = center + (j0 + 1)
                    lower_row = center - (j0 + 1)
                    for offset in range(width):
                        amx[upper_row, base + offset] = -val
                        amx[lower_row, base + (j0 + 1) + offset] = val


    @njit(cache=True)
    def fill_scattering_banded(S, b, n, nx, tcol):
        center = 2 * n

        for ix in range(nx):
            base = ix * n
            rho = 0.0
            for j in range(n):
                rho += b[base + j]

            for j in range(n):
                weight = 0.0
                if rho != 0.0:
                    weight = tcol * b[base + j] / rho
                for jp in range(n):
                    S[center + j - jp, base + jp] += weight
                S[center, base + j] -= tcol


    @njit(cache=True)
    def curcalc_density(B, rj, nx, n, coef):
        for ix in range(nx):
            total = 0.0
            for j in range(n):
                total += B[ix, j]
            rj[ix] = coef * total


    @njit(cache=True)
    def curcalc_current(B, rj, nx, n, cofj):
        nh = n // 2
        for ix in range(nx):
            rj[ix] = 0.0

        for i0 in range(1, nx - 2):
            total = 0.0
            for j in range(nh):
                weight = 2.0 * (j + 1) - n - 1.0
                total += weight * (3.0 * B[i0 + 1, j] - B[i0 + 2, j])
            for j in range(nh, n):
                weight = 2.0 * (j + 1) - n - 1.0
                total += weight * (-B[i0 - 1, j] + 3.0 * B[i0, j])
            rj[i0] = cofj * total

        rj[1] = 2.0 * rj[2] - rj[3]
        rj[0] = rj[1]
        rj[nx - 2] = 2.0 * rj[nx - 3] - rj[nx - 4]
        rj[nx - 1] = rj[nx - 2]

else:
    fill_potential_banded = None
    fill_scattering_banded = None
    curcalc_density = None
    curcalc_current = None
