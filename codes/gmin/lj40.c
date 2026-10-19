/* Unshifted Lennard-Jones, epsilon = 1, sigma = 1, cutoff = 40.
 * Same pairs as optbench.potentials.lennard_jones. The gradient is dE/dx.
 */
#include <stddef.h>

void optbench_lj40(int n, const double *x, double *energy, double *grad) {
    const double cut2 = 1600.0;
    int i, j, k;
    double e = 0.0;
    for (k = 0; k < 3 * n; k++) grad[k] = 0.0;
    for (i = 0; i < n; i++) {
        double xi = x[3 * i];
        double yi = x[3 * i + 1];
        double zi = x[3 * i + 2];
        for (j = i + 1; j < n; j++) {
            double dx = xi - x[3 * j];
            double dy = yi - x[3 * j + 1];
            double dz = zi - x[3 * j + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv, sr2, sr6, sr12, slope;
            if (!(r2 > 0.0 && r2 < cut2)) continue;
            inv = 1.0 / r2;
            sr2 = inv;
            sr6 = sr2 * sr2 * sr2;
            sr12 = sr6 * sr6;
            e += 4.0 * (sr12 - sr6);
            slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            grad[3 * i] += slope * dx;
            grad[3 * i + 1] += slope * dy;
            grad[3 * i + 2] += slope * dz;
            grad[3 * j] -= slope * dx;
            grad[3 * j + 1] -= slope * dy;
            grad[3 * j + 2] -= slope * dz;
        }
    }
    *energy = e;
}
