/* Unshifted binary Lennard-Jones with no cutoff. Species 0 is A, 1 is B.
 * Unlike sigma is the arithmetic mean. Same pairs as
 * optbench.potentials.binary_lennard_jones with cutoff None. The gradient
 * is dE/dx.
 */
#include <stddef.h>

void optbench_blj(int n, const double *x, const int *species, double sigma_aa, double sigma_bb,
                  double epsilon_aa, double epsilon_bb, double epsilon_ab, double *energy,
                  double *grad) {
    int i, j, k;
    double e = 0.0;
    for (k = 0; k < 3 * n; k++) grad[k] = 0.0;
    for (i = 0; i < n; i++) {
        double xi = x[3 * i];
        double yi = x[3 * i + 1];
        double zi = x[3 * i + 2];
        double si = species[i] == 0 ? sigma_aa : sigma_bb;
        for (j = i + 1; j < n; j++) {
            double dx = xi - x[3 * j];
            double dy = yi - x[3 * j + 1];
            double dz = zi - x[3 * j + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double sj, sigma, eps, inv, sr2, sr6, sr12, slope;
            if (!(r2 > 0.0)) continue;
            sj = species[j] == 0 ? sigma_aa : sigma_bb;
            sigma = 0.5 * (si + sj);
            if (species[i] != species[j]) {
                eps = epsilon_ab;
            } else {
                eps = species[i] == 0 ? epsilon_aa : epsilon_bb;
            }
            inv = 1.0 / r2;
            sr2 = sigma * sigma * inv;
            sr6 = sr2 * sr2 * sr2;
            sr12 = sr6 * sr6;
            e += 4.0 * eps * (sr12 - sr6);
            slope = 24.0 * eps * (sr6 - 2.0 * sr12) * inv;
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
