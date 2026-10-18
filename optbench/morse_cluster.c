/* Pair Morse, same arithmetic as benchmarks.bench_eon_develop.morse_cluster.
   Energy is D (1 - exp(-a (r - re)))^2. Forces are the negative gradient.
   No cutoff. Pair loop order matches the Python kernel. */
#include <math.h>
#include <string.h>

void morse_cluster_flat(
    int atoms,
    const double *pos,
    double *energy_out,
    double *forces,
    double well,
    double alpha,
    double re
) {
    int count = atoms * 3;
    memset(forces, 0, (size_t)count * sizeof(double));
    double energy = 0.0;
    for (int i = 0; i < atoms; i++) {
        int ix = 3 * i;
        double xi = pos[ix];
        double yi = pos[ix + 1];
        double zi = pos[ix + 2];
        for (int j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = xi - pos[jx];
            double dy = yi - pos[jx + 1];
            double dz = zi - pos[jx + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0) continue;
            double r = sqrt(r2);
            double e1 = exp(-alpha * (r - re));
            double gap = 1.0 - e1;
            energy += well * gap * gap;
            double dv_dr = 2.0 * well * alpha * (e1 - e1 * e1);
            double scale = -dv_dr / r;
            double fx = scale * dx;
            double fy = scale * dy;
            double fz = scale * dz;
            forces[ix] += fx;
            forces[ix + 1] += fy;
            forces[ix + 2] += fz;
            forces[jx] -= fx;
            forces[jx + 1] -= fy;
            forces[jx + 2] -= fz;
        }
    }
    *energy_out = energy;
}
