/* Pair kernels for the harness potential server.

   The arithmetic follows the Python loops in optbench.lj and
   benchmarks/bench_eon_develop.py, in the same order. Build with
   -ffp-contract=off so the C compiler does not fuse multiply-adds that
   CPython leaves separate.
*/

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int optbench_lj_gradient_avx(const double *positions, int atoms, double cutoff, double *energy_out,
                             double *gradient, double *min_distance);
int optbench_lj_gradient_scalar(const double *positions, int atoms, double cutoff, double *energy_out,
                                double *gradient, double *min_distance);

static double morse_fold(double delta, double length) {
    double scaled = delta / length;
    double raw = scaled + copysign(0.5, scaled);
    return delta - length * (double)(long)raw;
}

int optbench_morse(const double *positions, int atoms, const double *lengths,
                    double well, double width, double equilibrium, double cutoff,
                    double shift, double *energy_out, double *forces) {
    int count = atoms * 3;
    int i;
    double cutoff2;
    double two_de_a;
    double energy = 0.0;
    if (atoms < 0 || energy_out == NULL || (atoms > 0 && (positions == NULL || forces == NULL))) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        forces[i] = 0.0;
    }
    if (atoms < 2) {
        *energy_out = 0.0;
        return 0;
    }
    cutoff2 = cutoff * cutoff;
    two_de_a = 2.0 * well * width;
    for (i = 0; i < atoms; i++) {
        int ix = 3 * i;
        int j;
        for (j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = positions[ix] - positions[jx];
            double dy = positions[ix + 1] - positions[jx + 1];
            double dz = positions[ix + 2] - positions[jx + 2];
            double r2;
            double r;
            double gap;
            double scale;
            double fx;
            double fy;
            double fz;
            if (lengths != NULL) {
                dx = morse_fold(dx, lengths[0]);
                dy = morse_fold(dy, lengths[1]);
                dz = morse_fold(dz, lengths[2]);
            }
            r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0 || r2 > cutoff2) {
                continue;
            }
            r = sqrt(r2);
            gap = 1.0 - exp(-width * (r - equilibrium));
            energy += well * gap * gap - well - shift;
            scale = two_de_a * gap * (gap - 1.0) / r;
            fx = scale * dx;
            fy = scale * dy;
            fz = scale * dz;
            forces[ix] += fx;
            forces[ix + 1] += fy;
            forces[ix + 2] += fz;
            forces[jx] -= fx;
            forces[jx + 1] -= fy;
            forces[jx + 2] -= fz;
        }
    }
    *energy_out = energy;
    return 0;
}

int optbench_lj_cluster(const double *positions, int atoms, double u0, double psi,
                         double cutoff, double cutoff_u, double *energy_out, double *forces) {
    int count = atoms * 3;
    int i;
    double cutoff2;
    double psi2;
    double energy = 0.0;
    if (atoms < 0 || energy_out == NULL || (atoms > 0 && (positions == NULL || forces == NULL))) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        forces[i] = 0.0;
    }
    if (atoms < 2) {
        *energy_out = 0.0;
        return 0;
    }
    cutoff2 = cutoff * cutoff;
    psi2 = psi * psi;
    for (i = 0; i < atoms; i++) {
        int ix = 3 * i;
        int j;
        for (j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = positions[ix] - positions[jx];
            double dy = positions[ix + 1] - positions[jx + 1];
            double dz = positions[ix + 2] - positions[jx + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_r2;
            double sr2;
            double attraction;
            double well;
            double scale;
            double fx;
            double fy;
            double fz;
            if (r2 <= 0.0 || r2 >= cutoff2) {
                continue;
            }
            inv_r2 = 1.0 / r2;
            sr2 = psi2 * inv_r2;
            attraction = sr2 * sr2 * sr2;
            well = 4.0 * u0 * attraction;
            energy += well * (attraction - 1.0) - cutoff_u;
            scale = 6.0 * well * inv_r2 * (2.0 * attraction - 1.0);
            fx = scale * dx;
            fy = scale * dy;
            fz = scale * dz;
            forces[ix] += fx;
            forces[ix + 1] += fy;
            forces[ix + 2] += fz;
            forces[jx] -= fx;
            forces[jx + 1] -= fy;
            forces[jx + 2] -= fz;
        }
    }
    *energy_out = energy;
    return 0;
}

int optbench_lj_cut(const double *positions, int atoms, double epsilon, double sigma,
                     double cutoff, double *energy_out, double *forces) {
    int count = atoms * 3;
    int i;
    double cutoff2;
    double sigma2;
    double energy = 0.0;
    if (atoms < 0 || energy_out == NULL || (atoms > 0 && (positions == NULL || forces == NULL))) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        forces[i] = 0.0;
    }
    if (atoms < 2) {
        *energy_out = 0.0;
        return 0;
    }
    cutoff2 = cutoff * cutoff;
    sigma2 = sigma * sigma;
    for (i = 0; i < atoms; i++) {
        int ix = 3 * i;
        int j;
        for (j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = positions[ix] - positions[jx];
            double dy = positions[ix + 1] - positions[jx + 1];
            double dz = positions[ix + 2] - positions[jx + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_r2;
            double sr2;
            double attraction;
            double well;
            double scale;
            double fx;
            double fy;
            double fz;
            if (r2 <= 0.0 || r2 >= cutoff2) {
                continue;
            }
            inv_r2 = 1.0 / r2;
            sr2 = sigma2 * inv_r2;
            attraction = sr2 * sr2 * sr2;
            well = 4.0 * epsilon * attraction;
            energy += well * (attraction - 1.0);
            scale = 6.0 * well * inv_r2 * (2.0 * attraction - 1.0);
            fx = scale * dx;
            fy = scale * dy;
            fz = scale * dz;
            forces[ix] += fx;
            forces[ix + 1] += fy;
            forces[ix + 2] += fz;
            forces[jx] -= fx;
            forces[jx + 1] -= fy;
            forces[jx + 2] -= fz;
        }
    }
    *energy_out = energy;
    return 0;
}

/* Unshifted cutoff 12-6. The gradient is dE/dx, not the force. A pair at
   distance zero makes the energy +infinity and clears the gradient: the
   direction is undefined. Pairs at or beyond cutoff are omitted from the
   energy. The minimum pair distance still includes them. */
int optbench_lj_gradient_scalar(const double *positions, int atoms, double cutoff, double *energy_out,
                                double *gradient, double *min_distance) {
    int count = atoms * 3;
    int i;
    int singular = 0;
    double cutoff2;
    double energy = 0.0;
    double min2 = INFINITY;
    if (atoms < 0 || energy_out == NULL || min_distance == NULL || cutoff <= 0.0
        || (atoms > 0 && (positions == NULL || gradient == NULL))) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        gradient[i] = 0.0;
    }
    if (atoms < 2) {
        *energy_out = 0.0;
        *min_distance = INFINITY;
        return 0;
    }
    cutoff2 = cutoff * cutoff;
    for (i = 0; i < atoms; i++) {
        int ix = 3 * i;
        int j;
        for (j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = positions[ix] - positions[jx];
            double dy = positions[ix + 1] - positions[jx + 1];
            double dz = positions[ix + 2] - positions[jx + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv;
            double sr6;
            double sr12;
            double slope;
            double gx;
            double gy;
            double gz;
            if (r2 < min2) {
                min2 = r2;
            }
            if (r2 == 0.0) {
                singular = 1;
                continue;
            }
            if (r2 >= cutoff2) {
                continue;
            }
            inv = 1.0 / r2;
            sr6 = inv * inv * inv;
            sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            gx = slope * dx;
            gy = slope * dy;
            gz = slope * dz;
            gradient[ix] += gx;
            gradient[ix + 1] += gy;
            gradient[ix + 2] += gz;
            gradient[jx] -= gx;
            gradient[jx + 1] -= gy;
            gradient[jx + 2] -= gz;
        }
    }
    if (singular) {
        for (i = 0; i < count; i++) {
            gradient[i] = 0.0;
        }
        energy = INFINITY;
    }
    *energy_out = energy;
    *min_distance = sqrt(min2);
    return 0;
}

int optbench_lj_gradient(const double *positions, int atoms, double cutoff, double *energy_out,
                         double *gradient, double *min_distance) {
    static int avx = -1;
    if (avx < 0) {
        const char *scalar = getenv("OPTBENCH_SCALAR_LJ");
        int forced = scalar != NULL && scalar[0] == '1' && scalar[1] == '\0';
        avx = !forced && __builtin_cpu_supports("avx512f");
    }
    if (avx && optbench_lj_gradient_avx(positions, atoms, cutoff, energy_out, gradient, min_distance) == 0) {
        return 0;
    }
    return optbench_lj_gradient_scalar(positions, atoms, cutoff, energy_out, gradient, min_distance);
}
