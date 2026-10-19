/* Pair kernels for the harness potential server.

   The arithmetic follows the Python loops in optbench.lj and
   benchmarks/bench_eon_develop.py, in the same order. Build with
   -ffp-contract=off so the C compiler does not fuse multiply-adds that
   CPython leaves separate.
*/

#include <math.h>
#include <stddef.h>

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

/* Quadratic core inside core_radius, matched in value and dV/dr to the
   unshifted 12-6 at the join. The gradient is dE/dx, not the force.
   Pairs at or beyond cutoff are omitted from the energy. The minimum
   pair distance still includes them. */
int optbench_lj_capped(const double *positions, int atoms, double core_radius,
                        double cutoff, double *energy_out, double *gradient,
                        double *min_distance) {
    int count = atoms * 3;
    int i;
    double core2;
    double cutoff2;
    double inv_core;
    double sr6_core;
    double sr12_core;
    double core_value;
    double core_derivative;
    double core_curvature;
    double core_slope;
    double energy = 0.0;
    double min2 = INFINITY;
    if (atoms < 0 || energy_out == NULL || min_distance == NULL || core_radius <= 0.0
        || cutoff <= 0.0 || (atoms > 0 && (positions == NULL || gradient == NULL))) {
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
    core2 = core_radius * core_radius;
    cutoff2 = cutoff * cutoff;
    inv_core = 1.0 / core2;
    sr6_core = inv_core * inv_core * inv_core;
    sr12_core = sr6_core * sr6_core;
    core_value = 4.0 * (sr12_core - sr6_core);
    core_derivative = 24.0 * (sr6_core - 2.0 * sr12_core) * inv_core * core_radius;
    core_curvature = core_derivative / (2.0 * core_radius);
    core_slope = core_derivative / core_radius;
    for (i = 0; i < atoms; i++) {
        int ix = 3 * i;
        int j;
        for (j = i + 1; j < atoms; j++) {
            int jx = 3 * j;
            double dx = positions[ix] - positions[jx];
            double dy = positions[ix + 1] - positions[jx + 1];
            double dz = positions[ix + 2] - positions[jx + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double slope;
            double gx;
            double gy;
            double gz;
            if (r2 < min2) {
                min2 = r2;
            }
            if (r2 < core2) {
                energy += core_value + core_curvature * (r2 - core2);
                slope = core_slope;
            } else if (r2 > 0.0 && r2 < cutoff2) {
                double inv = 1.0 / r2;
                double sr6 = inv * inv * inv;
                double sr12 = sr6 * sr6;
                energy += 4.0 * (sr12 - sr6);
                slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            } else {
                continue;
            }
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
    *energy_out = energy;
    *min_distance = sqrt(min2);
    return 0;
}
