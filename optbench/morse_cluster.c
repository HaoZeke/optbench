/* r2549 client/potentials/Morse/Morse.cpp, Morse::force.
   Pair energy is De * ((1 - exp(-a (r - re)))^2 - 1), minus the same
   value at the cutoff, so the cutoff sphere is zero. The returned
   forces are -dE/dr along the bond. Pairs at or beyond the cutoff are
   skipped. A positive box diagonal applies the minimum image. */
#include <math.h>
#include <string.h>

void morse_eon_flat(
    int atoms,
    const double *pos,
    const double *box,
    double *energy_out,
    double *forces,
    double well,
    double alpha,
    double re,
    double cutoff
) {
    int count = atoms * 3;
    memset(forces, 0, (size_t)count * sizeof(double));
    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;
    if (box != 0) {
        bx = box[0];
        by = box[4];
        bz = box[8];
    }
    double dc = 1.0 - exp(-alpha * (cutoff - re));
    double ecut = well * (dc * dc - 1.0);
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
            if (bx > 0.0) dx -= bx * floor(dx / bx + 0.5);
            if (by > 0.0) dy -= by * floor(dy / by + 0.5);
            if (bz > 0.0) dz -= bz * floor(dz / bz + 0.5);
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0) continue;
            double r = sqrt(r2);
            if (r >= cutoff) continue;
            double gap = 1.0 - exp(-alpha * (r - re));
            energy += well * (gap * gap - 1.0) - ecut;
            double bond = 2.0 * well * gap * (gap - 1.0) * alpha;
            double scale = bond / r;
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
