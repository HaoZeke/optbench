/* Unshifted 12-6 Lennard-Jones. epsilon 1, sigma 1, cutoff 40, no images.
   Same formula as optbench.potentials.lennard_jones. */
void lj_cluster(const double *coords, int n, double *energy, double *grad) {
    const double cutoff2 = 1600.0;
    double e = 0.0;
    int i;
    for (i = 0; i < 3 * n; ++i) {
        grad[i] = 0.0;
    }
    for (i = 0; i < n; ++i) {
        const double ix = coords[3 * i];
        const double iy = coords[3 * i + 1];
        const double iz = coords[3 * i + 2];
        int j;
        for (j = i + 1; j < n; ++j) {
            const double dx = ix - coords[3 * j];
            const double dy = iy - coords[3 * j + 1];
            const double dz = iz - coords[3 * j + 2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            double inv, sr6, sr12, slope, fx, fy, fz;
            if (r2 <= 0.0 || r2 >= cutoff2) {
                continue;
            }
            inv = 1.0 / r2;
            sr6 = inv * inv * inv;
            sr12 = sr6 * sr6;
            e += 4.0 * (sr12 - sr6);
            slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            fx = slope * dx;
            fy = slope * dy;
            fz = slope * dz;
            grad[3 * i] += fx;
            grad[3 * i + 1] += fy;
            grad[3 * i + 2] += fz;
            grad[3 * j] -= fx;
            grad[3 * j + 1] -= fy;
            grad[3 * j + 2] -= fz;
        }
    }
    *energy = e;
}
