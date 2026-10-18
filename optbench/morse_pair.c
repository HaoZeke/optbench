/* LAMMPS morse/smooth/linear pair sum.

   Energy and dE/dx follow optbench.surfaces: the Morse value minus its
   value at the cutoff, minus (r - rc) times the Morse derivative at the
   cutoff. Pairs are i < j. A periodic axis uses round-half-to-even.
*/
#include <math.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
__m256d _ZGVdN4v_exp(__m256d);
#endif

static double round_half_even(double value) {
    return nearbyint(value);
}

static double morse_pair_scalar(
    const double *positions,
    double *gradient,
    int count,
    double depth,
    double alpha,
    double r0,
    double cutoff,
    double cell_x,
    double cell_y,
    double cell_z,
    int periodic_x,
    int periodic_y,
    int periodic_z
) {
    const double cutoff2 = cutoff * cutoff;
    const double at_cut = exp(-alpha * (cutoff - r0));
    const double offset = depth * (at_cut * at_cut - 2.0 * at_cut);
    const double der_cut = -2.0 * alpha * depth * (at_cut * at_cut - at_cut);
    double energy = 0.0;

    memset(gradient, 0, (size_t)count * 3u * sizeof(double));
    for (int i = 0; i < count; i++) {
        const double ix = positions[3 * i];
        const double iy = positions[3 * i + 1];
        const double iz = positions[3 * i + 2];
        for (int j = i + 1; j < count; j++) {
            double dx = ix - positions[3 * j];
            double dy = iy - positions[3 * j + 1];
            double dz = iz - positions[3 * j + 2];
            if (periodic_x && cell_x > 0.0) {
                dx -= cell_x * round_half_even(dx / cell_x);
            }
            if (periodic_y && cell_y > 0.0) {
                dy -= cell_y * round_half_even(dy / cell_y);
            }
            if (periodic_z && cell_z > 0.0) {
                dz -= cell_z * round_half_even(dz / cell_z);
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0 && r2 < cutoff2)) {
                continue;
            }
            const double radius = sqrt(r2);
            const double decay = exp(-alpha * (radius - r0));
            const double morse = depth * (decay * decay - 2.0 * decay);
            energy += morse - offset - (radius - cutoff) * der_cut;
            const double slope = -2.0 * alpha * depth * (decay * decay - decay) - der_cut;
            const double scale = slope / radius;
            const double gx = scale * dx;
            const double gy = scale * dy;
            const double gz = scale * dz;
            gradient[3 * i] += gx;
            gradient[3 * i + 1] += gy;
            gradient[3 * i + 2] += gz;
            gradient[3 * j] -= gx;
            gradient[3 * j + 1] -= gy;
            gradient[3 * j + 2] -= gz;
        }
    }
    return energy;
}

#if defined(__AVX2__)
static double morse_pair_avx(
    const double *positions,
    double *gradient,
    int count,
    double depth,
    double alpha,
    double r0,
    double cutoff,
    double cell_x,
    double cell_y,
    double cell_z,
    int periodic_x,
    int periodic_y,
    int periodic_z
) {
    const double cutoff2 = cutoff * cutoff;
    const double at_cut = exp(-alpha * (cutoff - r0));
    const double offset = depth * (at_cut * at_cut - 2.0 * at_cut);
    const double der_cut = -2.0 * alpha * depth * (at_cut * at_cut - at_cut);
    const __m256d depth_v = _mm256_set1_pd(depth);
    const __m256d alpha_v = _mm256_set1_pd(-alpha);
    const __m256d r0_v = _mm256_set1_pd(r0);
    const __m256d cutoff_v = _mm256_set1_pd(cutoff);
    const __m256d cutoff2_v = _mm256_set1_pd(cutoff2);
    const __m256d offset_v = _mm256_set1_pd(offset);
    const __m256d der_v = _mm256_set1_pd(der_cut);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d cellx = _mm256_set1_pd(cell_x);
    const __m256d celly = _mm256_set1_pd(cell_y);
    const __m256d cellz = _mm256_set1_pd(cell_z);
    double energy = 0.0;

    memset(gradient, 0, (size_t)count * 3u * sizeof(double));
    for (int i = 0; i < count; i++) {
        const double ix = positions[3 * i];
        const double iy = positions[3 * i + 1];
        const double iz = positions[3 * i + 2];
        const __m256d ixv = _mm256_set1_pd(ix);
        const __m256d iyv = _mm256_set1_pd(iy);
        const __m256d izv = _mm256_set1_pd(iz);
        int j = i + 1;
        for (; j + 3 < count; j += 4) {
            const double *pj = positions + 3 * j;
            __m256d dx = _mm256_sub_pd(ixv, _mm256_set_pd(pj[9], pj[6], pj[3], pj[0]));
            __m256d dy = _mm256_sub_pd(iyv, _mm256_set_pd(pj[10], pj[7], pj[4], pj[1]));
            __m256d dz = _mm256_sub_pd(izv, _mm256_set_pd(pj[11], pj[8], pj[5], pj[2]));
            if (periodic_x && cell_x > 0.0) {
                __m256d image = _mm256_round_pd(
                    _mm256_div_pd(dx, cellx), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
                );
                dx = _mm256_fnmadd_pd(cellx, image, dx);
            }
            if (periodic_y && cell_y > 0.0) {
                __m256d image = _mm256_round_pd(
                    _mm256_div_pd(dy, celly), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
                );
                dy = _mm256_fnmadd_pd(celly, image, dy);
            }
            if (periodic_z && cell_z > 0.0) {
                __m256d image = _mm256_round_pd(
                    _mm256_div_pd(dz, cellz), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
                );
                dz = _mm256_fnmadd_pd(cellz, image, dz);
            }
            const __m256d r2 = _mm256_fmadd_pd(
                dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))
            );
            const __m256d live = _mm256_and_pd(
                _mm256_cmp_pd(r2, zero, _CMP_GT_OQ),
                _mm256_cmp_pd(r2, cutoff2_v, _CMP_LT_OQ)
            );
            if (_mm256_movemask_pd(live) == 0) {
                continue;
            }
            const __m256d radius = _mm256_sqrt_pd(r2);
            const __m256d safe_r = _mm256_blendv_pd(one, radius, live);
            const __m256d decay = _ZGVdN4v_exp(_mm256_mul_pd(alpha_v, _mm256_sub_pd(radius, r0_v)));
            const __m256d decay2 = _mm256_mul_pd(decay, decay);
            const __m256d morse = _mm256_mul_pd(depth_v, _mm256_sub_pd(decay2, _mm256_mul_pd(two, decay)));
            const __m256d pair = _mm256_sub_pd(
                _mm256_sub_pd(morse, offset_v),
                _mm256_mul_pd(_mm256_sub_pd(radius, cutoff_v), der_v)
            );
            const __m256d kept = _mm256_and_pd(pair, live);
            const __m128d low = _mm256_castpd256_pd128(kept);
            const __m128d high = _mm256_extractf128_pd(kept, 1);
            const __m128d sum = _mm_add_pd(low, high);
            energy += _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));
            const __m256d slope = _mm256_sub_pd(
                _mm256_mul_pd(
                    _mm256_set1_pd(-2.0 * alpha * depth),
                    _mm256_sub_pd(decay2, decay)
                ),
                der_v
            );
            const __m256d scale = _mm256_and_pd(_mm256_div_pd(slope, safe_r), live);
            const __m256d gx = _mm256_mul_pd(scale, dx);
            const __m256d gy = _mm256_mul_pd(scale, dy);
            const __m256d gz = _mm256_mul_pd(scale, dz);
            double sx[4], sy[4], sz[4];
            _mm256_storeu_pd(sx, gx);
            _mm256_storeu_pd(sy, gy);
            _mm256_storeu_pd(sz, gz);
            gradient[3 * i] += sx[0] + sx[1] + sx[2] + sx[3];
            gradient[3 * i + 1] += sy[0] + sy[1] + sy[2] + sy[3];
            gradient[3 * i + 2] += sz[0] + sz[1] + sz[2] + sz[3];
            gradient[3 * j] -= sx[0];
            gradient[3 * j + 1] -= sy[0];
            gradient[3 * j + 2] -= sz[0];
            gradient[3 * (j + 1)] -= sx[1];
            gradient[3 * (j + 1) + 1] -= sy[1];
            gradient[3 * (j + 1) + 2] -= sz[1];
            gradient[3 * (j + 2)] -= sx[2];
            gradient[3 * (j + 2) + 1] -= sy[2];
            gradient[3 * (j + 2) + 2] -= sz[2];
            gradient[3 * (j + 3)] -= sx[3];
            gradient[3 * (j + 3) + 1] -= sy[3];
            gradient[3 * (j + 3) + 2] -= sz[3];
        }
        for (; j < count; j++) {
            double dx = ix - positions[3 * j];
            double dy = iy - positions[3 * j + 1];
            double dz = iz - positions[3 * j + 2];
            if (periodic_x && cell_x > 0.0) {
                dx -= cell_x * round_half_even(dx / cell_x);
            }
            if (periodic_y && cell_y > 0.0) {
                dy -= cell_y * round_half_even(dy / cell_y);
            }
            if (periodic_z && cell_z > 0.0) {
                dz -= cell_z * round_half_even(dz / cell_z);
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0 && r2 < cutoff2)) {
                continue;
            }
            const double radius = sqrt(r2);
            const double decay = exp(-alpha * (radius - r0));
            const double morse = depth * (decay * decay - 2.0 * decay);
            energy += morse - offset - (radius - cutoff) * der_cut;
            const double slope = -2.0 * alpha * depth * (decay * decay - decay) - der_cut;
            const double scale = slope / radius;
            gradient[3 * i] += scale * dx;
            gradient[3 * i + 1] += scale * dy;
            gradient[3 * i + 2] += scale * dz;
            gradient[3 * j] -= scale * dx;
            gradient[3 * j + 1] -= scale * dy;
            gradient[3 * j + 2] -= scale * dz;
        }
    }
    return energy;
}
#endif

double morse_pair_energy(
    const double *positions,
    double *gradient,
    int count,
    double depth,
    double alpha,
    double r0,
    double cutoff,
    double cell_x,
    double cell_y,
    double cell_z,
    int periodic_x,
    int periodic_y,
    int periodic_z
) {
#if defined(__AVX2__)
    return morse_pair_avx(
        positions, gradient, count, depth, alpha, r0, cutoff,
        cell_x, cell_y, cell_z, periodic_x, periodic_y, periodic_z
    );
#else
    return morse_pair_scalar(
        positions, gradient, count, depth, alpha, r0, cutoff,
        cell_x, cell_y, cell_z, periodic_x, periodic_y, periodic_z
    );
#endif
}
