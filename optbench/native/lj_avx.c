/* AVX-512 cutoff Lennard-Jones gradient. Same potential as the scalar
   kernel: unshifted 12-6, no quadratic core. Pair terms are accumulated
   in a vector reduction, so the sum differs from the scalar order by
   about 1e-13 on an LJ98 geometry. OPTBENCH_SCALAR_LJ=1 selects the
   scalar kernel. A pair at distance zero makes the energy +infinity. */

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int optbench_lj_gradient_avx(const double *positions, int atoms, double cutoff, double *energy_out,
                             double *gradient, double *min_distance) {
    static __thread double *scratch;
    static __thread int scratch_atoms;
    double *xs;
    double *ys;
    double *zs;
    double *gx;
    double *gy;
    double *gz;
    double cutoff2;
    double energy;
    double min2;
    int singular;
    int i;
    int j;
    __m512d vcut2;
    __m512d vfour;
    __m512d v24;
    __m512d vtwo;
    __m512d vone;
    if (atoms < 2 || positions == NULL || gradient == NULL || energy_out == NULL || min_distance == NULL ||
        cutoff <= 0.0) {
        return 1;
    }
    if (scratch_atoms < atoms) {
        size_t bytes = (size_t)atoms * sizeof(double) * 6;
        double *grown;
        bytes = (bytes + 63u) & ~(size_t)63u;
        grown = aligned_alloc(64, bytes);
        if (grown == NULL) {
            return 1;
        }
        free(scratch);
        scratch = grown;
        scratch_atoms = atoms;
    }
    xs = scratch;
    ys = xs + atoms;
    zs = ys + atoms;
    gx = zs + atoms;
    gy = gx + atoms;
    gz = gy + atoms;
    cutoff2 = cutoff * cutoff;
    energy = 0.0;
    min2 = INFINITY;
    singular = 0;
    vcut2 = _mm512_set1_pd(cutoff2);
    vfour = _mm512_set1_pd(4.0);
    v24 = _mm512_set1_pd(24.0);
    vtwo = _mm512_set1_pd(2.0);
    vone = _mm512_set1_pd(1.0);
    for (i = 0; i < atoms; i++) {
        xs[i] = positions[3 * i];
        ys[i] = positions[3 * i + 1];
        zs[i] = positions[3 * i + 2];
        gx[i] = 0.0;
        gy[i] = 0.0;
        gz[i] = 0.0;
    }
    for (i = 0; i < atoms; i++) {
        double xi = xs[i];
        double yi = ys[i];
        double zi = zs[i];
        double gxi = 0.0;
        double gyi = 0.0;
        double gzi = 0.0;
        __m512d vxi = _mm512_set1_pd(xi);
        __m512d vyi = _mm512_set1_pd(yi);
        __m512d vzi = _mm512_set1_pd(zi);
        __m512d vgx = _mm512_setzero_pd();
        __m512d vgy = _mm512_setzero_pd();
        __m512d vgz = _mm512_setzero_pd();
        __m512d ve = _mm512_setzero_pd();
        j = i + 1;
        for (; j + 8 <= atoms; j += 8) {
            __m512d dx = _mm512_sub_pd(vxi, _mm512_loadu_pd(xs + j));
            __m512d dy = _mm512_sub_pd(vyi, _mm512_loadu_pd(ys + j));
            __m512d dz = _mm512_sub_pd(vzi, _mm512_loadu_pd(zs + j));
            __m512d r2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
            __mmask8 nonzero = _mm512_cmp_pd_mask(r2, _mm512_setzero_pd(), _CMP_GT_OQ);
            __mmask8 inside = _mm512_cmp_pd_mask(r2, vcut2, _CMP_LT_OQ);
            __mmask8 zero = _mm512_cmp_pd_mask(r2, _mm512_setzero_pd(), _CMP_EQ_OQ);
            __mmask8 physical = nonzero & inside;
            __m512d inv = _mm512_mask_div_pd(_mm512_setzero_pd(), physical, vone, r2);
            __m512d sr6 = _mm512_mul_pd(_mm512_mul_pd(inv, inv), inv);
            __m512d sr12 = _mm512_mul_pd(sr6, sr6);
            __m512d lj_energy = _mm512_mul_pd(vfour, _mm512_sub_pd(sr12, sr6));
            __m512d slope = _mm512_mul_pd(v24, _mm512_mul_pd(_mm512_sub_pd(sr6, _mm512_mul_pd(vtwo, sr12)), inv));
            __m512d sgx;
            __m512d sgy;
            __m512d sgz;
            double block_min;
            if (zero) {
                singular = 1;
            }
            lj_energy = _mm512_maskz_mov_pd(physical, lj_energy);
            slope = _mm512_maskz_mov_pd(physical, slope);
            ve = _mm512_add_pd(ve, lj_energy);
            sgx = _mm512_mul_pd(slope, dx);
            sgy = _mm512_mul_pd(slope, dy);
            sgz = _mm512_mul_pd(slope, dz);
            vgx = _mm512_add_pd(vgx, sgx);
            vgy = _mm512_add_pd(vgy, sgy);
            vgz = _mm512_add_pd(vgz, sgz);
            _mm512_storeu_pd(gx + j, _mm512_sub_pd(_mm512_loadu_pd(gx + j), sgx));
            _mm512_storeu_pd(gy + j, _mm512_sub_pd(_mm512_loadu_pd(gy + j), sgy));
            _mm512_storeu_pd(gz + j, _mm512_sub_pd(_mm512_loadu_pd(gz + j), sgz));
            block_min = _mm512_reduce_min_pd(r2);
            if (block_min < min2) {
                min2 = block_min;
            }
        }
        energy += _mm512_reduce_add_pd(ve);
        gxi += _mm512_reduce_add_pd(vgx);
        gyi += _mm512_reduce_add_pd(vgy);
        gzi += _mm512_reduce_add_pd(vgz);
        for (; j < atoms; j++) {
            double dx = xi - xs[j];
            double dy = yi - ys[j];
            double dz = zi - zs[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double inv;
            double sr6;
            double sr12;
            double slope;
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
            gxi += slope * dx;
            gyi += slope * dy;
            gzi += slope * dz;
            gx[j] -= slope * dx;
            gy[j] -= slope * dy;
            gz[j] -= slope * dz;
        }
        gx[i] += gxi;
        gy[i] += gyi;
        gz[i] += gzi;
    }
    if (singular) {
        energy = INFINITY;
        for (i = 0; i < atoms; i++) {
            gx[i] = 0.0;
            gy[i] = 0.0;
            gz[i] = 0.0;
        }
    }
    for (i = 0; i < atoms; i++) {
        gradient[3 * i] = gx[i];
        gradient[3 * i + 1] = gy[i];
        gradient[3 * i + 2] = gz[i];
    }
    *energy_out = energy;
    *min_distance = sqrt(min2);
    return 0;
}
