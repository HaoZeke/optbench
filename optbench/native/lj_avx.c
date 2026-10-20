/* AVX-512 Lennard-Jones gradient. Same core and cutoff rules as the
   scalar kernel. Pair terms are accumulated in a vector reduction, so
   the sum differs from the scalar order by a few units in the last
   place. The harness tests allow that gap. */

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int optbench_lj_capped_avx(const double *positions, int atoms, double core_radius, double cutoff,
                           double *energy_out, double *gradient, double *min_distance) {
    static __thread double *scratch;
    static __thread int scratch_atoms;
    double *xs;
    double *ys;
    double *zs;
    double *gx;
    double *gy;
    double *gz;
    double core2;
    double cutoff2;
    double inv_core;
    double sr6_core;
    double sr12_core;
    double core_value;
    double core_derivative;
    double core_slope;
    double core_curvature;
    double energy;
    double min2;
    int i;
    int j;
    __m512d vcore2;
    __m512d vcut2;
    __m512d vcore_value;
    __m512d vcurv;
    __m512d vslope_core;
    __m512d vfour;
    __m512d v24;
    __m512d vtwo;
    __m512d vone;
    if (atoms < 2 || positions == NULL || gradient == NULL || energy_out == NULL || min_distance == NULL ||
        core_radius <= 0.0 || cutoff <= 0.0) {
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
    core2 = core_radius * core_radius;
    cutoff2 = cutoff * cutoff;
    inv_core = 1.0 / core2;
    sr6_core = inv_core * inv_core * inv_core;
    sr12_core = sr6_core * sr6_core;
    core_value = 4.0 * (sr12_core - sr6_core);
    core_derivative = 24.0 * (sr6_core - 2.0 * sr12_core) * inv_core * core_radius;
    core_slope = core_derivative / core_radius;
    core_curvature = core_derivative / (2.0 * core_radius);
    energy = 0.0;
    min2 = INFINITY;
    vcore2 = _mm512_set1_pd(core2);
    vcut2 = _mm512_set1_pd(cutoff2);
    vcore_value = _mm512_set1_pd(core_value);
    vcurv = _mm512_set1_pd(core_curvature);
    vslope_core = _mm512_set1_pd(core_slope);
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
            __mmask8 core = _mm512_cmp_pd_mask(r2, vcore2, _CMP_LT_OQ);
            __m512d inv = _mm512_mask_div_pd(_mm512_setzero_pd(), nonzero, vone, r2);
            __m512d sr6 = _mm512_mul_pd(_mm512_mul_pd(inv, inv), inv);
            __m512d sr12 = _mm512_mul_pd(sr6, sr6);
            __m512d lj_energy = _mm512_mul_pd(vfour, _mm512_sub_pd(sr12, sr6));
            __m512d lj_slope = _mm512_mul_pd(v24, _mm512_mul_pd(_mm512_sub_pd(sr6, _mm512_mul_pd(vtwo, sr12)), inv));
            __m512d core_energy = _mm512_fmadd_pd(vcurv, _mm512_sub_pd(r2, vcore2), vcore_value);
            __m512d pair_energy = _mm512_mask_blend_pd(core, lj_energy, core_energy);
            __m512d slope = _mm512_mask_blend_pd(core, lj_slope, vslope_core);
            __m512d sgx;
            __m512d sgy;
            __m512d sgz;
            double block_min;
            pair_energy = _mm512_maskz_mov_pd(inside, pair_energy);
            slope = _mm512_maskz_mov_pd(inside, slope);
            ve = _mm512_add_pd(ve, pair_energy);
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
            double slope;
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
    for (i = 0; i < atoms; i++) {
        gradient[3 * i] = gx[i];
        gradient[3 * i + 1] = gy[i];
        gradient[3 * i + 2] = gz[i];
    }
    *energy_out = energy;
    *min_distance = sqrt(min2);
    return 0;
}
