/* Step-capped NLopt LD_LBFGS for one basin-hopping quench.

   The objective is optbench_lj_capped. A repeated coordinate does not
   count, matching HarnessCounter. The call budget is the remaining
   force-call cap. A refused call sets capped and is not counted.
*/

#include <math.h>
#include <nlopt.h>
#include <stdlib.h>
#include <string.h>

int optbench_lj_capped(const double *positions, int atoms, double core_radius, double cutoff,
                        double *energy_out, double *gradient, double *min_distance);

struct optbench_quench_out {
    double energy;
    double min_distance;
    int calls;
    int code;
    int capped;
};

struct eval_state {
    double *gradient;
    double *cache_x;
    double *cache_gradient;
    int dimension;
    int atoms;
    int *calls;
    int cap;
    int capped;
    int cache_valid;
    double core_radius;
    double cutoff;
    double cache_energy;
    double cache_min;
    nlopt_opt opt;
};

static double gmax_of(const double *gradient, int dimension) {
    double biggest = 0.0;
    int i;
    for (i = 0; i < dimension; i++) {
        double magnitude = fabs(gradient[i]);
        if (magnitude > biggest) {
            biggest = magnitude;
        }
    }
    return biggest;
}

static double objective(unsigned dimension, const double *x, double *gradient, void *data) {
    struct eval_state *state = data;
    double energy;
    double minimum;
    (void)dimension;
    if (state->cache_valid &&
        memcmp(state->cache_x, x, (size_t)state->dimension * sizeof(double)) == 0) {
        if (gradient) {
            memcpy(gradient, state->cache_gradient, (size_t)state->dimension * sizeof(double));
        }
        return state->cache_energy;
    }
    if (*state->calls >= state->cap) {
        state->capped = 1;
        if (gradient) {
            memset(gradient, 0, (size_t)state->dimension * sizeof(double));
        }
        if (state->opt) {
            nlopt_force_stop(state->opt);
        }
        return 1.0e300;
    }
    optbench_lj_capped(x, state->atoms, state->core_radius, state->cutoff, &energy, state->gradient,
                        &minimum);
    (*state->calls)++;
    memcpy(state->cache_x, x, (size_t)state->dimension * sizeof(double));
    memcpy(state->cache_gradient, state->gradient, (size_t)state->dimension * sizeof(double));
    state->cache_energy = energy;
    state->cache_min = minimum;
    state->cache_valid = 1;
    if (gradient) {
        memcpy(gradient, state->gradient, (size_t)state->dimension * sizeof(double));
    }
    return energy;
}

static void finish(struct optbench_quench_out *out, double energy, double minimum, int calls, int code,
                   int capped) {
    out->energy = energy;
    out->min_distance = minimum;
    out->calls = calls;
    out->code = code;
    out->capped = capped;
}

int optbench_lj_quench(double *x, int dimension, int cap, int local_max, int window, double maxstep,
                        double tolg, double core_radius, double cutoff, struct optbench_quench_out *out) {
    int atoms;
    int calls = 0;
    int code = 0;
    int stall = 0;
    double *best;
    double *lower;
    double *upper;
    double *gradient;
    double *cache_x;
    double *cache_gradient;
    double *working;
    double best_energy;
    double best_min;
    double previous;
    struct eval_state state;

    if (out == NULL || x == NULL || dimension < 3 || dimension % 3 != 0 || cap < 0 || local_max < 1 ||
        window < 1 || maxstep <= 0.0 || tolg <= 0.0 || core_radius <= 0.0 || cutoff <= 0.0) {
        return 1;
    }
    atoms = dimension / 3;
    best = calloc((size_t)dimension, sizeof(double));
    lower = calloc((size_t)dimension, sizeof(double));
    upper = calloc((size_t)dimension, sizeof(double));
    gradient = calloc((size_t)dimension, sizeof(double));
    cache_x = calloc((size_t)dimension, sizeof(double));
    cache_gradient = calloc((size_t)dimension, sizeof(double));
    working = calloc((size_t)dimension, sizeof(double));
    if (best == NULL || lower == NULL || upper == NULL || gradient == NULL || cache_x == NULL ||
        cache_gradient == NULL || working == NULL) {
        free(best);
        free(lower);
        free(upper);
        free(gradient);
        free(cache_x);
        free(cache_gradient);
        free(working);
        return 1;
    }
    memcpy(working, x, (size_t)dimension * sizeof(double));
    memset(&state, 0, sizeof(state));
    state.gradient = gradient;
    state.cache_x = cache_x;
    state.cache_gradient = cache_gradient;
    state.dimension = dimension;
    state.atoms = atoms;
    state.calls = &calls;
    state.cap = cap;
    state.core_radius = core_radius;
    state.cutoff = cutoff;
    objective((unsigned)dimension, working, gradient, &state);
    if (state.capped) {
        finish(out, 0.0, 0.0, calls, code, 1);
        goto cleanup;
    }
    best_energy = state.cache_energy;
    best_min = state.cache_min;
    previous = best_energy;
    memcpy(best, working, (size_t)dimension * sizeof(double));
    if (gmax_of(gradient, dimension) <= tolg) {
        code = 1;
        finish(out, best_energy, best_min, calls, code, 0);
        memcpy(x, best, (size_t)dimension * sizeof(double));
        goto cleanup;
    }
    while (calls < local_max && calls < cap && !state.capped) {
        int i;
        int remain;
        int before;
        int on_bound = 0;
        double step_min;
        nlopt_result result;
        for (i = 0; i < dimension; i++) {
            lower[i] = working[i] - maxstep;
            upper[i] = working[i] + maxstep;
        }
        state.opt = nlopt_create(NLOPT_LD_LBFGS, (unsigned)dimension);
        if (state.opt == NULL) {
            free(best);
            free(lower);
            free(upper);
            free(gradient);
            free(cache_x);
            free(cache_gradient);
            free(working);
            return 1;
        }
        nlopt_set_min_objective(state.opt, objective, &state);
        nlopt_set_lower_bounds(state.opt, lower);
        nlopt_set_upper_bounds(state.opt, upper);
        nlopt_set_xtol_rel(state.opt, 1e-14);
        nlopt_set_ftol_abs(state.opt, 1e-14);
        remain = local_max - calls;
        if (cap - calls < remain) {
            remain = cap - calls;
        }
        if (remain > window) {
            remain = window;
        }
        if (remain < 1) {
            remain = 1;
        }
        nlopt_set_maxeval(state.opt, remain);
        nlopt_set_param(state.opt, "tolg", tolg);
        before = calls;
        result = nlopt_optimize(state.opt, working, &step_min);
        code = (int)result;
        nlopt_destroy(state.opt);
        state.opt = NULL;
        (void)step_min;
        if (state.capped) {
            break;
        }
        if (result < 0) {
            break;
        }
        objective((unsigned)dimension, working, gradient, &state);
        if (state.capped) {
            break;
        }
        if (state.cache_energy < best_energy) {
            best_energy = state.cache_energy;
            best_min = state.cache_min;
            memcpy(best, working, (size_t)dimension * sizeof(double));
        }
        for (i = 0; i < dimension; i++) {
            if (working[i] <= lower[i] + 1e-8 || working[i] >= upper[i] - 1e-8) {
                on_bound = 1;
                break;
            }
        }
        if (calls == before) {
            break;
        }
        if (gmax_of(gradient, dimension) <= tolg && !on_bound) {
            break;
        }
        if (fabs(state.cache_energy - previous) < 1e-10 && !on_bound) {
            stall++;
            if (stall >= 3) {
                break;
            }
        } else {
            stall = 0;
        }
        previous = state.cache_energy;
    }
    memcpy(x, best, (size_t)dimension * sizeof(double));
    finish(out, best_energy, best_min, calls, code, state.capped);
cleanup:
    free(best);
    free(lower);
    free(upper);
    free(gradient);
    free(cache_x);
    free(cache_gradient);
    free(working);
    return 0;
}
