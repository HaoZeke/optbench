/* Stock NLopt LD_LBFGS for one basin-hopping quench.

   The objective is the cutoff 12-6. There are no coordinate bounds.
   A repeated coordinate does not count, matching HarnessCounter.
   The call budget is the remaining force-call cap. A refused call
   sets capped and is not counted.
*/

#include <math.h>
#include <nlopt.h>
#include <stdlib.h>
#include <string.h>

int optbench_lj_gradient(const double *positions, int atoms, double cutoff, double *energy_out,
                         double *gradient, double *min_distance);

struct optbench_quench_out {
    double energy;
    double min_distance;
    int calls;
    int code;
    int capped;
};

struct quench_buffers {
    int dimension;
    double *gradient;
    double *cache_x;
    double *cache_gradient;
    double *working;
};

static __thread struct quench_buffers reused;

static int ensure_buffers(struct quench_buffers *buffers, int dimension) {
    if (buffers->dimension == dimension && buffers->gradient != NULL) {
        return 0;
    }
    free(buffers->gradient);
    free(buffers->cache_x);
    free(buffers->cache_gradient);
    free(buffers->working);
    memset(buffers, 0, sizeof(*buffers));
    buffers->gradient = calloc((size_t)dimension, sizeof(double));
    buffers->cache_x = calloc((size_t)dimension, sizeof(double));
    buffers->cache_gradient = calloc((size_t)dimension, sizeof(double));
    buffers->working = calloc((size_t)dimension, sizeof(double));
    if (buffers->gradient == NULL || buffers->cache_x == NULL || buffers->cache_gradient == NULL ||
        buffers->working == NULL) {
        return 1;
    }
    buffers->dimension = dimension;
    return 0;
}

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
    if (state->cache_valid && memcmp(state->cache_x, x, (size_t)state->dimension * sizeof(double)) == 0) {
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
    optbench_lj_gradient(x, state->atoms, state->cutoff, &energy, state->gradient, &minimum);
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

int optbench_lj_quench(double *x, int dimension, int cap, int local_max, double tolg, double cutoff,
                       struct optbench_quench_out *out) {
    int atoms;
    int calls = 0;
    int code = 0;
    int remain;
    double *gradient;
    double *cache_x;
    double *cache_gradient;
    double *working;
    double step_min;
    nlopt_result result;
    struct eval_state state;

    if (out == NULL || x == NULL || dimension < 3 || dimension % 3 != 0 || cap < 0 || local_max < 1 ||
        tolg <= 0.0 || cutoff <= 0.0) {
        return 1;
    }
    atoms = dimension / 3;
    if (ensure_buffers(&reused, dimension) != 0) {
        return 1;
    }
    gradient = reused.gradient;
    cache_x = reused.cache_x;
    cache_gradient = reused.cache_gradient;
    working = reused.working;
    memcpy(working, x, (size_t)dimension * sizeof(double));
    memset(&state, 0, sizeof(state));
    state.gradient = gradient;
    state.cache_x = cache_x;
    state.cache_gradient = cache_gradient;
    state.dimension = dimension;
    state.atoms = atoms;
    state.calls = &calls;
    state.cap = cap;
    state.cutoff = cutoff;
    objective((unsigned)dimension, working, gradient, &state);
    if (state.capped) {
        finish(out, 0.0, 0.0, calls, code, 1);
        return 0;
    }
    if (gmax_of(gradient, dimension) <= tolg) {
        finish(out, state.cache_energy, state.cache_min, calls, 1, 0);
        memcpy(x, working, (size_t)dimension * sizeof(double));
        return 0;
    }
    remain = local_max - calls;
    if (cap - calls < remain) {
        remain = cap - calls;
    }
    if (remain < 1) {
        finish(out, state.cache_energy, state.cache_min, calls, 0, 0);
        memcpy(x, working, (size_t)dimension * sizeof(double));
        return 0;
    }
    state.opt = nlopt_create(NLOPT_LD_LBFGS, (unsigned)dimension);
    if (state.opt == NULL) {
        return 1;
    }
    nlopt_set_min_objective(state.opt, objective, &state);
    nlopt_set_xtol_rel(state.opt, 1e-14);
    nlopt_set_ftol_abs(state.opt, 1e-14);
    nlopt_set_maxeval(state.opt, remain);
    nlopt_set_param(state.opt, "tolg", tolg);
    result = nlopt_optimize(state.opt, working, &step_min);
    code = (int)result;
    nlopt_destroy(state.opt);
    state.opt = NULL;
    (void)step_min;
    if (!state.capped) {
        objective((unsigned)dimension, working, gradient, &state);
    }
    memcpy(x, working, (size_t)dimension * sizeof(double));
    finish(out, state.cache_energy, state.cache_min, calls, code, state.capped);
    return 0;
}
