/* NLopt LD_LBFGS quench with eOn's per-step move bound.

   eOn scales a step whose largest atomic displacement exceeds max_move
   (HelperFunctions.cpp maxAtomMotionAppliedV). NLopt 2.11 LD_LBFGS is
   Luksan PLIS. Its xmax is a Euclidean norm of the whole step, hardcoded
   to 0, which the routine replaces with 1e16, and no public call scales
   one trial. The equivalent is one L-BFGS solve inside a coordinate
   window of half-width max_move centered on the current point, then the
   eOn scale of the accepted displacement, then a new window. The scale
   uses a strict greater-than, as eOn does.

   The objective is the cutoff 12-6. A repeated coordinate does not
   count, matching HarnessCounter. The call budget is the remaining
   force-call cap. A refused call sets capped and is not counted.
   Stop when the Euclidean force norm is strictly below converged_force,
   which is eOn's default convergence metric.
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
    double *anchor;
    double *trial;
    double *last;
    double *scaled;
    double *lower;
    double *upper;
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
    free(buffers->anchor);
    free(buffers->trial);
    free(buffers->last);
    free(buffers->scaled);
    free(buffers->lower);
    free(buffers->upper);
    memset(buffers, 0, sizeof(*buffers));
    buffers->gradient = calloc((size_t)dimension, sizeof(double));
    buffers->cache_x = calloc((size_t)dimension, sizeof(double));
    buffers->cache_gradient = calloc((size_t)dimension, sizeof(double));
    buffers->working = calloc((size_t)dimension, sizeof(double));
    buffers->anchor = calloc((size_t)dimension, sizeof(double));
    buffers->trial = calloc((size_t)dimension, sizeof(double));
    buffers->last = calloc((size_t)dimension, sizeof(double));
    buffers->scaled = calloc((size_t)dimension, sizeof(double));
    buffers->lower = calloc((size_t)dimension, sizeof(double));
    buffers->upper = calloc((size_t)dimension, sizeof(double));
    if (buffers->gradient == NULL || buffers->cache_x == NULL || buffers->cache_gradient == NULL ||
        buffers->working == NULL || buffers->anchor == NULL || buffers->trial == NULL ||
        buffers->last == NULL || buffers->scaled == NULL || buffers->lower == NULL ||
        buffers->upper == NULL) {
        return 1;
    }
    buffers->dimension = dimension;
    return 0;
}

struct eval_state {
    double *gradient;
    double *cache_x;
    double *cache_gradient;
    double *last;
    int dimension;
    int atoms;
    int *calls;
    int cap;
    int local_max;
    int capped;
    int hit_local;
    int cache_valid;
    int converged;
    double cutoff;
    double converged_force;
    double cache_energy;
    double cache_min;
    nlopt_opt opt;
};

static double force_norm(const double *gradient, int dimension) {
    double total = 0.0;
    int i;
    for (i = 0; i < dimension; i++) {
        total += gradient[i] * gradient[i];
    }
    return sqrt(total);
}

static int same_bytes(const double *left, const double *right, int dimension) {
    return memcmp(left, right, (size_t)dimension * sizeof(double)) == 0;
}

static void scale_atomic_step(const double *anchor, const double *point, double *dest, int dimension,
                              double max_move) {
    double motion = 0.0;
    double factor;
    int i;
    for (i = 0; i < dimension; i += 3) {
        double dx = point[i] - anchor[i];
        double dy = point[i + 1] - anchor[i + 1];
        double dz = point[i + 2] - anchor[i + 2];
        double length = sqrt(dx * dx + dy * dy + dz * dz);
        if (length > motion) {
            motion = length;
        }
    }
    factor = motion > max_move ? max_move / motion : 1.0;
    for (i = 0; i < dimension; i++) {
        dest[i] = anchor[i] + factor * (point[i] - anchor[i]);
    }
}

static int evaluate(struct eval_state *state, const double *x) {
    double energy;
    double minimum;
    if (state->cache_valid && same_bytes(state->cache_x, x, state->dimension)) {
        memcpy(state->last, x, (size_t)state->dimension * sizeof(double));
        state->converged = isfinite(state->cache_energy) &&
                           force_norm(state->cache_gradient, state->dimension) < state->converged_force;
        return 0;
    }
    if (*state->calls >= state->cap) {
        state->capped = 1;
        return 2;
    }
    if (*state->calls >= state->local_max) {
        state->hit_local = 1;
        return 3;
    }
    optbench_lj_gradient(x, state->atoms, state->cutoff, &energy, state->gradient, &minimum);
    (*state->calls)++;
    memcpy(state->cache_x, x, (size_t)state->dimension * sizeof(double));
    memcpy(state->cache_gradient, state->gradient, (size_t)state->dimension * sizeof(double));
    memcpy(state->last, x, (size_t)state->dimension * sizeof(double));
    state->cache_energy = energy;
    state->cache_min = minimum;
    state->cache_valid = 1;
    state->converged = isfinite(energy) && force_norm(state->gradient, state->dimension) < state->converged_force;
    return 0;
}

static double objective(unsigned dimension, const double *x, double *gradient, void *data) {
    struct eval_state *state = data;
    int status;
    (void)dimension;
    status = evaluate(state, x);
    if (status != 0) {
        if (gradient) {
            memset(gradient, 0, (size_t)state->dimension * sizeof(double));
        }
        if (state->opt) {
            nlopt_force_stop(state->opt);
        }
        return 1.0e300;
    }
    if (gradient) {
        memcpy(gradient, state->cache_gradient, (size_t)state->dimension * sizeof(double));
    }
    if (state->opt && state->converged) {
        nlopt_force_stop(state->opt);
    }
    if (!isfinite(state->cache_energy)) {
        return 1.0e300;
    }
    return state->cache_energy;
}

static void finish(struct optbench_quench_out *out, double energy, double minimum, int calls, int code,
                   int capped) {
    out->energy = energy;
    out->min_distance = minimum;
    out->calls = calls;
    out->code = code;
    out->capped = capped;
}

int optbench_lj_quench(double *x, int dimension, int cap, int local_max, double converged_force,
                       double cutoff, double max_move, int max_iterations, struct optbench_quench_out *out) {
    int atoms;
    int calls = 0;
    int iteration;
    int remain;
    double step_min;
    struct eval_state state;

    if (out == NULL || x == NULL || dimension < 3 || dimension % 3 != 0 || cap < 0 || local_max < 1 ||
        converged_force <= 0.0 || cutoff <= 0.0 || max_move <= 0.0 || max_iterations < 1) {
        return 1;
    }
    atoms = dimension / 3;
    if (ensure_buffers(&reused, dimension) != 0) {
        return 1;
    }
    memcpy(reused.working, x, (size_t)dimension * sizeof(double));
    memset(&state, 0, sizeof(state));
    state.gradient = reused.gradient;
    state.cache_x = reused.cache_x;
    state.cache_gradient = reused.cache_gradient;
    state.last = reused.last;
    state.dimension = dimension;
    state.atoms = atoms;
    state.calls = &calls;
    state.cap = cap;
    state.local_max = local_max;
    state.cutoff = cutoff;
    state.converged_force = converged_force;
    if (evaluate(&state, reused.working) != 0) {
        finish(out, 0.0, 0.0, calls, 0, state.capped);
        return 0;
    }
    if (state.converged) {
        memcpy(x, reused.working, (size_t)dimension * sizeof(double));
        finish(out, state.cache_energy, state.cache_min, calls, (int)NLOPT_SUCCESS, 0);
        return 0;
    }
    for (iteration = 0; iteration < max_iterations; iteration++) {
        int i;
        if (calls >= cap || calls >= local_max) {
            break;
        }
        remain = local_max - calls;
        if (cap - calls < remain) {
            remain = cap - calls;
        }
        if (remain < 1) {
            break;
        }
        memcpy(reused.anchor, reused.working, (size_t)dimension * sizeof(double));
        memcpy(reused.trial, reused.anchor, (size_t)dimension * sizeof(double));
        memcpy(reused.last, reused.anchor, (size_t)dimension * sizeof(double));
        for (i = 0; i < dimension; i++) {
            reused.lower[i] = reused.anchor[i] - max_move;
            reused.upper[i] = reused.anchor[i] + max_move;
        }
        state.opt = nlopt_create(NLOPT_LD_LBFGS, (unsigned)dimension);
        if (state.opt == NULL) {
            return 1;
        }
        nlopt_set_min_objective(state.opt, objective, &state);
        nlopt_set_lower_bounds(state.opt, reused.lower);
        nlopt_set_upper_bounds(state.opt, reused.upper);
        nlopt_set_xtol_rel(state.opt, 1e-14);
        nlopt_set_ftol_abs(state.opt, 1e-14);
        nlopt_set_maxeval(state.opt, remain);
        nlopt_set_param(state.opt, "tolg", 1e-14);
        nlopt_optimize(state.opt, reused.trial, &step_min);
        nlopt_destroy(state.opt);
        state.opt = NULL;
        (void)step_min;
        scale_atomic_step(reused.anchor, reused.last, reused.scaled, dimension, max_move);
        if (same_bytes(reused.scaled, reused.anchor, dimension)) {
            break;
        }
        if (!same_bytes(reused.scaled, reused.last, dimension) && (calls >= cap || calls >= local_max)) {
            break;
        }
        memcpy(reused.working, reused.scaled, (size_t)dimension * sizeof(double));
        if (evaluate(&state, reused.working) != 0) {
            memcpy(reused.working, reused.anchor, (size_t)dimension * sizeof(double));
            break;
        }
        if (state.converged || state.capped || state.hit_local) {
            break;
        }
    }
    memcpy(x, reused.working, (size_t)dimension * sizeof(double));
    finish(out, state.cache_energy, state.cache_min, calls, state.converged ? (int)NLOPT_SUCCESS : (int)NLOPT_MAXEVAL_REACHED,
           state.capped);
    return 0;
}
