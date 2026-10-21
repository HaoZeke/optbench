// C callback in front of the SoftSaddle Lanczos or Davidson solver.
// One translation unit is compiled into each shared library. The two
// libraries both define gradient_scanning::Lanczos, so they stay apart.

#include <cmath>
#include <limits>
#include <vector>

#ifdef SOFTSADDLE_DAVIDSON
#include "davidson.hpp"
extern int softsaddle_davidson_kk;
#else
#include "lanczos.hpp"
#endif

namespace {

gradient_scanning::Lanczos solver;
int callback_failed = 0;

struct Bridge {
    void (*fn)(double const *x, double *g, int n, void *user);
    void *user;

    void operator()(blitz::Array<double, 1> &coordinates, blitz::Array<double, 1> & gradient) const {
        int const n = static_cast<int>(coordinates.size());
        std::vector<double> x(static_cast<size_t>(n));
        std::vector<double> g(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = coordinates(i);
        gradient.resize(n);
        if (callback_failed || fn == nullptr) {
            gradient = 0.0;
            return;
        }
        fn(x.data(), g.data(), n, user);
        if (callback_failed) {
            gradient = 0.0;
            return;
        }
        for (int i = 0; i < n; ++i) {
            coordinates(i) = x[static_cast<size_t>(i)];
            gradient(i) = g[static_cast<size_t>(i)];
        }
    }
};

}  // namespace

extern "C" {

void softsaddle_reset(void) {
    callback_failed = 0;
    solver.clear();
#ifdef SOFTSADDLE_DAVIDSON
    softsaddle_davidson_kk = 0;
#endif
}

void softsaddle_seed(unsigned int seed) {
    gradient_scanning::random::seed(seed);
}

void softsaddle_set_error(int flag) {
    callback_failed = flag;
}

int softsaddle_kk(void) {
#ifdef SOFTSADDLE_DAVIDSON
    return softsaddle_davidson_kk;
#else
    return -1;
#endif
}

int softsaddle_minimum_mode(
    double const *x,
    int n,
    double *eigenvalue,
    double *eigenvector,
    double *gradient_out,
    int initial,
    int iteration_limit,
    double finite_difference,
    double convergence_limit,
    int *iterations,
    double *convergence,
    int *status,
    void (*fn)(double const *coords, double *gradient, int length, void *user),
    void *user
) {
    using gradient_scanning::Lanczos;
    blitz::Array<double, 1> coords(n);
    blitz::Array<double, 1> evec(n);
    blitz::Array<double, 1> grad(n);
    for (int i = 0; i < n; ++i) {
        coords(i) = x[i];
        evec(i) = eigenvector[i];
    }
    double eval = *eigenvalue;
    solver.setIterationLimit(iteration_limit);
    solver.setFiniteDifference(finite_difference);
    solver.setConvergenceLimit(convergence_limit);
    solver.setReport(false);
    Lanczos::Initial which = Lanczos::RANDOM;
    if (initial == 1)
        which = Lanczos::GRADIENT;
    else if (initial == 2)
        which = Lanczos::USER;
    else if (initial == 3)
        which = Lanczos::PREVIOUS;
    solver.setInitial(which);
    callback_failed = 0;
    Bridge bridge{fn, user};
    gradient_scanning::GradientFunction function = bridge;
    int st = 0;
    try {
        st = static_cast<int>(solver.minimumMode(function, coords, eval, evec, grad));
    } catch (...) {
        callback_failed = 1;
        st = 8;
    }
    *eigenvalue = eval;
    *iterations = solver.iteration();
    *convergence = solver.convergence();
    *status = st;
    for (int i = 0; i < n; ++i) {
        eigenvector[i] = evec(i);
        gradient_out[i] = grad(i);
    }
    return callback_failed ? 1 : 0;
}

}
