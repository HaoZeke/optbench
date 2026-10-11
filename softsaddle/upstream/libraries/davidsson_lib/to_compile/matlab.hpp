#include <blitz/array.h>
#include "davidson.hpp"
#include "Morse.h"

using namespace gradient_scanning;
using namespace blitz;

extern Morse _morse;
extern Lanczos _lanczos;
extern double _box[9];
extern double _energy;
extern Array<double, 1> _x;
extern Array<double, 1> _g;


extern "C"
void initialise(double const box[], double convergence, int iteration, double finitediff, bool report,
        double const x[], int n);
void morse(Array<double, 1> const &x, Array<double, 1> &g);
extern "C"
double lanczos(int n, double const x[], double g[], double v[], double*energy);
extern "C"
void report(double*convergence, int*iteration);
