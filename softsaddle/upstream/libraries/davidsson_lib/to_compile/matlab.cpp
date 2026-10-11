#include <blitz/array.h>
#include "davidson.hpp"
#include "Morse.h"

using namespace gradient_scanning;
using namespace blitz;

static Morse _morse;
static Lanczos _lanczos;
static double _box[9];
static double _energy;
static Array<double, 1> _x;
static Array<double, 1> _g;


extern "C"
void initialise(double const box[], double convergence, int iteration, double finitediff, bool report,
        double const x[], int n)
{
    for(int i=0; i<9; ++i)	
        _box[i]=box[i];
    _lanczos.setConvergenceLimit(convergence);
    _lanczos.setIterationLimit(iteration);
    _lanczos.setFiniteDifference(finitediff);
    _lanczos.setInitial(Lanczos::PREVIOUS);
    _lanczos.setReport(report);
    _x.resize(n);
    _g.resize(n);
    for(int i=0; i<n; ++i)
        _x(i)=x[i];
}


void morse(Array<double, 1> const &x, Array<double, 1> &g)
{
    // calculate gradient by calling external script
    // x is given (position) the gradient is returned on "g"
    double e;
    int const n=x.size();
    _x(Range(0,n-1))=x; 
    _morse.force(long(_x.size()/3), _x.data(), _g.data(), &e, _box);
    g=-_g(Range(0,n-1));
    if(isnan(_energy))
        _energy=e;
}

extern "C"
double lanczos(int n, double const x[], double g[], double v[], double*energy)
{
    double e;
    _energy=NAN;
    n*=3; //comment if number of parameters is passes instead of number of atoms
    Array<double, 1> const  _x((double*)x, shape(n), neverDeleteData);
    Array<double, 1> _v(v, shape(n), duplicateData);
    Array<double, 1> _g(n);
    _lanczos.minimumMode(morse, _x, e, _v, _g);
    *energy=_energy;
    for(int i=0; i<n; ++i) {
        v[i]=_v(i);
        g[i]=_g(i);
    };
    return e;
}

extern "C"
void report(double*convergence, int*iteration)
{
    *convergence=_lanczos.convergence();
    *iteration=_lanczos.iteration();
}
