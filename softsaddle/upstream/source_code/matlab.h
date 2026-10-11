void initialise(double const box[], double convergence, int iteration, double finitediff, bool report, double const x[], int n);
double lanczos(int n, double const x[], double g[], double v[], double*energy);
void report(double*convergence, int*iteration);

