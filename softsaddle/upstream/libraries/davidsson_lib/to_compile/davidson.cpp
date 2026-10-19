/** @file
Class Lanczos.      
@author Original J. C. Berthet
@date 2007-2008
@authorDavidsonModification Carlos Argaez
@date 2014-2015
University of Iceland
*/
#undef NDEBUG
#define NDEBUG
#ifdef NDEBUG
#define HAVE_INLINE
#define GSL_RANGE_CHECK_OFF
#else
#warning NDEBUG not defined. Compute with NDEBUG defined for release.
#define BZ_DEBUG
#endif
#include <cfloat>
#include <cassert>
#include <iostream>
#include <fstream>
#include <complex>

#include <blitz/array.h>
#include <random/normal.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "gsl_eigen.h"
#include "davidson.hpp"
using namespace std;

using namespace blitz;
using namespace gradient_scanning;

int softsaddle_davidson_kk = 0;
// The dense minimum Hessian is an internal anchor. MATLAB counts
// sp_func_calls as report()'s iteration count plus one, which is the
// initial gradient plus one finite-difference product per iteration.
// Column builds must not increment the harness counter.
extern "C" int softsaddle_charge_gradients;

namespace {
struct SuppressCharge {
    int previous;
    SuppressCharge() : previous(softsaddle_charge_gradients) { softsaddle_charge_gradients = 0; }
    ~SuppressCharge() { softsaddle_charge_gradients = previous; }
};
}  // namespace
//-------------------------------------------------Declaration variable and functions-----------------------------
namespace {
    /// Parameters to be used with blitz::Array @{
    blitz::firstIndex I;
    blitz::secondIndex J;
    blitz::thirdIndex K;
    blitz::Range const All=Range::all();
    /// @}
    
    /** Diagonalise a tridiagonal matrix.
    @pre Size of @a diagonal @em n and size of @a subDiagonal @em n-1.
    eigenvectors are in columns (e.g. first eigenvector: @em eigenvectors(All,0);)
    */
    void diagonalise(Array<double, 1> const & diagonal, Array<double, 1> const & subDiagonal,
                     Array<double, 1> & eigenvalues, Array<double, 2> & eigenvectors)
    {
        int const n=diagonal.size();
        assert(n == subDiagonal.size()+1);
        gsl_vector const diag=fastWrapToGsl(diagonal);
        gsl_vector const sub_diag=fastWrapToGsl(subDiagonal);      
        
        gsl_vector eval=wrapToGsl(eigenvalues);
        gsl_matrix_view evec=
            gsl_matrix_view_array_with_tda(
                                           eigenvectors.data(),
                                           eigenvectors.rows(),
                                           eigenvectors.cols(),
                                           eigenvectors.stride(0)
                                           ); 
        gsl_eigen_triagv_workspace * const w=gsl_eigen_triagv_alloc(n);
        gsl_eigen_triagv(&diag, &sub_diag, &eval, &evec.matrix, w);
        gsl_eigen_triagv_free(w);
        gsl_eigen_symmv_sort(&eval, &evec.matrix, GSL_EIGEN_SORT_VAL_ASC);
    }
    

    void diagonalisedavidson(Array<double, 2> & diagonal,
                     Array<double, 1> & eigenvalues, Array<double, 2> & eigenvectors)
    {
        int const n=diagonal.size();
        gsl_matrix diag=wrapToGsl(diagonal);
        
        gsl_vector eval=wrapToGsl(eigenvalues);
        gsl_matrix_view evec=
            gsl_matrix_view_array_with_tda(
                                           eigenvectors.data(),
                                           eigenvectors.rows(),
                                           eigenvectors.cols(),
                                           eigenvectors.stride(0)
                                           ); 
        gsl_eigen_symmv_workspace * const w=gsl_eigen_symmv_alloc(n);
        gsl_eigen_symmv(&diag, &eval, &evec.matrix, w);
        gsl_eigen_symmv_free(w);
        gsl_eigen_symmv_sort(&eval, &evec.matrix, GSL_EIGEN_SORT_VAL_ASC);
    }



    BZ_DECLARE_FUNCTION_RET(isnan, bool)
        
} // end anonymous namespace

//----------------------------------------------Definition functions in namespace lanczos------------------------
/** @class gradient_scanning::Lanczos
@brief Compute the lowest eigenvalue and eigenvector.
@section hessian Hessian Minimum Mode
Compute the hessian lowest eigenvalue from function computing gradient by using finite differences.
@see minimumMode(), gradient_scanning#GradientFunction.
@section reference Reference
<em> Comparison of methods fo finding the saddle points without knowledge of the final sate.</em> R.A. Olsen, G.J. Kroes et al. J. Chem. Physics, (2004), Vol. @b 121, n 20.
*/

Lanczos::Lanczos()
{
    initialiseOperatingMembers();
    initialiseParametersMembers();
    initialiseResultsMembers();
}

Lanczos::Lanczos(Lanczos const & lanczos)
{
    operator=(lanczos);
}

/// Delete data about past Lanczos Iterative Methods performed.
void Lanczos::clear()
{
    initialiseOperatingMembers();
    initialiseResultsMembers();
}

/// Convergence reached by last Lanczos.
double Lanczos::convergence() const
{
    return convergence_;
}

/** Parameter controling when the iteration process stops.
The iteration process stops when @f$ \left| \lambda_i-\lambda_{i-1} \right| < tolerance\_ @f$ where @f$ \lambda @f$ is the smallest eigenvalue.
@see getMaximumIterations() and setMaximumIterations()
*/
/// @{
double Lanczos::getConvergenceLimit() const
{
    return convergenceLimit_;
}

void Lanczos::setConvergenceLimit(double convergenceLimit)
{
    convergenceLimit_=convergenceLimit;
}
/// @}

/// Lowest eigenvalue for last call to minimumMode().
double Lanczos::eigenvalue() const
{
    return eigenvalue_;
}

/// Lowest mode's eigenvector for last call to minimumMode(). 
blitz::Array<double, 1> Lanczos::eigenvector() const
{
    return eigenvector_;
}

/** Step to calculate derivatives.
To calculate the second derivatives of the gradient, a finite difference approximation is used: @f$ \mathbf g'=\frac{\mathbf g(\mathbf x+\Delta \mathbf x)-\mathbf g(\mathbf x)}{|\Delta \mathbf x|} @f$. The functions getFiniteDifference() and setFiniteDifference() allows to read and set the step @f$ |\Delta \mathbf x| @f$. 
@{ */
double Lanczos::getFiniteDifference() const
{
    return finiteDifference_;
}

void Lanczos::setFiniteDifference(double difference)
{
    finiteDifference_=difference;
}
/// @}

/* Generate random vector from Gaussian distribution.
Average is 0 and Stdv 1.
@param[out]  vector      Vector to randomise.
*/
void Lanczos::gaussian(blitz::Array<double, 1> & vector)
{
    for (Array<double, 1>::iterator i=vector.begin(); i != vector.end(); i++)
        *i=random::normal.random();
}

/** Tell how to initialise the Lanczos iterative algorithm.
Lanczos needs to initial vector to start the iterative process.
The convergence of the process is tested by comparing the lowest eigenvalues found by two successive iterations.
#Initial controls how the initial vector is chosen and to what value is compared the first eigenvalue.
*/
/// @{
Lanczos::Initial Lanczos::getInitial() const
{
    return initial_;
}

void Lanczos::setInitial(Initial initial)
{
    initial_=initial;
}
/// @}

/* Random isotropic vector of norm one.
@param[out]  vector      Vector to randomise.
*/
void Lanczos::isotropic(blitz::Array<double, 1> & vector)
{
    double norm;
    do {
        gaussian(vector);
        norm=sum(vector*vector) ;
    } while (norm == 0.0);
    vector/=sqrt(norm);
}

/// Number of iterations used by last Lanczos.
int Lanczos::iteration() const
{
    return iteration_;
}

/** Maximum number of iterations.
@{ */
int Lanczos::getIterationLimit() const
{
    return iterationLimit_;
}

void Lanczos::setIterationLimit(int iterationLimit)
{
    iterationLimit_=iterationLimit;
}
/// @}

/** Hessian lowest eigenvalue and corresponding eigenvector.
@param[in] function Function to compute the gradient.
@param[in] coordinates Coordinates at which to compute the minimum mode.
@param[in,out] eigenvalue  Output lowest eigenvalue. For input depends on getInitial().
@param[in,out] eigenvector Output eigenvector of the lowest eigenvalue. For input depends on getInitial().
@param[out] gradient Gradient at @a coordinates.
@return     The status (see status()).
@note Input values in @a eigenvalue and @a eigenvector are used only if getInitial() is #USER.\n
When returned status is #UNCOMPLETE, the function returns with the best estimation  it has of the lowest eigenvalue and eigenvector.
When the returned status is #UNCOMPLETE because the initial vector was an eigenvector, the function returns the eigenvector (i.e. initial vector) and its eigenvalue.
@pre  If getInitial() is #USER the norm of @a eigenvector must not be zero.
*/
Lanczos::Status Lanczos::minimumMode(
                                     GradientFunction const & function,
                                     blitz::Array<double, 1> const & coordinates,
                                     double & eigenvalue,
                                     blitz::Array<double, 1> & eigenvector,
                                     blitz::Array<double, 1> & gradient
                                     ) {
    function_=&function;
    const unsigned ncoord=coordinates.size();
    gradient.resize(ncoord);
    Array<double, 1> coord(ncoord);
    Array<double, 1>  coordinatesprevious_(ncoord),  gradientprevious_(ncoord);
    coord=coordinates;
    coordinates_=&coord;
    gradient_=&gradient;
    setStatus(STARTING);
    function(coord, gradient);
    switch (initial_) {
        case RANDOM:
            eigenvalue=numeric_limits<double>::infinity();
            eigenvector.resize(ncoord);
            isotropic(eigenvector);
            break;
        case GRADIENT:
            eigenvalue=numeric_limits<double>::infinity();
            eigenvector.resize(ncoord);
            eigenvector=gradient;
        case USER:
            assert(sum(eigenvector*eigenvector) != 0.0);
            break;
        case PREVIOUS:
            eigenvalue=eigenvalue_;
            eigenvector.resize(ncoord);
            swap(eigenvector_, eigenvector);
            break;
        default:
            std::cerr << __FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << std::endl;
            exit(EXIT_FAILURE);                  
    };
    if (eigenvector.size() != ncoord) {
        eigenvector.resize(ncoord);
        isotropic(eigenvector);
        eigenvalue=numeric_limits<double>::infinity();
    };
    Status status=lanczos(&Lanczos::finiteH, eigenvalue, eigenvector);
    coordinatesprevious_.resize(ncoord);
    gradientprevious_.resize(ncoord);
    coordinatesprevious_=coordinates;
    gradientprevious_=gradient; 
    function_=0;
    coordinates_=0;
    gradient_=0;
    if (report_)
        report();
    assert(eigenvector.size() == ncoord);
    return status;
}

/// Tell if a final report is to be printed.
bool Lanczos::getReport() const
{
    return report_;
}

/// Tell if a final report is to be printed.
void Lanczos::setReport(bool turn)
{
    report_=turn;
}

/** Status of the Lanczos process.
When the lanczos is not running, status indicates either NOT_RUNNING (before any call to minimumMode()) or why the last minimumMode() execution has ended (e.g. CONVERGED). 
Function status() may also be used while minimumMode() is running, during a call back to the function computing the gradient
(see parameter @a gradient in
 @link
 Lanczos::minimumMode(GradientFunction const &, blitz::Array<double, 1> const &, double &, blitz::Array<double, 1> &, blitz::Array<double, 1> &)
 minimumMode(GradientFunction, ... )
@endlink 
).
In this case status() indicates where the minimisation is currently.
@see #Status.
*/
Lanczos::Status Lanczos::status() const
{
    return status_;
}

/** Warning when a certain event occurs.
Here you may specify a certain number of events, you want to be warned about when they occurs.
The default value is #NOTHING which means that no warning will occur.
Events may be combined with operator @c bitor.
Example:
@code
warnWhen(EXCEED_ITERATIONS | EIGENVECTOR);
@endcode
@see status().
@{ */
Lanczos::Status Lanczos::getWarnWhen() const
{
    return warnWhen_;
}

void Lanczos::setWarnWhen(Status status)
{
    warnWhen_ = status;
}

void Lanczos::setWarnWhen(int status)
{
    warnWhen_ = static_cast<Status>(status);
}

// @}
/** Copy parameters.
Copy parameters controling how the Lanczos Iterative method is performed. Note that it does does copy the results displayed by @a lanczos. In other word, results produced by iteration(), eigenvalue(), etc ... will remain different.
*/
Lanczos const & Lanczos::operator=(Lanczos const & lanczos)
{
    initialiseResultsMembers();
    initialiseOperatingMembers();
    convergenceLimit_=lanczos.convergenceLimit_;
    iterationLimit_=lanczos.iterationLimit_;
    finiteDifference_=lanczos.finiteDifference_;
    initial_=lanczos.initial_;
    report_=lanczos.report_;
    warnWhen_=lanczos.warnWhen_;
    return *this;
}

/** Compute product H.q.
Calling the finite difference @f$ \delta @f$. The product @f$ \mathrm H. \mathbf q @f$ is calculated from the gradient:
@f$  \mathrm H. \mathbf q =\frac{ \mathbf G(\mathbf x+\delta \mathbf q)-\mathbf G(\mathbf x)}{\delta}  @f$.
The gradient is computed by function compute();
@see #finiteDifference_, realH(), and compute().
@ingroup finiteH
*/
blitz::Array<double, 1> Lanczos::finiteH(blitz::Array<double, 1> const & q)
{
    const int N=q.size();
    Array<double, 1> xk(N), gk(N), result(N);
    xk=*coordinates_+q*getFiniteDifference();
    (*function_)(xk, gk);
    result=(gk - *gradient_)/getFiniteDifference();
    return result;
}


//////  Buidling the FULL Hessian's DIAGONAL
// This routine construct the Hessian in the given point.
blitz::Array<double, 2> Lanczos::DiagH()
{
    const int N=coordinates_->size();
    Array<double, 1> xk(N), gk(N);
    Array<double, 2> result(N,N);
       for(int i=0; i< N; ++i)
        {
        xk=*coordinates_;
        xk(i)+=getFiniteDifference();
        (*function_)(xk, gk);
        result(i,All)=(gk- *gradient_)/getFiniteDifference();
        }
    return result;
}


/////// Building the FULL Hessian MATRIX
// This routine builds the Hessian at the minimum point.
blitz::Array<double, 2> Lanczos::Hessianominimo(Array<double, 1> const &minpos)
{
    SuppressCharge quiet;
    const int N=coordinates_->size();
    Array<double, 1> xk(N), gk(N), ggk(N);
    Array<double, 2> result(N,N);
       for(int i=0; i< N; ++i)
        {
        xk=minpos;
(*function_)(xk, ggk);
        xk(i)+=getFiniteDifference();
        (*function_)(xk, gk);
        result(i,All)=(gk-ggk)/getFiniteDifference();
        }
    return result;
}

/// Clear and (re-)initialise members used while running the lanczos iterative method.
void Lanczos::initialiseOperatingMembers()
{
    function_=0;
    hessian_.free();
    coordinates_=0;
    gradient_=0;
    coordinatesprevious_.free();
    gradientprevious_.free();
    softsaddle_davidson_kk = 0;
    eigenvalue_=numeric_limits<double>::infinity();
}

/// Clear and (re-)initialise members which controls how the lanczos iterative method is performed.
void Lanczos::initialiseParametersMembers()
{
    convergenceLimit_=0;
    iterationLimit_=20;
    finiteDifference_=1e-5;
    initial_=RANDOM;
    report_=false;
    warnWhen_=NOTHING;
}

/// Clear and (re-)initialise members which contains data about the last lanczos iterative method executed.
void Lanczos::initialiseResultsMembers()
{
    iteration_=0;
    convergence_=0.0;
    eigenvalue_=0.0;
    eigenvector_.free();
    status_=NOTHING;
}

/** Lanczos Iterative Method.
The function tridiagonalise a symmetric matrix using Lanczos Iterative Method.
An approximation of the solution may be provided by the user through @a lowestEigenvalue and @a eigenvector.
If the user does not wish to provide an approximation @a lowestEigenvalue must be set to INFINITY (to make first convergence test fail) and eigenvector must be set to a non-zero random vector.
The function returns with the latest estimation of @a lowestEigenvalue and @a eigenvector 
@param[in] H Function computing the product @f$ H.\mathbf{q} @f$.
@param[in, out] lowestEigenvalue Lowest eigenvalue. 
@param[in, out] eigenvector Eigenvector corresponding to lowest eigenvalue.
@return     Why the iterative process stops.
@see Status minimumMode().
@note The returned values are copied into #eigenvalue_ and #eigenvector_.
*/




       Lanczos::Status Lanczos::lanczos(
                                 blitz::Array<double, 1> (Lanczos::*H)(const blitz::Array<double, 1>&),
                                 double & lowestEigenvalue,
                                 blitz::Array<double, 1> & eigenvector
                                 ) {
    int const ncoord=eigenvector.size();
    assert(ncoord > 0);
    // Equations and notations are based on the description of the algorithm in RA Olsen, GJ Kroes et al, J. chem. phys. vol 121, no 20 p 9776

        Array<double, 1> minpos(ncoord);
        if (ncoord == 525) {
        minpos=7.4940034, 7.42614783, 14.5736894, 7.48480085, 12.3764495, 14.5730335, 10.3427798, 7.42614773, 14.5736893, 10.3519825, 12.3764494, 14.5730333, 11.7762657, 9.90910704, 14.5737596, 6.06051742, 9.90910714, 14.5737597, 8.91839158, 9.9039789, 14.5845006, 0.687292407, 0.394217841, 7.84431476, 2.05942808, 2.77011235, 7.84505945, 0.686639629, 3.56097738, 10.0891613, 2.06181218, 1.1842469, 10.0912013, 0.687633689, 1.97363201, 12.3706321, 2.05912247, 4.35576465, 12.3728552, 0.684328788, 5.14681297, 7.84425471, 2.05298223, 7.52287023, 7.84241335, 0.686163917, 8.32147618, 10.0953915, 2.05441977, 5.93916056, 10.0926707, 0.686526143, 6.73424807, 12.3738942, 2.07108313, 9.1072715, 12.384102, 0.686818566, 9.90504962, 7.85068317, 2.06139914, 12.2790041, 7.85397914, 0.690671965, 13.0707088, 10.0953891, 2.06839112, 10.6954524, 10.1088743, 0.699618189, 11.4821128, 12.3812683, 2.0816867, 13.8454208, 12.3839387, 0.687264783, 14.6548321, 7.84494239, 2.06003275, 17.0293807, 7.84531827, 0.688249937, 17.8187353, 10.0898272, 2.06327921, 15.4410279, 10.0943747, 0.692104579, 16.2252389, 12.3717902, 2.06371926, 18.6061569, 12.3726475, 3.432987, 0.394441212, 7.84624542, 4.80503879, 2.77350883, 7.84944647, 3.43291061, 3.56414286, 10.0936832, 4.80773387, 1.19013333, 10.0945587, 3.43528715, 1.98167996, 12.3747292, 4.80633177, 4.37237428, 12.3828335, 3.42515055, 5.14650787, 7.84195059, 4.79729524, 7.52351117, 7.83429515, 3.42345429, 8.30968428, 10.0775481, 4.79120298, 5.94148534, 10.0770132, 3.4319963, 6.73544398, 12.3838406, 4.81168832, 9.08917964, 12.2973125, 3.42547746, 9.90041667, 7.84150678, 4.79941341, 12.2801397, 7.84136074, 3.44435838, 13.0629408, 10.1195585, 4.7998643, 10.6914911, 10.070122, 3.47852711, 11.461286, 12.4288243, 4.84984999, 13.8335825, 12.4283099, 3.43306439, 14.6517926, 7.85343753, 4.80394922, 17.0285754, 7.84906198, 3.4349817, 17.8176798, 10.0937318, 4.81061432, 15.4368506, 10.107293, 3.44508173, 16.2225532, 12.3792153, 4.80912135, 18.6159435, 12.3752646, 6.17573391, 0.397475721, 7.84619864, 7.54790245, 2.77723181, 7.85347505, 6.1801809, 3.5764363, 10.1076547, 7.54887196, 1.19610191, 10.0949024, 6.1817551, 2.00097842, 12.3799754, 7.5490255, 4.41514901, 12.4283747, 6.17032466, 5.14756057, 7.84107927, 7.5451876, 7.52268664, 7.8273122, 6.17812579, 8.32107249, 10.0490514, 7.54249285, 5.9426634, 10.0698722, 6.16010334, 6.75461948, 12.2969462, 7.5554511, 9.11683155, 12.2967111, 6.17040444, 9.90322243, 7.82733716, 7.54383846, 12.2808898, 7.82711708, 6.17677551, 13.0740199, 10.0697117, 7.5498972, 10.6925363, 10.0462527, 6.20311623, 11.4710034, 12.2805541, 7.5707532, 13.8681913, 12.2966079, 6.17491292, 14.658986, 7.84068578, 7.54753657, 17.0370968, 7.84133308, 6.17678908, 17.8236513, 10.0937511, 7.55211642, 15.4574215, 10.0760876, 6.18717747, 16.2331544, 12.3821926, 7.54930864, 18.6285884, 12.3742567, 8.91839084, 0.400577231, 7.84515156, 10.2888792, 2.77723166, 7.85347503, 8.91839123, 3.5826669, 10.1191025, 10.2879106, 1.19610166, 10.0949023, 8.91839148, 2.01351768, 12.3836914, 10.2877574, 4.4151488, 12.4283746, 8.91839083, 5.14688232, 7.84127127, 10.291594, 7.5226865, 7.82731217, 8.91839121, 8.3228768, 10.0463073, 10.2942897, 5.94266315, 10.0698721, 8.91839147, 6.76887695, 12.2806621, 10.2813318, 9.11683136, 12.296711, 8.91839082, 9.90224077, 7.82165712, 10.2929431, 12.2808896, 7.82711704, 8.91839121, 13.0660971, 10.0486548, 10.2868853, 10.6925361, 10.0462526, 8.91839146, 11.4776524, 12.2963854, 10.2660296, 13.8681911, 12.2966078, 8.91839083, 14.6596819, 7.83358885, 10.2892451, 17.0370967, 7.84133304, 8.91839124, 17.8305425, 10.0921556, 10.2846662, 15.4574212, 10.0760875, 8.91839147, 16.2436495, 12.3823172, 10.2874742, 18.6285882, 12.3742567, 11.661048, 0.397475371, 7.84619855, 13.0317431, 2.77350825, 7.84944635, 11.6566016, 3.57643582, 10.1076545, 13.029049, 1.19013255, 10.0945583, 11.6550279, 2.00097803, 12.3799752, 13.0304511, 4.3723738, 12.3828333, 11.6664572, 5.14756023, 7.84107919, 13.0394866, 7.5235106, 7.83429504, 11.6586567, 8.32107202, 10.0490512, 13.0455799, 5.94148456, 10.0770128, 11.6766796, 6.75461906, 12.296946, 13.0250945, 9.08917917, 12.2973124, 11.6663774, 9.9032221, 7.8273371, 13.0373685, 12.2801391, 7.84136063, 11.660007, 13.0740194, 10.0697115, 13.0369185, 10.6914903, 10.0701216, 11.6336667, 11.471003, 12.2805538, 12.9869328, 13.833582, 12.4283098, 11.661869, 14.6589857, 7.84068569, 13.0328327, 17.0285748, 7.84906186, 11.6599935, 17.8236509, 10.093751, 13.0261685, 15.4368498, 10.1072926, 11.6496055, 16.233154, 12.3821923, 13.0276615, 18.6159431, 12.3752645, 14.4037959, 0.394440225, 7.84624518, 15.7773544, 2.7701109, 7.84505955, 14.403872, 3.56414202, 10.0936829, 15.7749716, 1.18424588, 10.091199, 14.4014959, 1.98167946, 12.3747279, 15.7776603, 4.35576459, 12.372854, 14.4116323, 5.1465069, 7.84195036, 15.7838002, 7.52286879, 7.84241345, 14.4133283, 8.30968343, 10.0775478, 15.7823639, 5.93915953, 10.0926683, 14.4047867, 6.73544347, 12.3838393, 15.7656996, 9.10727144, 12.3841008, 14.4113054, 9.9004157, 7.84150659, 15.7753833, 12.2790027, 7.85397927, 14.3924243, 13.0629399, 10.1195582, 15.7683926, 10.6954514, 10.108872, 14.3582559, 11.4612856, 12.4288231, 15.755096, 13.8454207, 12.3839376, 14.4037185, 14.6517916, 7.85343729, 15.7767497, 17.0293792, 7.84531839, 14.401801, 17.8176789, 10.0937314, 15.7735045, 15.4410269, 10.0943725, 14.3917013, 16.2225527, 12.379214, 15.7730635, 18.6061568, 12.3726463, 17.1494942, 0.394215389, 7.8443089, 18.5227917, 2.76980123, 7.843641, 17.1501433, 3.56097769, 10.0891601, 18.5227912, 1.1833448, 10.0891247, 17.1491488, 1.9736321, 12.3706309, 18.5227912, 4.3538324, 12.369524, 17.1524578, 5.14681052, 7.8442488, 18.5227916, 7.52719116, 7.84628493, 17.1506189, 8.3214765, 10.0953902, 18.5227912, 5.94151707, 10.0901478, 17.1502563, 6.73424815, 12.3738929, 18.5227912, 9.11230362, 12.3779016, 17.149968, 9.9050472, 7.85067732, 18.5227917, 12.2805969, 7.84685574, 17.1461109, 13.070709, 10.0953879, 18.5227912, 10.6992684, 10.0976514, 17.1371642, 11.4821129, 12.3812671, 18.5227912, 13.852537, 12.3735364, 17.1495218, 14.6548297, 7.84493655, 18.5227917, 17.0307284, 7.84382719, 17.148533, 17.8187356, 10.089826, 18.5227912, 15.4443097, 10.0895882, 17.1446779, 16.225239, 12.371789, 18.5227912, 18.6043869, 12.3704446;
        } else {
            minpos = *coordinates_;
        }
int &kk = softsaddle_davidson_kk;

   Array<double,2> Iden(ncoord,ncoord);
    for (int i=0; i<ncoord; i++)
    {
        for (int j=0; j<ncoord; j++)
        {
            if(j==i)
            {
                Iden(i,j)=1.0;
            } else {
                Iden(i,j)=0.0;
            }
        }
    }


        double denominator,denominatorold, fourthproduct;
        Array<double,1> gradiente2(ncoord),coordxx(ncoord),deltacoord(ncoord),gammagradient(ncoord), firstproduct(ncoord),thirdproduct(ncoord);
        coordxx=*coordinates_;
        Array<double, 1>  &coordx=coordinatesprevious_,  &gradiente1=gradientprevious_;
        gradiente2=*gradient_;
      Array<double, 2> &HessBFGS=hessian_;
       
        if(coordx.size() != ncoord)
            {
             coordx.resize(ncoord);
             gradiente1.resize(ncoord);
             HessBFGS.resize(ncoord,ncoord);
             coordinatesprevious_=minpos;
             gradientprevious_=0.0;
             denominatorold=0.0;
            }

  
        Array<double,2> firstmatrix(ncoord,ncoord);
        Array<double,2> gradientMatrix(ncoord,ncoord),secondproduct(ncoord,ncoord),secondmatrix(ncoord,ncoord),secondmatrix1(ncoord,ncoord),secondmatrix2(ncoord,ncoord),secondmatrix3(ncoord,ncoord),secondmatrix4(ncoord,ncoord),secondmatrix5(ncoord,ncoord);


        deltacoord=coordxx-coordx;
        gammagradient=gradiente2-gradiente1;
        denominator=sum(deltacoord*gammagradient);
        if(fabs(denominator)<=1e-19)
       {
cout << "ITS BEEN ZERO " << denominator << endl;
             coordx.resize(ncoord);
             gradiente1.resize(ncoord);
             HessBFGS.resize(ncoord,ncoord);
             coordinatesprevious_=minpos;
             coordx=minpos;
             gradientprevious_=0.0;
             gradiente1=0.0;
Array<double, 2> minHess=Hessianominimo(minpos);
HessBFGS=minHess;
             deltacoord=coordxx-coordx;
             gammagradient=gradiente2-gradiente1;
             denominator=sum(deltacoord*gammagradient);
       }

if(kk==0)
{
Array<double, 2> minHess=Hessianominimo(minpos);
HessBFGS=minHess;
        deltacoord=coordxx+coordx;
        gammagradient=gradiente2+gradiente1;
        denominator=sum(deltacoord*gammagradient);
}
        secondproduct=gammagradient(I)*gammagradient(J);
        firstproduct=sum(HessBFGS(I,J)*deltacoord(J),J);
        thirdproduct=sum(deltacoord(J)*HessBFGS(I,J),J);
        fourthproduct=sum(thirdproduct*deltacoord);
        firstmatrix=secondproduct/denominator;
        secondmatrix1=firstproduct(I)*thirdproduct(J);
        secondmatrix2=secondmatrix1/fourthproduct;
        HessBFGS=HessBFGS+firstmatrix-secondmatrix2;


        coordinatesprevious_=coordxx;

        gradientprevious_=gradiente2;
kk++;

    int maxIteration;
    if (iterationLimit_ >= ncoord)
        maxIteration=ncoord;
    else
        maxIteration=iterationLimit_;

    Array<double, 1> eval(maxIteration); 
    Array<double, 2> evec(maxIteration, maxIteration); 
    Array<double, 1> alpha(maxIteration+1), beta(maxIteration+1); 
    Array<double, 2> alphadavidson(maxIteration,maxIteration);
    Array<double, 1> r(ncoord), wd(ncoord),  rd(ncoord), x(ncoord), u(ncoord), t(ncoord),td(ncoord);
    Array<double, 2> q(ncoord, maxIteration+1), uu(ncoord,maxIteration+1), qq(ncoord,maxIteration), asf(ncoord,ncoord);
    q(All, 0)=0.0;
    Array<double, 2> vT; 
    Array<double, 1> evl(maxIteration);
    Array<double, 1> a, b, lambda;
    Array<double, 1> finiteH(ncoord,ncoord);
    Array<double, 1> hessian_(ncoord,ncoord);
    r=eigenvector;
    beta(0)=sqrt(sum(r*r));
    assert(beta(0) > 0.0);
    alpha(0)=0.0;
    ofstream myfile; 
    ofstream filemethod; 
    ofstream eigenvalues;
    myfile.open ("criteria.txt", fstream::app);
    filemethod.open ("method.txt", fstream::app);
    eigenvalues.open ("eigenvalues.txt", fstream::app);
    myfile << "kk " << kk << ".\n";
    filemethod << "kk " << kk << ".\n";
    eigenvalues << "kk " << kk << ".\n";



    setStatus(RUNNING);
    double oldEigenvalue=lowestEigenvalue;
    double convergence=0;
    int k=0;
    Status status=RUNNING;
    double decide, criteria;
    Array<double,1> direction(ncoord), criteria1(ncoord), prueba(ncoord), prueba2(ncoord);

    q(All,0)=r/beta(0);
    direction=(this->*H)( q(All,0) );
    criteria1=(direction-sum(HessBFGS(I,J)*q(All,0)(J),J))*beta(0);


    criteria=sqrt(sum(criteria1*criteria1));


    myfile << "criteria " << criteria << "\n";
        if(criteria>12.0){
         filemethod << "lanczos \n";
        while (status == RUNNING) {
        k++;
       if(k==1)
      {
      q(All, k)=q(All,0);
      q(All,0)=0.0;
      u=direction;
      }else{
        q(All, k)=r/beta(k-1);
        u=(this->*H)( q(All, k) );
           }
        uu(All,k)=u;
        r = u - q(All, k-1)*beta(k-1);
        alpha(k)=sum(q(All, k)*r);
        r -= q(All, k)*alpha(k);
        beta(k)=sqrt(sum(r*r));
        if (beta(k) <= 1e-10*fabs(alpha(k))) {
            -- k;
            status=UNCOMPLETE;           
            break;
        };
        Range range(0, k-1);
        lambda.reference(eval(range));
        vT.reference(evec(range, range));
        if (k > 1) {
            a.reference(alpha(Range(1, k)));
            b.reference(beta(Range(1, k - 1)));
            diagonalise(a, b, lambda, vT);
      }
        else {
            lambda(0)=alpha(1);
            vT(0)=1.0;
        };
        eigenvalues << " lambda Lanczos " << lambda(0) << endl;
        convergence=fabs(eval(0) - oldEigenvalue); 
        oldEigenvalue=eval(0);
        if (k >= maxIteration) 
            status=EXCEED_ITERATION;
        if (convergence < convergenceLimit_)
            status=CONVERGED;
        if (k == ncoord)
            status=FULL_MATRIX;
    };
    setStatus(status);
    // Memorise the final state of the lanczos
    iteration_ = k;

    convergence_=convergence;
    // It may happen that the number of iteration is 0, lanczos exited with EIGENVECTOR. In this case keep data provided by user.
    if (k > 0) {
        Array<double, 2> Q=q(All, Range(1, k));
        Array<double, 2> _v(Q.rows(), vT.cols());
        _v=sum(Q(I, K)*vT(K, J), K);
        lowestEigenvalue=lambda(0);
        Array<double, 1> v=vT(All, 0);
        eigenvector.resize(ncoord);
        eigenvector=sum(Q(I, J)*v(J), J);
    }
    else {
        assert(status == UNCOMPLETE);
        // case where initial vector is an eigenvector, calculate eigenvalue.
        lowestEigenvalue=sum(u*q(All, 1));// q is normal.
    };
    eigenvalue_=lowestEigenvalue;
    eigenvector_=eigenvector;
    eigenvalues << "The lowest eigenvalue from Lanczos is: " << lowestEigenvalue <<" \n " ;
    return status_;

} else {
         filemethod << "davidson \n";
    setStatus(RUNNING);
    double oldEigenvalue=lowestEigenvalue;
   double convergence=0;
   int k=0;
   eigenvector.resize(ncoord);
   while (status == RUNNING) {
     if(k==0)
    {
    q(All, k)=r/sqrt(sum(r*r));
    u=direction;
    }else{
       q(All, k)=r/sqrt(sum(r*r));
       u=(this->*H)( q(All, k) );
         }
       qq(All,k)=q(All, k);
       uu(All,k)=u;
       alphadavidson=sum(qq(K,I)*uu(K,J),K);
       Range range(0, k);
       lambda.reference(eval(range));
       vT.reference(evec(range, range));
       Array<double, 2> adavidson(k+1,k+1);
       if (k > 0) {
           adavidson=alphadavidson(Range(0, k), Range(0,k));
           diagonalisedavidson(adavidson, lambda, vT);
       } else {
           adavidson=alphadavidson(0,0);
           lambda(0)=alphadavidson(0,0);
          vT(0)=1.0;
      };
      lowestEigenvalue=lambda(0);
       eigenvalues << " lambda Davidson " << lambda(0) << endl;
       evl=vT(All, 0);
       x(All)=0.0;
       t(All)=0.0;
       for(int i=0; i<=k; i++)
       {
          x+=uu(All,i);
           t+=lambda(0)*qq(All,i);
      }
       wd=x-t;
       asf=lambda(0)*Iden-HessBFGS;
       for(int i=0; i<ncoord; i++)
       td(i)=wd(i)/asf(i,i);
       r=td;
       for(int i=0; i<= k; i++)
       r-=  sum( td*qq(All,i) )*qq(All,i) ;
      beta(k)=sqrt(sum(r*r));
      convergence=fabs(eval(0) - oldEigenvalue); 
      oldEigenvalue=eval(0);
       k++;
       if (k >= maxIteration)
           status=EXCEED_ITERATION;
       if (convergence < convergenceLimit_)
          status=CONVERGED;
      if (k == ncoord)
           status=FULL_MATRIX;
   };
   setStatus(status);
   iteration_ = k;
  convergence_=convergence;


if (k > 0) {
        Array<double, 2> Q=q(All, Range(0, k-1));
        Array<double, 2> _v(Q.rows(), vT.cols());
        _v=sum(Q(I, K)*vT(K, J), K);
        lowestEigenvalue=lambda(0);
        Array<double, 1> v=vT(All, 0);
        eigenvector.resize(ncoord);
        eigenvector=sum(Q(I, J)*v(J), J);
        }
        else { 
        assert(status == UNCOMPLETE);
        lowestEigenvalue=sum(uu(All,0)*q(All, 0));
        };
   eigenvalue_=lowestEigenvalue;
   eigenvector_=eigenvector;
eigenvalues << "The lowest eigenvalue from Davidson is: " << lowestEigenvalue <<" \n ";
  return status_;
}
myfile.close();
filemethod.close();
eigenvalues.close();

}




/** Compute product @f$ \mathrm H. \mathbf q @f$.
compute the product @f$ \mathrm H. \mathbf q @f$ using the matrix stored in #hessian_ .
@see #hessian_ and finiteH()
@ingroup realH
*/
blitz::Array<double, 1> Lanczos::realH(const blitz::Array<double, 1>& q)
{
    const int N=q.size();
    Array<double, 1> u(N);
    u=sum(  hessian_(I, J)*q(J), J  );
    return u;
}

///< Print final report.
void Lanczos::report()
{
    cout << "Lanczos::\n";
    string a, b;
    if (convergence() <= getConvergenceLimit()) {
        a="Has converged: ";
        b=" <= ";
    }
    else {
        a="Has not converged: ";
        b=" > ";
    };
    cout << a << convergence() << b << getConvergenceLimit() << '\n';
    if (iteration() >= getIterationLimit()) {
        a="Has exceeded iteration limit: ";
        b=" >= ";
    }
    else {
        a="Has not exceeded iteration limit: ";
        b=" < ";
    };
    cout << a << iteration() << b << getIterationLimit() << '\n';
    if (iteration() == (int)eigenvector_.size() ) {
        a="Has computed full Hesse matrix: ";
        b=" == ";
    }
    else {
        a="Has not computed full Hesse matrix: ";
        b=" < ";
    };
    cout << a << iteration() << b << eigenvector_.size() << '\n';
    cout << "Status is ";
    switch (status_) {
        case NOTHING:
            cout << "NOTHING\n";
            break;
        case UNCOMPLETE:
            cout << "UNCOMPLETE\n";
            break;
        case CONVERGED:
            cout << "CONVERGED\n";
            break;
        case EXCEED_ITERATION:
            cout << "EXCEED_ITERATION\n";
            break;
        case FULL_MATRIX:
            cout << "FULL_MATRIX\n";
            break;
        case STARTING:
            cout << "STARTING\n";
            break;
        case RUNNING:
            cout << "RUNNING\n";
            break;
        default:
            cerr << "Lanczos: status unknown\n";
            exit(EXIT_FAILURE);
    };
    cout << "The lowest eigenvalue is " << eigenvalue() << '\n';
    cout << flush;
}

/** Set the new status of the Lanczos.
Warning may be displayed depending on @a status.
@see #warnWhen_, warnWhen(), ....
*/
void Lanczos::setStatus(Status status)
{
    status_=status;
    if (status_ & warnWhen_ & CONVERGED)
        cerr << "Lanczos::CONVERGED\n";
    if (status_ & warnWhen_ & EXCEED_ITERATION)
        cerr << "Lanczos::EXCEED_ITERATION\n";
    if (status_ & warnWhen_ & FULL_MATRIX)
        cerr << "Lanczos::FULL_MATRIX\n";
    if (status_ & warnWhen_ & UNCOMPLETE)
        cerr << "Lanczos::UNCOMPLETE\n";
    if (status_ & warnWhen_ & RUNNING)
        cerr << "Lanczos::RUNNING\n";
    if (status_ & warnWhen_ & EXCEED_ITERATION)
        cerr << "Lanczos::STARTING\n";
}
