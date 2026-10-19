/* r2549 client/potentials/Morse/Morse.cpp, without the Potential base.

   g++ -O2 of this class follows the client's Morse::force on the
   one-image Pt-island band. A free-function transcription does not.
*/
#include <math.h>

class MorseEon {
  public:
    MorseEon() { setParameters(0.7102, 1.6047, 2.8970, 9.5); }

    void setParameters(double De, double a, double re, double cutoff) {
        re_ = re;
        De_ = De;
        a_ = a;
        cutoff_ = cutoff;
        double f;
        morse(cutoff, energyCutoff_, f);
    }

    void force(long N, const double *R, double *F, double *U, const double *box) {
        double diffR = 0, diffRX, diffRY, diffRZ;
        *U = 0;
        for (int i = 0; i < N; i++) {
            F[3 * i] = 0;
            F[3 * i + 1] = 0;
            F[3 * i + 2] = 0;
        }
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                diffRX = R[3 * i] - R[3 * j];
                diffRY = R[3 * i + 1] - R[3 * j + 1];
                diffRZ = R[3 * i + 2] - R[3 * j + 2];
                diffRX = diffRX - box[0] * floor(diffRX / box[0] + 0.5);
                diffRY = diffRY - box[4] * floor(diffRY / box[4] + 0.5);
                diffRZ = diffRZ - box[8] * floor(diffRZ / box[8] + 0.5);
                diffR = sqrt(diffRX * diffRX + diffRY * diffRY + diffRZ * diffRZ);
                if (diffR < cutoff_) {
                    double force, energy;
                    morse(diffR, energy, force);
                    *U += energy;
                    F[3 * i] += force * diffRX / diffR;
                    F[3 * i + 1] += force * diffRY / diffR;
                    F[3 * i + 2] += force * diffRZ / diffR;
                    F[3 * j] -= force * diffRX / diffR;
                    F[3 * j + 1] -= force * diffRY / diffR;
                    F[3 * j + 2] -= force * diffRZ / diffR;
                    *U -= energyCutoff_;
                }
            }
        }
    }

  private:
    void morse(double r, double &energy, double &force) {
        double const d = 1 - exp(-a_ * (r - re_));
        energy = De_ * d * d - De_;
        force = 2 * De_ * d * (d - 1) * a_;
    }

    double re_;
    double De_;
    double cutoff_;
    double a_;
    double energyCutoff_;
};

extern "C" void morse_eon_force(
    int atoms,
    const double *pos,
    const double *box,
    double *energy_out,
    double *forces
) {
    static MorseEon morse;
    morse.force(atoms, pos, forces, energy_out, box);
}
