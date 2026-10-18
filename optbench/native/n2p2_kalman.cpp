// Optbench wrapper around n2p2's standard Kalman filter.
//
// The network is the pyamff 10-7 tanh model. One Kalman update is one
// structure: the observation is the total-energy residual and the Jacobian
// is the total-energy derivative. The loss gradient used for the stop test
// is computed here and is not a Kalman update.

#include "KalmanFilter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kH1 = 10;
constexpr int kH2 = 7;

struct Trainer {
    int n_images = 0;
    int n_atoms = 0;
    int n_elements = 0;
    int fp_stride = 0;
    int n_weights = 0;
    int natoms = 0;
    double tol = 0.0;
    std::vector<int> element;
    std::vector<int> n_fp;
    std::vector<int> w_off;
    std::vector<double> fp;
    std::vector<double> eref;
    std::vector<double> w;
    std::vector<double> H;
    std::vector<double> grad;
    double xi = 0.0;
    std::unique_ptr<nnp::KalmanFilter> filter;
};

int element_width(int n_fp)
{
    return kH1 * n_fp + kH1 + kH2 * kH1 + kH2 + kH2 + 1;
}

void accumulate_atom(const double* x, int n_fp, const double* wblock, double* hblock, double& energy)
{
    const int o_b1 = kH1 * n_fp;
    const int o_w2 = o_b1 + kH1;
    const int o_b2 = o_w2 + kH2 * kH1;
    const int o_w3 = o_b2 + kH2;
    const int o_b3 = o_w3 + kH2;

    double h1[kH1];
    double h2[kH2];
    for (int i = 0; i < kH1; ++i) {
        double sum = wblock[o_b1 + i];
        const double* row = wblock + i * n_fp;
        for (int j = 0; j < n_fp; ++j) {
            sum += row[j] * x[j];
        }
        h1[i] = std::tanh(sum);
    }
    for (int i = 0; i < kH2; ++i) {
        double sum = wblock[o_b2 + i];
        const double* row = wblock + o_w2 + i * kH1;
        for (int j = 0; j < kH1; ++j) {
            sum += row[j] * h1[j];
        }
        h2[i] = std::tanh(sum);
    }
    double y = wblock[o_b3];
    const double* w3 = wblock + o_w3;
    for (int j = 0; j < kH2; ++j) {
        y += w3[j] * h2[j];
    }
    energy += y;

    for (int j = 0; j < kH2; ++j) {
        hblock[o_w3 + j] += h2[j];
    }
    hblock[o_b3] += 1.0;

    double dz2[kH2];
    for (int i = 0; i < kH2; ++i) {
        dz2[i] = w3[i] * (1.0 - h2[i] * h2[i]);
    }
    for (int i = 0; i < kH2; ++i) {
        hblock[o_b2 + i] += dz2[i];
        double* row = hblock + o_w2 + i * kH1;
        for (int j = 0; j < kH1; ++j) {
            row[j] += dz2[i] * h1[j];
        }
    }

    double dz1[kH1];
    for (int j = 0; j < kH1; ++j) {
        double sum = 0.0;
        for (int i = 0; i < kH2; ++i) {
            sum += wblock[o_w2 + i * kH1 + j] * dz2[i];
        }
        dz1[j] = sum * (1.0 - h1[j] * h1[j]);
    }
    for (int i = 0; i < kH1; ++i) {
        hblock[o_b1 + i] += dz1[i];
        double* row = hblock + i * n_fp;
        const double scale = dz1[i];
        for (int j = 0; j < n_fp; ++j) {
            row[j] += scale * x[j];
        }
    }
}

double image_energy(Trainer* trainer, int image)
{
    std::fill(trainer->H.begin(), trainer->H.end(), 0.0);
    double energy = 0.0;
    const double* base = trainer->fp.data() + static_cast<std::size_t>(image) * trainer->n_atoms * trainer->fp_stride;
    for (int atom = 0; atom < trainer->n_atoms; ++atom) {
        const int element = trainer->element[atom];
        const int off = trainer->w_off[element];
        accumulate_atom(
            base + static_cast<std::size_t>(atom) * trainer->fp_stride,
            trainer->n_fp[element],
            trainer->w.data() + off,
            trainer->H.data() + off,
            energy);
    }
    return energy;
}

void tensor_maxabs(const Trainer* trainer, double* out)
{
    int slot = 0;
    for (int element = 0; element < trainer->n_elements; ++element) {
        const int n_fp = trainer->n_fp[element];
        const int off = trainer->w_off[element];
        const int sizes[6] = {kH1 * n_fp, kH1, kH2 * kH1, kH2, kH2, 1};
        int cursor = off;
        for (int size : sizes) {
            double peak = 0.0;
            for (int i = 0; i < size; ++i) {
                peak = std::max(peak, std::fabs(trainer->grad[cursor + i]));
            }
            out[slot++] = peak;
            cursor += size;
        }
    }
}

}  // namespace

extern "C" {

void* n2p2_create(
    int n_images,
    int n_atoms,
    int n_elements,
    int fp_stride,
    int natoms,
    double tol,
    const int* element_of_atom,
    const int* n_fp,
    const double* fingerprints,
    const double* energy_ref,
    double epsilon,
    double q0,
    double qtau,
    double qmin,
    double eta0,
    double etatau,
    double etamax)
{
    if (n_images < 1 || n_atoms < 1 || n_elements < 1 || fp_stride < 1 || natoms < 1) {
        return nullptr;
    }
    if (element_of_atom == nullptr || n_fp == nullptr || fingerprints == nullptr || energy_ref == nullptr) {
        return nullptr;
    }
    try {
        std::unique_ptr<Trainer> trainer(new Trainer());
        trainer->n_images = n_images;
        trainer->n_atoms = n_atoms;
        trainer->n_elements = n_elements;
        trainer->fp_stride = fp_stride;
        trainer->natoms = natoms;
        trainer->tol = tol;
        trainer->element.assign(element_of_atom, element_of_atom + n_atoms);
        trainer->n_fp.assign(n_fp, n_fp + n_elements);
        trainer->w_off.resize(n_elements);
        int width = 0;
        for (int element = 0; element < n_elements; ++element) {
            if (trainer->n_fp[element] < 1 || trainer->n_fp[element] > fp_stride) {
                return nullptr;
            }
            trainer->w_off[element] = width;
            width += element_width(trainer->n_fp[element]);
        }
        for (int atom = 0; atom < n_atoms; ++atom) {
            if (trainer->element[atom] < 0 || trainer->element[atom] >= n_elements) {
                return nullptr;
            }
        }
        trainer->n_weights = width;
        const std::size_t fp_count = static_cast<std::size_t>(n_images) * n_atoms * fp_stride;
        trainer->fp.assign(fingerprints, fingerprints + fp_count);
        trainer->eref.assign(energy_ref, energy_ref + n_images);
        trainer->w.assign(width, 0.0);
        trainer->H.assign(width, 0.0);
        trainer->grad.assign(width, 0.0);
        trainer->filter.reset(new nnp::KalmanFilter(
            static_cast<std::size_t>(width), nnp::KalmanFilter::KT_STANDARD));
        trainer->filter->setState(trainer->w.data());
        trainer->filter->setSizeObservation(1);
        // n2p2 divides the published taus by the projected updates in one epoch.
        const double updates = static_cast<double>(n_images);
        trainer->filter->setParametersStandard(
            epsilon, q0, qtau / updates, qmin, eta0, etatau / updates, etamax);
        return trainer.release();
    } catch (const std::exception&) {
        return nullptr;
    }
}

void n2p2_destroy(void* handle)
{
    delete static_cast<Trainer*>(handle);
}

int n2p2_n_weights(void* handle)
{
    return static_cast<Trainer*>(handle)->n_weights;
}

int n2p2_set_weights(void* handle, const double* weights, int n)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (weights == nullptr || n != trainer->n_weights) {
        return 1;
    }
    std::copy(weights, weights + n, trainer->w.begin());
    return 0;
}

int n2p2_get_weights(void* handle, double* weights, int n)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (weights == nullptr || n != trainer->n_weights) {
        return 1;
    }
    std::copy(trainer->w.begin(), trainer->w.end(), weights);
    return 0;
}

int n2p2_eval(void* handle, double* grad, int n_grad, double* tensor_max, int n_tensors, double* loss, double* energy_rmse, int* finite, int* converged)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (n_grad != trainer->n_weights || n_tensors != trainer->n_elements * 6) {
        return 1;
    }
    std::fill(trainer->grad.begin(), trainer->grad.end(), 0.0);
    double total = 0.0;
    bool ok = true;
    for (int image = 0; image < trainer->n_images; ++image) {
        const double energy = image_energy(trainer, image);
        if (!std::isfinite(energy)) {
            ok = false;
            break;
        }
        const double per_atom = (energy - trainer->eref[image]) / static_cast<double>(trainer->natoms);
        total += per_atom * per_atom;
        const double scale = 2.0 * per_atom / static_cast<double>(trainer->natoms);
        for (int i = 0; i < trainer->n_weights; ++i) {
            trainer->grad[i] += scale * trainer->H[i];
        }
    }
    for (double value : trainer->grad) {
        if (!std::isfinite(value)) {
            ok = false;
            break;
        }
    }
    if (grad != nullptr) {
        std::copy(trainer->grad.begin(), trainer->grad.end(), grad);
    }
    tensor_maxabs(trainer, tensor_max);
    bool stopped = ok;
    for (int i = 0; i < n_tensors; ++i) {
        if (!(tensor_max[i] < trainer->tol)) {
            stopped = false;
        }
    }
    if (loss != nullptr) {
        *loss = total;
    }
    if (energy_rmse != nullptr) {
        *energy_rmse = std::sqrt(total / static_cast<double>(trainer->n_images));
    }
    if (finite != nullptr) {
        *finite = ok ? 1 : 0;
    }
    if (converged != nullptr) {
        *converged = stopped ? 1 : 0;
    }
    return 0;
}

int n2p2_epoch(void* handle, const int* order, int n_order)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (n_order != trainer->n_images) {
        return 2;
    }
    try {
        for (int step = 0; step < n_order; ++step) {
            const int image = order == nullptr ? step : order[step];
            if (image < 0 || image >= trainer->n_images) {
                return 2;
            }
            const double energy = image_energy(trainer, image);
            if (!std::isfinite(energy)) {
                return 1;
            }
            trainer->xi = trainer->eref[image] - energy;
            trainer->filter->setError(&trainer->xi, 1);
            trainer->filter->setJacobian(trainer->H.data(), 1);
            trainer->filter->update();
        }
    } catch (const std::exception&) {
        return 1;
    }
    for (double value : trainer->w) {
        if (!std::isfinite(value)) {
            return 1;
        }
    }
    return 0;
}

std::size_t n2p2_updates(void* handle)
{
    return static_cast<Trainer*>(handle)->filter->getNumUpdates();
}

}  // extern "C"
