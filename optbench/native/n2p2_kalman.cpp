// Optbench wrapper around n2p2's standard Kalman filter.
//
// The network is the pyamff 10-7 tanh model. One Kalman update is one
// structure: the observation is the total-energy residual and the Jacobian
// is the total-energy derivative. The loss gradient used for the stop test
// is computed here and is not a Kalman update.

#include "KalmanFilter.h"

#include <gsl/gsl_rng.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kH1 = 10;
constexpr int kH2 = 7;
constexpr int kMaxFp = 32;

struct AtomProp {
    double h1[kH1];
    double h2[kH2];
    double dedg[kMaxFp];
};

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
    bool use_forces = false;
    double energy_coeff = 1.0;
    double force_coeff = 0.0;
    double force_weight = 1.0;
    int n_force_updates = 0;
    const double* fref = nullptr;
    const double* dg = nullptr;
    double force_rmse = -1.0;
    std::vector<AtomProp> props;
};

struct NativeOrder {
    std::mt19937_64 engine;
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

void save_activations(const double* x, int n_fp, const double* wblock, AtomProp* prop, double& energy)
{
    const int o_b1 = kH1 * n_fp;
    const int o_w2 = o_b1 + kH1;
    const int o_b2 = o_w2 + kH2 * kH1;
    const int o_w3 = o_b2 + kH2;
    double z1[kH1];
    double dz2_pre[kH2];
    for (int i = 0; i < kH1; ++i) {
        double sum = wblock[o_b1 + i];
        const double* row = wblock + i * n_fp;
        for (int j = 0; j < n_fp; ++j) {
            sum += row[j] * x[j];
        }
        z1[i] = sum;
        prop->h1[i] = std::tanh(sum);
    }
    for (int i = 0; i < kH2; ++i) {
        double sum = wblock[o_b2 + i];
        const double* row = wblock + o_w2 + i * kH1;
        for (int j = 0; j < kH1; ++j) {
            sum += row[j] * prop->h1[j];
        }
        dz2_pre[i] = sum;
        prop->h2[i] = std::tanh(sum);
    }
    double y = wblock[o_w3 + kH2];
    const double* w3 = wblock + o_w3;
    for (int j = 0; j < kH2; ++j) {
        y += w3[j] * prop->h2[j];
    }
    energy += y;
    double dz2[kH2];
    for (int i = 0; i < kH2; ++i) {
        dz2[i] = w3[i] * (1.0 - prop->h2[i] * prop->h2[i]);
    }
    double dz1[kH1];
    for (int j = 0; j < kH1; ++j) {
        double sum = 0.0;
        for (int i = 0; i < kH2; ++i) {
            sum += wblock[o_w2 + i * kH1 + j] * dz2[i];
        }
        dz1[j] = sum * (1.0 - prop->h1[j] * prop->h1[j]);
    }
    for (int j = 0; j < n_fp; ++j) {
        double sum = 0.0;
        for (int i = 0; i < kH1; ++i) {
            sum += wblock[i * n_fp + j] * dz1[i];
        }
        prop->dedg[j] = sum;
    }
    (void)z1;
    (void)dz2_pre;
}

void directional_into(const double* x, const double* direction, int n_fp, const double* wblock, const AtomProp& prop, double* hblock)
{
    const int o_b1 = kH1 * n_fp;
    const int o_w2 = o_b1 + kH1;
    const int o_b2 = o_w2 + kH2 * kH1;
    const int o_w3 = o_b2 + kH2;
    double sech1[kH1];
    double sech2[kH2];
    double dz1[kH1];
    double dh1[kH1];
    double dz2[kH2];
    double dh2[kH2];
    for (int i = 0; i < kH1; ++i) {
        sech1[i] = 1.0 - prop.h1[i] * prop.h1[i];
        double sum = 0.0;
        const double* row = wblock + i * n_fp;
        for (int j = 0; j < n_fp; ++j) {
            sum += row[j] * direction[j];
        }
        dz1[i] = sum;
        dh1[i] = sech1[i] * dz1[i];
    }
    for (int i = 0; i < kH2; ++i) {
        sech2[i] = 1.0 - prop.h2[i] * prop.h2[i];
        double sum = 0.0;
        const double* row = wblock + o_w2 + i * kH1;
        for (int j = 0; j < kH1; ++j) {
            sum += row[j] * dh1[j];
        }
        dz2[i] = sum;
        dh2[i] = sech2[i] * dz2[i];
    }
    const double* w3 = wblock + o_w3;
    double g_dz2[kH2];
    double g_z2[kH2];
    for (int i = 0; i < kH2; ++i) {
        const double g_sech2 = w3[i] * dz2[i];
        g_dz2[i] = w3[i] * sech2[i];
        const double g_h2 = g_sech2 * (-2.0 * prop.h2[i]);
        g_z2[i] = g_h2 * sech2[i];
    }
    double g_dz1[kH1];
    double g_z1[kH1];
    for (int j = 0; j < kH1; ++j) {
        double from_dz2 = 0.0;
        double from_z2 = 0.0;
        for (int i = 0; i < kH2; ++i) {
            const double weight = wblock[o_w2 + i * kH1 + j];
            from_dz2 += weight * g_dz2[i];
            from_z2 += weight * g_z2[i];
        }
        const double g_sech1 = from_dz2 * dz1[j];
        g_dz1[j] = from_dz2 * sech1[j];
        const double g_h1 = g_sech1 * (-2.0 * prop.h1[j]) + from_z2;
        g_z1[j] = g_h1 * sech1[j];
    }
    for (int i = 0; i < kH1; ++i) {
        double* row = hblock + i * n_fp;
        for (int j = 0; j < n_fp; ++j) {
            row[j] += g_dz1[i] * direction[j] + g_z1[i] * x[j];
        }
        hblock[o_b1 + i] += g_z1[i];
    }
    for (int i = 0; i < kH2; ++i) {
        double* row = hblock + o_w2 + i * kH1;
        for (int j = 0; j < kH1; ++j) {
            row[j] += g_dz2[i] * dh1[j] + g_z2[i] * prop.h1[j];
        }
        hblock[o_b2 + i] += g_z2[i];
        hblock[o_w3 + i] += dh2[i];
    }
}

const double* fingerprint_row(const Trainer* trainer, int image, int atom)
{
    const std::size_t offset = (static_cast<std::size_t>(image) * trainer->n_atoms + atom) * trainer->fp_stride;
    return trainer->fp.data() + offset;
}

const double* dg_at(const Trainer* trainer, int image, int center, int atom, int component)
{
    const std::size_t atoms = static_cast<std::size_t>(trainer->n_atoms);
    const std::size_t stride = static_cast<std::size_t>(trainer->fp_stride);
    const std::size_t offset = ((((static_cast<std::size_t>(image) * atoms + center) * atoms + atom) * 3 + component) * stride);
    return trainer->dg + offset;
}

double reference_force(const Trainer* trainer, int image, int atom, int component)
{
    const std::size_t offset = (static_cast<std::size_t>(image) * trainer->n_atoms + atom) * 3 + component;
    return trainer->fref[offset];
}

void forward_image(Trainer* trainer, int image, double& energy)
{
    std::fill(trainer->H.begin(), trainer->H.end(), 0.0);
    energy = 0.0;
    for (int atom = 0; atom < trainer->n_atoms; ++atom) {
        const int element = trainer->element[atom];
        const int off = trainer->w_off[element];
        const int count = trainer->n_fp[element];
        double unused = 0.0;
        save_activations(
            fingerprint_row(trainer, image, atom),
            count,
            trainer->w.data() + off,
            &trainer->props[atom],
            unused);
        accumulate_atom(
            fingerprint_row(trainer, image, atom),
            count,
            trainer->w.data() + off,
            trainer->H.data() + off,
            energy);
    }
}

void save_image(Trainer* trainer, int image)
{
    for (int atom = 0; atom < trainer->n_atoms; ++atom) {
        const int element = trainer->element[atom];
        double unused = 0.0;
        save_activations(
            fingerprint_row(trainer, image, atom),
            trainer->n_fp[element],
            trainer->w.data() + trainer->w_off[element],
            &trainer->props[atom],
            unused);
    }
}

double component_force(const Trainer* trainer, int image, int atom, int component)
{
    double value = 0.0;
    for (int center = 0; center < trainer->n_atoms; ++center) {
        const int count = trainer->n_fp[trainer->element[center]];
        const double* slice = dg_at(trainer, image, center, atom, component);
        double dot = 0.0;
        for (int feature = 0; feature < count; ++feature) {
            dot += slice[feature] * trainer->props[center].dedg[feature];
        }
        value -= dot;
    }
    return value;
}

void predict_forces(const Trainer* trainer, int image, std::vector<double>& forces)
{
    std::fill(forces.begin(), forces.end(), 0.0);
    for (int center = 0; center < trainer->n_atoms; ++center) {
        const int count = trainer->n_fp[trainer->element[center]];
        for (int atom = 0; atom < trainer->n_atoms; ++atom) {
            for (int component = 0; component < 3; ++component) {
                const double* slice = dg_at(trainer, image, center, atom, component);
                double dot = 0.0;
                for (int feature = 0; feature < count; ++feature) {
                    dot += slice[feature] * trainer->props[center].dedg[feature];
                }
                forces[atom * 3 + component] -= dot;
            }
        }
    }
}

void force_jacobian(Trainer* trainer, int image, int atom, int component)
{
    std::fill(trainer->H.begin(), trainer->H.end(), 0.0);
    for (int center = 0; center < trainer->n_atoms; ++center) {
        const int element = trainer->element[center];
        const int off = trainer->w_off[element];
        const int count = trainer->n_fp[element];
        directional_into(
            fingerprint_row(trainer, image, center),
            dg_at(trainer, image, center, atom, component),
            count,
            trainer->w.data() + off,
            trainer->props[center],
            trainer->H.data() + off);
    }
    for (double& value : trainer->H) {
        value = -value;
    }
}

int eval_forces(Trainer* trainer, double* grad, int n_grad, double* tensor_max, int n_tensors, double* loss, double* energy_rmse, int* finite, int* converged)
{
    std::fill(trainer->grad.begin(), trainer->grad.end(), 0.0);
    double energyloss = 0.0;
    double forceloss = 0.0;
    bool ok = true;
    std::vector<double> predicted(static_cast<std::size_t>(trainer->n_atoms) * 3, 0.0);
    std::vector<double> direction(static_cast<std::size_t>(trainer->fp_stride), 0.0);
    const double natoms = static_cast<double>(trainer->natoms);
    for (int image = 0; image < trainer->n_images && ok; ++image) {
        double energy = 0.0;
        forward_image(trainer, image, energy);
        if (!std::isfinite(energy)) {
            ok = false;
            break;
        }
        predict_forces(trainer, image, predicted);
        const double per_atom = (energy - trainer->eref[image]) / natoms;
        energyloss += per_atom * per_atom;
        const double energy_scale = trainer->energy_coeff * 2.0 * per_atom / natoms;
        for (int i = 0; i < trainer->n_weights; ++i) {
            trainer->grad[i] += energy_scale * trainer->H[i];
        }
        double image_force = 0.0;
        for (int slot = 0; slot < trainer->n_atoms * 3; ++slot) {
            const int atom = slot / 3;
            const int component = slot % 3;
            const double diff = predicted[slot] - reference_force(trainer, image, atom, component);
            if (!std::isfinite(diff)) {
                ok = false;
            }
            image_force += diff * diff;
        }
        forceloss += image_force / natoms / 3.0;
        const double force_scale = trainer->force_coeff * 2.0 / natoms / 3.0;
        for (int center = 0; center < trainer->n_atoms; ++center) {
            const int element = trainer->element[center];
            const int count = trainer->n_fp[element];
            const int off = trainer->w_off[element];
            std::fill(direction.begin(), direction.begin() + count, 0.0);
            for (int slot = 0; slot < trainer->n_atoms * 3; ++slot) {
                const int atom = slot / 3;
                const int component = slot % 3;
                const double diff = predicted[slot] - reference_force(trainer, image, atom, component);
                const double lam = force_scale * diff;
                const double* slice = dg_at(trainer, image, center, atom, component);
                for (int feature = 0; feature < count; ++feature) {
                    direction[feature] -= lam * slice[feature];
                }
            }
            directional_into(
                fingerprint_row(trainer, image, center),
                direction.data(),
                count,
                trainer->w.data() + off,
                trainer->props[center],
                trainer->grad.data() + off);
        }
    }
    for (double value : trainer->grad) {
        if (!std::isfinite(value)) {
            ok = false;
            break;
        }
    }
    if (grad != nullptr && n_grad == trainer->n_weights) {
        std::copy(trainer->grad.begin(), trainer->grad.end(), grad);
    }
    tensor_maxabs(trainer, tensor_max);
    bool stopped = ok;
    for (int i = 0; i < n_tensors; ++i) {
        if (!(tensor_max[i] < trainer->tol)) {
            stopped = false;
        }
    }
    const double images = static_cast<double>(trainer->n_images);
    trainer->force_rmse = std::sqrt(forceloss / images);
    if (loss != nullptr) {
        *loss = trainer->energy_coeff * energyloss + trainer->force_coeff * forceloss;
    }
    if (energy_rmse != nullptr) {
        *energy_rmse = std::sqrt(energyloss / images);
    }
    if (finite != nullptr) {
        *finite = ok ? 1 : 0;
    }
    if (converged != nullptr) {
        *converged = stopped ? 1 : 0;
    }
    return 0;
}

int observe_energy(Trainer* trainer, int image)
{
    double energy = 0.0;
    const double* base = trainer->fp.data() + static_cast<std::size_t>(image) * trainer->n_atoms * trainer->fp_stride;
    std::fill(trainer->H.begin(), trainer->H.end(), 0.0);
    energy = 0.0;
    for (int atom = 0; atom < trainer->n_atoms; ++atom) {
        const int element = trainer->element[atom];
        accumulate_atom(
            base + static_cast<std::size_t>(atom) * trainer->fp_stride,
            trainer->n_fp[element],
            trainer->w.data() + trainer->w_off[element],
            trainer->H.data() + trainer->w_off[element],
            energy);
    }
    if (!std::isfinite(energy)) {
        return 1;
    }
    trainer->xi = trainer->eref[image] - energy;
    trainer->filter->setError(&trainer->xi, 1);
    trainer->filter->setJacobian(trainer->H.data(), 1);
    trainer->filter->update();
    return 0;
}

int observe_force(Trainer* trainer, int image, int atom, int component)
{
    save_image(trainer, image);
    const double value = component_force(trainer, image, atom, component);
    if (!std::isfinite(value)) {
        return 1;
    }
    force_jacobian(trainer, image, atom, component);
    const double residual = reference_force(trainer, image, atom, component) - value;
    trainer->xi = trainer->force_weight * residual;
    for (double& entry : trainer->H) {
        entry *= trainer->force_weight;
    }
    trainer->filter->setError(&trainer->xi, 1);
    trainer->filter->setJacobian(trainer->H.data(), 1);
    trainer->filter->update();
    return 0;
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
    if (trainer->use_forces) {
        return eval_forces(trainer, grad, n_grad, tensor_max, n_tensors, loss, energy_rmse, finite, converged);
    }
    trainer->force_rmse = -1.0;
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

double n2p2_force_rmse(void* handle)
{
    return static_cast<Trainer*>(handle)->force_rmse;
}

int n2p2_enable_forces(
    void* handle,
    const double* forces,
    const double* dg,
    double energy_coeff,
    double force_coeff,
    double force_weight,
    int n_force_updates,
    double epsilon,
    double q0,
    double qtau,
    double qmin,
    double eta0,
    double etatau,
    double etamax)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (forces == nullptr || dg == nullptr || n_force_updates < 0) {
        return 1;
    }
    for (int count : trainer->n_fp) {
        if (count > kMaxFp) {
            return 1;
        }
    }
    trainer->use_forces = true;
    trainer->fref = forces;
    trainer->dg = dg;
    trainer->energy_coeff = energy_coeff;
    trainer->force_coeff = force_coeff;
    trainer->force_weight = force_weight;
    trainer->n_force_updates = n_force_updates;
    trainer->props.assign(static_cast<std::size_t>(trainer->n_atoms), AtomProp());
    const double updates = static_cast<double>(trainer->n_images + n_force_updates);
    if (!(updates > 0.0)) {
        return 1;
    }
    trainer->filter.reset(new nnp::KalmanFilter(
        static_cast<std::size_t>(trainer->n_weights), nnp::KalmanFilter::KT_STANDARD));
    trainer->filter->setState(trainer->w.data());
    trainer->filter->setSizeObservation(1);
    trainer->filter->setParametersStandard(
        epsilon, q0, qtau / updates, qmin, eta0, etatau / updates, etamax);
    return 0;
}

int n2p2_epoch_mixed(
    void* handle,
    const int* energy_order,
    int n_energy,
    const int* force_ids,
    int n_force,
    const int* schedule,
    int n_schedule)
{
    Trainer* trainer = static_cast<Trainer*>(handle);
    if (!trainer->use_forces || energy_order == nullptr || schedule == nullptr) {
        return 2;
    }
    if (n_force > 0 && force_ids == nullptr) {
        return 2;
    }
    int energy_steps = 0;
    int force_steps = 0;
    for (int step = 0; step < n_schedule; ++step) {
        if (schedule[step] == 0) {
            ++energy_steps;
        } else if (schedule[step] == 1) {
            ++force_steps;
        } else {
            return 2;
        }
    }
    if (energy_steps != n_energy || force_steps != n_force) {
        return 2;
    }
    const int components = trainer->n_atoms * 3;
    try {
        int energy_cursor = 0;
        int force_cursor = 0;
        for (int step = 0; step < n_schedule; ++step) {
            int status = 0;
            if (schedule[step] == 0) {
                const int image = energy_order[energy_cursor++];
                if (image < 0 || image >= trainer->n_images) {
                    return 2;
                }
                status = observe_energy(trainer, image);
            } else {
                const int packed = force_ids[force_cursor++];
                if (packed < 0) {
                    return 2;
                }
                const int image = packed / components;
                const int slot = packed % components;
                const int atom = slot / 3;
                const int component = slot % 3;
                if (image < 0 || image >= trainer->n_images || atom < 0 || atom >= trainer->n_atoms) {
                    return 2;
                }
                status = observe_force(trainer, image, atom, component);
            }
            if (status != 0) {
                return status;
            }
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

void* n2p2_rng_create(unsigned long seed)
{
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    if (rng == nullptr) {
        return nullptr;
    }
    gsl_rng_set(rng, seed);
    // Rank 0 with one process: the documented seed, then one draw for the
    // global taus generator. No per-rank reseeding.
    const unsigned long seed_global = gsl_rng_get(rng);
    gsl_rng* rng_global = gsl_rng_alloc(gsl_rng_taus);
    if (rng_global == nullptr) {
        gsl_rng_free(rng);
        return nullptr;
    }
    gsl_rng_set(rng_global, seed_global);
    NativeOrder* order = new NativeOrder();
    // Training.cpp seeds rngGlobalNew first, then rngNew. Energy shuffles use rngNew.
    (void)gsl_rng_get(rng_global);
    order->engine.seed(gsl_rng_get(rng));
    gsl_rng_free(rng_global);
    gsl_rng_free(rng);
    return order;
}

void n2p2_rng_destroy(void* handle)
{
    delete static_cast<NativeOrder*>(handle);
}

int n2p2_rng_shuffle(void* handle, int* values, int n)
{
    if (handle == nullptr || values == nullptr || n < 0) {
        return 1;
    }
    NativeOrder* order = static_cast<NativeOrder*>(handle);
    std::vector<int> data(values, values + n);
    std::shuffle(data.begin(), data.end(), order->engine);
    std::copy(data.begin(), data.end(), values);
    return 0;
}

}  // extern "C"
