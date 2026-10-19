# Optax force rows and the PyAMFF python fingerprint derivative

The ranked Optax force rows on `mlopt/emt-pt38` and `mlopt/C2H4` differentiate the Behler-Parrinello fingerprints with `jacrev`, the neighbor list held fixed, and the cutoff taken from `fpParas.dat`. This note compares that derivative with PyAMFF `loop_FPs` at commit `8037eb98e2288c132c112aa6c2032f00cd5c5277`. The checkout used for the comparison is `/tmp/pyamff` at that commit. Torch is not installed. The script below stubs it so the python fingerprint modules can be imported. Numpy in the harness virtualenv is 2.5.4.

## Where the python zip is

`NeighborLists.loop_images` in `pyamff/neighborlist.py` builds an ASE neighbor list with cutoff 3.0 Å on each atom, `bothways`, no self pairs, and skin 0. Lines 140-143 sort the neighbor indices and store the periodic offsets in the order ASE returned them:

```python
indices,offsets = nl.get_neighbors(i)
self.nlist[a_index].append(np.sort(indices))
self.offsets[a_index].append(offsets)
```

`loop_FPs` in `pyamff/fingerprints/behlerParrinello.py` lines 786-788 zips those sorted indices and symbols with the unsorted offsets. A cross derivative is stored only when `noffset.all() == 0`:

```python
for nindex, nsymbol, noffset in zip(selfneighborindices, selfneighborsymbols, selfneighboroffsets):
    if noffset.all() == 0:
```

On numpy 2.5.4, `ndarray.all() == 0` is true for `[0, 0, 0]`, `[1, 0, 0]`, `[0, 1, 0]`, and `[1, 1, 0]`. It is false for `[1, -1, 1]` and `[1, 1, 1]`. An in-cell neighbor passes. A neighbor whose cell offset is nonzero in every component fails. The failing slot is the slot in the unsorted offset array, which is not the slot of the sorted index.

`fingerprintsWrapper.py` lines 341-344 catch a missing key and continue, so a dropped block contributes nothing to the force.

`BehlerParrinello.__init__` sets `r_cut=6.0` at line 38. `set` at lines 51-66 copies eta, Rs, zeta, and lambda from the fingerprint parameter and does not copy `g.rcut`. The python calculator therefore uses 6.0 Å unless something else assigns `r_cut`. The Pt `fpParas.dat` used by these rows has Rc 5.0. The C2H4 file has Rc 6.5. `calculate_G2` initializes `values = 0.` at line 236. `calculate_dG2` assigns `values` only after a matching triplet, at line 435, and returns it at line 438. A center with no matching triplet raises `UnboundLocalError`.

The archived training configs take the Fortran engine. `pyamff/config.yaml` lines 117-119 default `fp_engine` to Fortran. Each C2H4 `config.ini` sets `fp_engine = Fortran` at line 20. The six emt-pt38 `config.ini` files omit `fp_engine`, so they keep that default. `main.py` lines 488-510 call `Fingerprints.calcFPs`. That method calls `fmodules.fpcalc.calcfps` at `pyamff/fingerprints/fingerprints.py` line 77. The Fortran reader stores the file Rc at `pyamff/fortran/fingerprints.f90` lines 161 and 205. The python branch is `main.py` lines 518-538, which calls the neighbor list with `fortran=False`. The Fortran neighbor path in `neighborlist.py` lines 111-114 stores a zero offset for every neighbor, so it does not execute this zip. `gfortran` is not on this machine, and numpy 2.5.4 does not provide `numpy.distutils`. The Fortran extension was not built. The numbers below are the python `loop_FPs` path.

## Four-atom periodic cell

The cell is a 10 Å cube with periodic boundaries and four Pt atoms:

```text
[8.6964, 7.8873, 1.5741]
[8.8873, 4.1428, 8.8971]
[1.0862, 9.5476, 1.2094]
[7.3070, 7.0155, 7.4174]
```

The fingerprint file is `archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat`. The script at the bottom of this note computes `loop_FPs` and the Optax `jacrev` derivative, then a central finite difference with step `1e-5`. One finite difference rebuilds the PyAMFF neighbor list at the shifted coordinates. The other holds the Optax neighbor list fixed. Run it from the repository root with `uv run python` after placing that script on disk.

Atom 1 has indices `[3, 0, 2]` and offsets `[[0, 0, 0], [0, 0, 1], [1, -1, 1]]`. The gate accepts the first two offsets. Zipping the sorted indices with those unsorted offsets stores neighbors 0 and 2. Pairing each offset with its own index stores neighbors 3 and 0. Atom 2 has offsets `[[-1, 0, 0], [-1, 1, -1], [-1, 0, -1]]` and the gate drops the middle neighbor on both pairings. Two of the 16 center-wrt blocks are absent from the PyAMFF dictionary.

With the file Rc copied onto `r_cut`, so both calculators use 5.0, the maximum absolute Pt differences are:

| Comparison | Maximum absolute difference |
| --- | --- |
| PyAMFF G versus jacrev G | 5.273559366969494e-16 |
| PyAMFF stored dG versus jacrev | 0.19069594465522602 |
| PyAMFF stored dG versus the finite difference of rebuilt PyAMFF G | 0.19069594464626857 |
| jacrev versus the finite difference of rebuilt PyAMFF G | 2.3650081892867547e-11 |
| jacrev versus the finite difference of the jacrev G, neighbors fixed | 2.061160964128561e-11 |
| PyAMFF stored dG versus the finite difference of the jacrev G | 0.19069594465459525 |

On the blocks `loop_FPs` stores, the two derivatives differ by at most 4.440892098500626e-16. On the two dropped blocks the jacrev derivative reaches 0.19069594465522602. The finite-difference match is the Optax `jacrev` derivative.

With PyAMFF left at its constructor cutoff of 6.0 and the Optax fingerprints left at the file Rc of 5.0, the fingerprints differ by 0.38255909376400277. The stored PyAMFF derivative differs from the finite difference of the rebuilt PyAMFF fingerprint by 0.19811046652329353. The jacrev derivative differs from the finite difference of its own fingerprint by 2.061160964128561e-11. The ranked rows use the file Rc.

## Training trajectories

The neighbor census uses the same ASE list as `loop_images`: cutoff 3.0 Å on each atom, `bothways`, skin 0. A fail slot is an offset for which `offset.all() == 0` is false.

`inputs/sha256/2ff6a9ab51619942c08dd9239b9a2be57552481808ced9de9e703a03ba2da402/pt_md.traj` has 500 images, 38 atoms, and cell lengths `[24.0, 24.0, 24.0]`. It has 451010 neighbor slots, 0 slots with a nonzero offset, 17739 unsorted lists, 0 fail slots, and 0 lists whose zip-included indices differ from the offset-aligned indices.

`inputs/sha256/0d82bbb8892bcc8871bb805c1a84de5e429aaa90c254a510fad0dad96ff36d4d/C2H4_md.traj` has 570 images, 6 atoms, and cell lengths `[13.0, 13.0, 13.0]`. It has 17100 neighbor slots, 670 slots with a nonzero offset, 67 images with a nonzero offset, 2280 unsorted lists, 0 fail slots, and 0 lists whose zip-included indices differ from the offset-aligned indices. No atom index appears only in a failing slot.

On both trajectories the misaligned zip stores the same neighbor indices as an offset-aligned gate.

The derivative comparison below copies the file Rc onto `r_cut` before calling `loop_FPs`, and compares the raw, unnormalized derivatives with `jacrev`. For C2H4, `calculate_dG2` was wrapped so an `UnboundLocalError` returns 0.0. That is the value `calculate_G2` uses when no triplet matches. The unwrapped call was counted separately.

On all 570 C2H4 images the unwrapped `loop_FPs` raises `UnboundLocalError`. With the wrapper, the maximum absolute differences are 2.220446049250313e-15 in G and 2.1649348980190553e-15 in dG for C, and 4.440892098500626e-15 in G and 1.9984014443252818e-15 in dG for H.

On all 500 Pt images `loop_FPs` returns, and the `calculate_dG2` wrapper catches no `UnboundLocalError`. The maximum absolute differences are 9.769962616701378e-15 in G and 5.551115123125783e-15 in dG. Image 130 has the largest derivative difference.

## Force loss

The training loss is `energy_se + 0.1 * force_se`. `energy_se` is the sum over images of `((E - Eref) / n_atoms)^2`. `force_se` is the sum of `||F - Fref||^2 / n_atoms / 3`. The force is `F = -dE/dG · dG/dR`, and `dG/dR` is the normalized fingerprint derivative. The evaluation uses `jax_enable_x64`, seed `20261018`, starts 0 through 49, hidden layers `(10, 7)`, energy coefficient 1, and force coefficient 0.1. Initial weights are `uniform[-1, 1)`. The loss quoted here is the vmapped harness value. Reconstructing it from the reported RMSE squares differs from that value by at most 4.547473508864641e-13 on emt-pt38 and 7.275957614183426e-12 on C2H4.

emt-pt38, 500 images, 38 atoms, start 0: `energy_se` 381.6578509977635, `force_se` 140.93287994529894, loss 395.7511389922934. Over the 50 starts, `force_se` runs from 86.34187251124247 to 5245.50997881496 with mean 721.592379597496, and the loss runs from 34.382655038139376 to 6258.733667257354 with mean 825.5864229359432.

C2H4, 570 images, 6 atoms, start 0: `energy_se` 16420.767195994245, `force_se` 43018.85641449632, loss 20722.65283744388. Over the 50 starts, `force_se` runs from 13975.750089578632 to 428877.69405860885 with mean 88207.32016278425, and the loss runs from 9998.938141468712 to 62457.7976577654 with mean 23808.665164230275.

Those values are the finite-difference derivative. The python `loop_FPs` loss on C2H4 does not exist: every image raises `UnboundLocalError` before a derivative tensor is returned. With the undefined angular derivative replaced by 0, the C2H4 raw tensors agree to the maxima above. On emt-pt38, `loop_FPs` returns a raw tensor that agrees with `jacrev` to the maxima above.

The loss bound treats that agreement as a perturbation of the normalized derivative. Each raw fingerprint component may move by the measured maximum absolute G difference, which widens its range by twice that amount and changes the scale `2 / span`. Each raw derivative component may move by the measured maximum absolute dG difference. The resulting per-component ceiling on the normalized derivative is 6.6185501224261e-14 for Pt, 3.4062463628535186e-12 for C, and 7.683661377392523e-13 for H. Moving every normalized component within its ceiling changes `force_se` by at most 1.5045756029250555e-08 on emt-pt38 and 4.3231011368265585e-07 on C2H4, and changes the loss by at most 1.5045756029250556e-09 and 4.3231011368265585e-08. The Pt ceiling uses all 500 images. The C2H4 ceiling uses all 570 images and the zero substitution for the undefined `calculate_dG2`.

## What is on the board

The six ranked force templates stay on the finite-difference derivative. That is the main row. A second campaign that copied python `loop_FPs` would, on these two trajectories, see the same stored neighbor blocks. On emt-pt38 the returned derivative agrees with `jacrev` to the tolerance above, so the force losses agree to the bound above. On C2H4 the unwrapped call raises, so it has no force loss to train. The archived configs select the Fortran engine, and that derivative was not executed here. No paper-comparable force campaign was added.

The Note on each problem page states which derivative the ranked force rows use. The force template notes point here.

## Script

```python
"""Compare PyAMFF loop_FPs derivatives with finite differences and Optax jacrev."""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/pyamff")
sys.path.insert(0, str(ROOT))


class Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return Dummy()

    def __getattr__(self, name):
        return Dummy()


class Torch(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Dummy()


torch = Torch("torch")
nn = Torch("torch.nn")


class Module:
    def __init__(self, *a, **k):
        pass


nn.Module = Module
torch.nn = nn
torch.device = lambda *a, **k: "cpu"
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn
pkg = types.ModuleType("pyamff")
pkg.__path__ = [str(ROOT / "pyamff")]
sys.modules["pyamff"] = pkg
for sub in ("fingerprints", "utilities", "mlModels"):
    mod = types.ModuleType(f"pyamff.{sub}")
    mod.__path__ = [str(ROOT / "pyamff" / sub)]
    sys.modules[f"pyamff.{sub}"] = mod

from pyamff.config import G1, G2
from pyamff.fingerprints.behlerParrinello import BehlerParrinello, loop_FPs
from pyamff.neighborlist import NeighborLists
from ase import Atoms
from ase.neighborlist import NeighborList

sys.path.insert(0, "/workspace")
from optbench.ml_bp import fingerprints_from_arrays, parse_fp_file


def paras_from_file(path):
    order, specs = parse_fp_file(Path(path))
    gparas = {}
    for element, rows in specs.items():
        built = []
        for row in rows:
            if row["kind"] == "G1":
                built.append(G1(neighbor=row["neighbor"], eta=row["eta"], Rs=row["rs"], rcut=row["rc"], center=element))
            else:
                built.append(
                    G2(
                        neighbor1=row["neighbor1"],
                        neighbor2=row["neighbor2"],
                        eta=row["eta"],
                        zeta=row["zeta"],
                        lambda_=row["lam"],
                        thetas=0.0,
                        rcut=row["rc"],
                        center=element,
                    )
                )
        gparas[element] = built
    nfps = {element: len(rows) for element, rows in gparas.items()}
    return order, specs, gparas, nfps


def pyamff_bundle(atoms, gparas, nfps):
    images = {0: atoms}
    nl = NeighborLists(cutoff=6.0)
    nl.calculate(images, fortran=False)
    fps, primes = loop_FPs(0, nl, nfps, gparas, fortran=False, fNN=False)
    natoms = len(atoms)
    symbols = list(atoms.get_chemical_symbols())
    width = {element: nfps[element] for element in nfps}
    dg = {element: np.zeros((natoms, width[element], natoms, 3)) for element in nfps}
    present = {element: np.zeros((natoms, natoms), dtype=bool) for element in nfps}
    for (wrt, wsym, center, csym, direction), values in primes.items():
        dg[csym][center, :, wrt, direction] = np.asarray(values, dtype=float)
        present[csym][center, wrt] = True
    values = {}
    for center, (symbol, fp) in enumerate(fps):
        values.setdefault(symbol, []).append((center, np.asarray(fp, dtype=float)))
    g = {}
    for element, rows in values.items():
        arr = np.zeros((natoms, nfps[element]))
        for center, fp in rows:
            arr[center] = fp
        g[element] = arr
    return nl, g, dg, present


def passes(offset):
    return bool(np.asarray(offset).all() == 0)


def report_neighbors(atoms):
    nl = NeighborList(cutoffs=[3.0] * len(atoms), sorted=False, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    print("cell", np.asarray(atoms.cell.lengths()), "pbc", atoms.pbc.tolist(), "n", len(atoms))
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        indices = np.asarray(indices, dtype=int)
        offsets = np.asarray(offsets, dtype=int)
        if indices.size == 0:
            continue
        order = np.argsort(indices, kind="mergesort")
        raw_pass = np.array([passes(off) for off in offsets])
        zip_ids = indices[order][raw_pass]
        aligned_ids = indices[raw_pass]
        unsorted = not np.array_equal(order, np.arange(indices.size))
        if unsorted or (not raw_pass.all()):
            print(
                f"  atom {i} unsorted {unsorted} indices {indices.tolist()} "
                f"offsets {offsets.tolist()} pass {raw_pass.tolist()} "
                f"zip_included {zip_ids.tolist()} aligned_included {aligned_ids.tolist()}"
            )


def fd_pyamff_g(atoms, gparas, nfps, eps=1e-5):
    """Finite difference of loop_FPs fingerprint values, neighbor list rebuilt."""
    elements = list(nfps)
    fd = {element: np.zeros((len(atoms), nfps[element], len(atoms), 3)) for element in elements}
    pos = atoms.get_positions().copy()
    for atom in range(len(atoms)):
        for direction in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            pp = pos.copy()
            pm = pos.copy()
            pp[atom, direction] += eps
            pm[atom, direction] -= eps
            plus.set_positions(pp)
            minus.set_positions(pm)
            _, gp, _, _ = pyamff_bundle(plus, gparas, nfps)
            _, gm, _, _ = pyamff_bundle(minus, gparas, nfps)
            for element in elements:
                fd[element][:, :, atom, direction] = (gp[element] - gm[element]) / (2 * eps)
    return fd


def our_raw(atoms, fp_file):
    order, specs = parse_fp_file(Path(fp_file))
    symbols = list(atoms.get_chemical_symbols())
    from optbench.ml_bp import _neighbor_index

    neigh = _neighbor_index(atoms, 6.0)
    positions = atoms.get_positions()[None, :, :].astype(np.float64)
    lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)[None, :]
    pbc = np.asarray(atoms.pbc, dtype=bool)
    neighbors = neigh[None, :, :]
    prepared = fingerprints_from_arrays(positions, lengths, pbc, symbols, neighbors, order, specs, normalize=False)
    out_g = {}
    out_dg = {}
    for element in prepared["elements"]:
        centers = prepared["centers"][element]
        g = np.zeros((len(atoms), prepared["n_fp"][element]))
        dg = np.zeros((len(atoms), prepared["n_fp"][element], len(atoms), 3))
        g[centers] = prepared["fps"][element][0]
        dg[centers] = prepared["dgdr"][element][0]
        out_g[element] = g
        out_dg[element] = dg
    return out_g, out_dg


def max_err(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(np.max(np.abs(diff))) if diff.size else 0.0


def compare(label, atoms, fp_file):
    print("\n==", label)
    report_neighbors(atoms)
    order, specs, gparas, nfps = paras_from_file(fp_file)
    bp = BehlerParrinello()
    bp.set(gparas[order[0]][0])
    print("after set r_cut", bp.r_cut, "file rcut", gparas[order[0]][0].rcut)
    g, dg, present = pyamff_bundle(atoms, gparas, nfps)[1:]
    our_g, our_dg = our_raw(atoms, fp_file)
    print("G pyamff vs our max abs", {el: max_err(g[el], our_g[el]) for el in g})
    print("dG pyamff vs our max abs", {el: max_err(dg[el], our_dg[el]) for el in dg})
    missing = {el: int((~present[el]).sum()) for el in present}
    print("missing center-wrt blocks", missing, "of", {el: int(present[el].size) for el in present})
    if len(atoms) <= 6:
        fd = fd_pyamff_g(atoms, gparas, nfps)
        print("PyAMFF analytic vs FD of rebuilt PyAMFF G", {el: max_err(dg[el], fd[el]) for el in dg})
        print("our jacrev vs that FD", {el: max_err(our_dg[el], fd[el]) for el in dg})
        from optbench.ml_bp import _neighbor_index

        neigh = _neighbor_index(atoms, 6.0)
        symbols = list(atoms.get_chemical_symbols())
        lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        eps = 1e-5
        pos = atoms.get_positions().astype(np.float64)
        fd_our = {el: np.zeros_like(our_dg[el]) for el in our_dg}

        def eval_g(positions):
            prepared = fingerprints_from_arrays(
                positions[None], lengths[None], pbc, symbols, neigh[None], order, specs, normalize=False
            )
            out = {}
            for element in prepared["elements"]:
                arr = np.zeros((len(atoms), prepared["n_fp"][element]))
                arr[prepared["centers"][element]] = prepared["fps"][element][0]
                out[element] = arr
            return out

        for atom in range(len(atoms)):
            for direction in range(3):
                pp = pos.copy()
                pm = pos.copy()
                pp[atom, direction] += eps
                pm[atom, direction] -= eps
                gp = eval_g(pp)
                gm = eval_g(pm)
                for element in fd_our:
                    fd_our[element][:, :, atom, direction] = (gp[element] - gm[element]) / (2 * eps)
        print("our jacrev vs FD of our G, neighbors fixed", {el: max_err(our_dg[el], fd_our[el]) for el in our_dg})
        print("PyAMFF analytic vs FD of our G", {el: max_err(dg[el], fd_our[el]) for el in dg})


def small_pt():
    positions = np.array(
        [
            [8.6964, 7.8873, 1.5741],
            [8.8873, 4.1428, 8.8971],
            [1.0862, 9.5476, 1.2094],
            [7.3070, 7.0155, 7.4174],
        ]
    )
    return Atoms("Pt4", positions=positions, cell=[10.0, 10.0, 10.0], pbc=True)


def use_file_rcut(enabled):
    if not enabled:
        return
    original = BehlerParrinello.set

    def patched(self, g):
        original(self, g)
        if getattr(g, "rcut", None) is not None:
            self.r_cut = float(g.rcut)

    BehlerParrinello.set = patched


if __name__ == "__main__":
    fp = "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat"
    compare("small periodic Pt, PyAMFF default r_cut", small_pt(), fp)
    use_file_rcut(True)
    compare("small periodic Pt, file Rc copied onto r_cut", small_pt(), fp)
```

The kept-block numbers come from the same file-Rc geometry. `present` marks the center-wrt blocks `loop_FPs` stored. The maximum absolute difference on those blocks is 4.440892098500626e-16. The maximum absolute jacrev value on the missing blocks is 0.19069594465522602. Two of 16 blocks are missing.
