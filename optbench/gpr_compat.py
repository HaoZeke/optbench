"""Import shims for GPR calculator 0.0.4 on this machine's libraries.

The pinned source imports ``scipy.special.sph_harm`` and
``ase.constraints.full_3x3_to_voigt_6_stress``. SciPy 1.18 moved the
harmonic to ``sph_harm_y``. ASE 3.29 moved the Voigt helper to
``ase.stress``. Both shims are installed before ``gpr_calc`` is imported.
The angular convention matches SciPy 1.14: ``sph_harm(m, n, azimuth, polar)``
is ``sph_harm_y(n, m, polar, azimuth)``.
"""

from __future__ import annotations


def install() -> None:
    """Register the two names the pinned source imports."""
    import scipy.special as special

    if not hasattr(special, "sph_harm"):

        def sph_harm(m, n, theta, phi):
            return special.sph_harm_y(n, m, phi, theta)

        special.sph_harm = sph_harm

    import ase.constraints as constraints

    if not hasattr(constraints, "full_3x3_to_voigt_6_stress"):
        from ase.stress import full_3x3_to_voigt_6_stress

        constraints.full_3x3_to_voigt_6_stress = full_3x3_to_voigt_6_stress
