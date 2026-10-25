from optbench.counter import CountedPotential, ForceCallCap, HarnessCounter, PeleCounter, attach_ase_counter


def test_repeated_coordinates_are_one_force_call():
    seen = []

    def evaluate(coords):
        seen.append(coords)
        return (coords[0] ** 2, (2.0 * coords[0], 0.0))

    potential = CountedPotential(evaluate)
    potential.energy_forces((1.0, 0.0))
    potential.energy_forces((1.0, 0.0))
    assert potential.calls == 1
    assert len(seen) == 1
    potential.energy_forces((2.0, 0.0))
    assert potential.calls == 2


def test_verification_calls_are_not_search_calls():
    potential = CountedPotential(lambda coords: (0.0, (0.0, 0.0)))
    potential.energy_forces((0.0, 0.0))
    with potential.verification():
        potential.energy_forces((1.0, 0.0))
        potential.energy_forces((1.0, 0.0))
    assert potential.calls == 1
    assert potential.verify_calls == 1


def test_two_optimizers_pay_the_wrapper_not_their_own_tally():
    def evaluate(coords):
        return (0.0, (1.0, 0.0))

    def optimizer(potential, steps, own_count):
        for index in range(steps):
            potential.energy_forces((index, 0.0))
            potential.energy_forces((index, 1.0))
        return own_count

    potential = CountedPotential(evaluate)
    reported = optimizer(potential, 3, own_count=3)
    assert reported == 3
    assert potential.calls == 6


def test_ase_style_calculate_counts_one_entry():
    class Calculator:
        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=("positions",)):
            return properties

    calc = attach_ase_counter(Calculator(), cap=2)
    calc.calculate()
    calc.calculate()
    assert calc.optbench_counter["calls"] == 2
    try:
        calc.calculate()
    except RuntimeError as exc:
        assert "cap" in str(exc)
    else:
        raise AssertionError("cap was not enforced")
    assert calc.optbench_counter["calls"] == 2


def test_harness_counter_refuses_the_call_past_the_cap():
    seen = []

    def evaluate(coords):
        seen.append(coords)
        return 0.0, (0.0,)

    counter = HarnessCounter(cache=False, cap=2)
    counter.charge((0.0,))
    counter.charge((1.0,))
    try:
        counter.charge((2.0,))
    except ForceCallCap as exc:
        assert "2" in str(exc)
    else:
        raise AssertionError("cap was not enforced")
    assert counter.calls == 2
    assert seen == []
    with counter.verification():
        fresh, _cached = counter.charge((3.0,))
    assert fresh is True
    assert counter.calls == 2
    assert counter.verify_calls == 1


def test_pele_energy_is_one_gradient_entry():
    class Inner:
        def __init__(self):
            self.entries = 0

        def getEnergyGradient(self, coords):
            self.entries += 1
            return 1.0, [0.0, 0.0, 0.0]

    wrapped = PeleCounter(Inner())
    assert wrapped.getEnergy([0.0, 0.0, 0.0]) == 1.0
    wrapped.getEnergyGradient([0.1, 0.0, 0.0])
    assert wrapped.calls == 2
    assert wrapped._inner.entries == 2


def test_a_davidson_column_build_counts_each_request():
    """The Pt-island Davidson anchor, products, and curvature probe.

    ``Hessianominimo`` requests the base point and one displaced coordinate
    for each of 525 free coordinates. ``finiteH`` adds one new point per
    product and does not re-request the stored base. The iteration-0 LBFGS
    step adds one curvature point. An immediate repeat is still one call,
    and one ``energy_forces`` entry is one charge.
    """
    seen = []

    def evaluate(coords):
        seen.append(tuple(coords))
        return (0.0, tuple(0.0 for _ in coords))

    potential = CountedPotential(evaluate)
    width = 525
    base = tuple(0.0 for _ in range(width))
    step = 1.0e-4
    # minimumMode's first gradient is the search point. The column build
    # uses the stored minimum, a different geometry, so it is not a cache hit.
    potential.energy_forces(tuple(1.0 for _ in range(width)))
    for index in range(width):
        potential.energy_forces(base)
        displaced = [0.0] * width
        displaced[index] = step
        potential.energy_forces(tuple(displaced))
    for product in range(8):
        point = [0.0] * width
        point[0] = step * (2.0 + product)
        potential.energy_forces(tuple(point))
    probe = [0.0] * width
    probe[1] = 1.0
    potential.energy_forces(tuple(probe))
    assert potential.calls == 1 + 2 * width + 8 + 1
    assert len(seen) == potential.calls
    potential.energy_forces(tuple(probe))
    assert potential.calls == 1060
    assert len(seen) == 1060
