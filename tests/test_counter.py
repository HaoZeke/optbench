from optbench.counter import CountedPotential, ForceCallCap, HarnessCounter, PeleCounter, attach_ase_counter


def test_cache_hit_is_only_the_last_geometry():
    counter = HarnessCounter(cache=True)
    fresh, _cached = counter.charge((1.0, 0.0))
    assert fresh is True
    counter.remember((1.0, 0.0), (0.0, (0.0, 0.0)))
    fresh, cached = counter.charge((1.0, 0.0))
    assert fresh is False
    assert cached == (0.0, (0.0, 0.0))
    fresh, _cached = counter.charge((2.0, 0.0))
    assert fresh is True
    counter.remember((2.0, 0.0), (1.0, (0.0, 0.0)))
    fresh, _cached = counter.charge((1.0, 0.0))
    assert fresh is True
    assert counter.calls == 3
    assert counter.hits == 1
    counter.reset()
    assert counter.hits == 0
    assert counter.calls == 0


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
