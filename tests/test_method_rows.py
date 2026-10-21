"""A ranked row is one code, one version, and one stock method."""

from optbench.families import finite_difference_calls, method_name


def test_fd_calls_use_one_force_call_per_coordinate():
    assert finite_difference_calls(1.0, 114) == 114.0
    assert finite_difference_calls(80.881, 114) == 80.881 * 114
    assert finite_difference_calls(0.0, None) == 0.0
    assert finite_difference_calls(1.0, None) is None


def test_a_parameter_record_is_not_the_method_name():
    blob = '{"counter":"harness","parameters":{"memory":100}}'
    assert method_name({"algorithm": "LBFGS"}, {"method": blob, "algorithm": "LBFGS"}) == "LBFGS"
    assert method_name({"method": "CI-NEB lj/cut", "algorithm": "CI-NEB"}, {"method": blob}) == "CI-NEB lj/cut"
