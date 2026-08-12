from kernels.blackwell_e0_probe.probe.run_probe import run_all_cases


def test_public_e2m1_mma_matches_all_positive_cases_and_catches_control():
    report = run_all_cases()
    assert report["passed"]
    assert all(case["passed"] for case in report["positive_cases"])
    assert report["negative_control"]["expected_to_fail"]
    assert report["negative_control"]["caught"]
