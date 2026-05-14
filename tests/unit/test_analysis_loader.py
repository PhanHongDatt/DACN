from analysis.loader import _parse_filename


def test_parse_trimmed_mean_log_filename():
    meta = _parse_filename("mnist_K3_TrimmedMean_a00_da010_20260514_120000.csv")

    assert meta["dataset"] == "mnist"
    assert meta["scenario"] == "K3"
    assert meta["config"] == "TrimmedMean"
    assert meta["dirichlet_alpha"] == 0.1
