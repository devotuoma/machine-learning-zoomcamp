import sklearn

from q2_locked_version import locked_version, pinned_version


def test_lockfile_pins_scikit_learn_1_7_2():
    assert locked_version("scikit-learn") == "1.7.2"
    assert pinned_version("scikit-learn") == "1.7.2"


def test_running_environment_matches_lockfile():
    # Fails if someone ran `uv sync` without --locked and drifted.
    assert sklearn.__version__ == locked_version("scikit-learn")
