"""End-to-end correctness gate: the upstream tutorial-3 model, its normalization file and its
test files, through our data layer, rollout and metrics, must reproduce the numbers the
upstream notebook printed. Also checks the two baseline architectures load the upstream
weights strictly and have the published parameter counts."""
import numpy as np
import pytest
import torch

from conftest import ref_path
from src.checkpoint import load_state_dict_file
from src.data import load_test_h5
from src.evaluate import check_targets, evaluate_model
from src.models import OfficialLSTM, TutorialLSTM, count_parameters
from src.normalize import NormParams
from src.reference import PRETEST, TUTORIAL3, check_hash


def test_parameter_counts():
    assert count_parameters(OfficialLSTM()) == 771
    assert count_parameters(TutorialLSTM()) == 2399


@pytest.mark.parametrize("material", sorted(PRETEST))
def test_pretest_weights_load_into_official_lstm(material):
    ref = PRETEST[material]
    p = ref_path(ref.weights)
    check_hash(p, ref.weights_sha256)
    OfficialLSTM().load_state_dict(load_state_dict_file(p), strict=True)


def test_tutorial_weights_do_not_fit_official_lstm():
    sd = load_state_dict_file(ref_path(TUTORIAL3.weights))
    with pytest.raises(RuntimeError):
        OfficialLSTM().load_state_dict(sd, strict=True)


def test_test_files_layout():
    ts = load_test_h5(ref_path(TUTORIAL3.test_padded), ref_path(TUTORIAL3.test_true))
    assert ts.B.shape == (105, 1000)
    w = ts.warmups
    assert sorted(set(w.tolist())) == [100, 300, 500, 700, 900]
    assert all(np.sum(w == k) == 21 for k in (100, 300, 500, 700, 900))


@pytest.fixture(scope="module")
def tutorial3_result():
    torch.manual_seed(1)
    wp, npth = ref_path(TUTORIAL3.weights), ref_path(TUTORIAL3.norm)
    check_hash(wp, TUTORIAL3.weights_sha256)
    check_hash(npth, TUTORIAL3.norm_sha256)
    model = TutorialLSTM()
    model.load_state_dict(load_state_dict_file(wp), strict=True)
    norm = NormParams.from_json(npth)
    ts = load_test_h5(ref_path(TUTORIAL3.test_padded), ref_path(TUTORIAL3.test_true))
    return evaluate_model(model, norm, ts, device="cpu"), ts


def test_tutorial3_gate(tutorial3_result):
    res, _ = tutorial3_result
    assert check_targets(res["summary"], TUTORIAL3.targets) == []


def test_tutorial3_predictions_match_upstream_csv(tutorial3_result):
    """Element-wise: our rollout vs the predictions the upstream notebook saved."""
    res, ts = tutorial3_result
    rows = [np.array([float(x) for x in line.split(",")])
            for line in open(ref_path("tutorials/tutorial-3/Output/pred.csv")).read().splitlines()]
    w = ts.warmups
    worst = 0.0
    for i, r in enumerate(rows):
        ours = res["H_pred"][i, w[i]:]
        assert ours.shape == r.shape
        worst = max(worst, float(np.max(np.abs(ours - r)) / (np.mean(np.abs(r)) + 1e-12)))
    assert worst < 1e-3, f"worst relative deviation {worst:.2e}"


def test_wrong_normalization_breaks_gate():
    """Using the tutorial-1 normalization with the tutorial-3 weights must NOT pass."""
    model = TutorialLSTM()
    model.load_state_dict(load_state_dict_file(ref_path(TUTORIAL3.weights)), strict=True)
    wrong = NormParams.from_json(ref_path("tutorials/tutorial-1/Normalization_Params.json"))
    ts = load_test_h5(ref_path(TUTORIAL3.test_padded), ref_path(TUTORIAL3.test_true))
    res = evaluate_model(model, wrong, ts, device="cpu")
    assert check_targets(res["summary"], TUTORIAL3.targets) != []
