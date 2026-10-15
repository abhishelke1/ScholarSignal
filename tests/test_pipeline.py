import json

import numpy as np
import pandas as pd
import pytest

from student_agent.agent import PerformanceAgent
from student_agent.data import DataError, LABELS, csv_bytes, encode, features, load_demo, read_csv, targets, validate
from student_agent.model import explain, predict


@pytest.fixture(scope="module")
def demo():
    return load_demo()


@pytest.fixture(scope="module")
def agent(demo):
    instance = PerformanceAgent(demo, "With prior grades", "UCI test")
    instance.execute("train")
    return instance


def test_grade_boundaries():
    assert targets(pd.Series([0, 9, 10, 13, 14, 20])).tolist() == [LABELS[0], LABELS[0], LABELS[1], LABELS[1], LABELS[2], LABELS[2]]


def test_real_dataset_and_disjoint_holdout(agent):
    assert len(agent.clean) == 395
    assert not set(agent.run.train_indices) & set(agent.run.test_indices)
    assert len(set(agent.run.train_indices) | set(agent.run.test_indices)) == 395
    assert agent.run.metrics["train_rows"] == 316
    assert agent.run.metrics["test_rows"] == 79
    assert np.array(agent.run.metrics["confusion_matrix"]).sum() == 79
    assert "G3" not in features(agent.mode)


def test_target_never_used_during_prediction(agent, demo):
    first = agent.execute("predict", demo.head(4))["predictions"]
    changed = demo.head(4).copy()
    changed["G3"] = [0, 20, 0, 20]
    second = agent.execute("predict", changed)["predictions"]
    pd.testing.assert_frame_equal(first, second)


def test_early_mode_omits_both_grades(demo):
    instance = PerformanceAgent(demo.drop(columns=["G1", "G2"]), "Early warning", "test")
    instance.execute("train")
    assert instance.run.model.n_features_in_ == 9
    assert np.isfinite(instance.run.metrics["macro_f1"])
    assert len(instance.execute("predict", demo.head(2))["predictions"]) == 2


@pytest.mark.parametrize("field,value", [("G1", 21), ("G2", -1), ("studytime", 1.5), ("absences", np.inf), ("failures", "oops"), ("schoolsup", "sometimes")])
def test_invalid_input_is_rejected(demo, field, value):
    row = demo.head(1).copy().astype(object)
    row.loc[0, field] = value
    with pytest.raises(DataError, match=field):
        validate(row, "With prior grades")


def test_missing_column_is_actionable(demo):
    with pytest.raises(DataError, match="G2"):
        validate(demo.drop(columns="G2"), "With prior grades")


def test_missing_features_are_finite_and_flagged(agent, demo):
    row = demo.head(1).copy().astype(object)
    row.loc[0, ["G2", "schoolsup"]] = None
    result = agent.execute("predict", row)
    probabilities = result["predictions"].filter(like="p_").to_numpy()
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1)
    assert bool(result["predictions"].review_recommended.iloc[0])
    assert len(result["notes"]) == 2


def test_empty_student_rejected(demo):
    row = demo.head(1).copy().astype(object)
    row.loc[:, features("With prior grades")] = None
    with pytest.raises(DataError, match="no observed features"):
        validate(row, "With prior grades")


def test_invalid_training_target_and_small_class(demo):
    frame = demo.copy()
    frame.loc[0, "G3"] = np.nan
    with pytest.raises(DataError, match="cannot be blank"):
        validate(frame, "With prior grades", training=True)
    with pytest.raises(DataError, match="every band"):
        validate(demo.assign(G3=12), "With prior grades", training=True)


def test_fixed_bins_and_unknown_category(demo):
    frame, _ = validate(demo, "With prior grades")
    matrix, sizes = encode(frame, "With prior grades")
    assert (matrix >= 0).all()
    assert (matrix < sizes).all()
    small, _ = encode(frame.head(2), "With prior grades")
    np.testing.assert_array_equal(small, matrix[:2])


def test_explanation_reconstructs_posterior_odds(agent):
    row = agent.clean.head(1)
    detail = explain(agent.run, row)
    probabilities = predict(agent.run, row).iloc[0]
    probability = lambda label: probabilities[f"p_{label.lower().replace(' ', '_')}"]
    expected = np.log(probability(detail["winner"]) / probability(detail["runner_up"]))
    assert detail["prior_log_odds"] + sum(item["log_odds"] for item in detail["evidence"]) == pytest.approx(expected)


def test_lazy_training_and_model_reuse(demo):
    instance = PerformanceAgent(demo, "With prior grades", "test")
    assert instance.run is None
    instance.execute("predict", demo.head(1))
    previous = instance.run
    instance.execute("evaluate")
    assert instance.run is previous
    assert sum(event["tool"] == "train_naive_bayes" for event in instance.events) == 1


def test_exact_duplicate_rows_removed_before_split(demo):
    instance = PerformanceAgent(pd.concat([demo, demo.head(2)]), "With prior grades", "test")
    instance.execute("train")
    assert len(instance.clean) == len(demo)


def test_csv_formats_and_invalid_headers():
    assert read_csv(b"a;b\n1;2").shape == (1, 2)
    assert read_csv(b"a,b\n1,2").shape == (1, 2)
    with pytest.raises(DataError, match="duplicate"):
        read_csv(b"a,a\n1,2")
    with pytest.raises(DataError):
        read_csv(b"")


def test_csv_export_neutralizes_formula():
    exported = csv_bytes(pd.DataFrame({"student_id": ["=SUM(1,2)", "student-1"]})).decode("utf-8-sig")
    assert "'=SUM" in exported
    assert "student-1" in exported


def test_report_is_serializable_and_contains_no_student_records(agent):
    report = json.loads(agent.report_json())
    assert len(report["dataset_sha256"]) == 64
    assert "metrics" in report and "trace" in report
    assert "data" not in report


def test_agent_rejects_unsupported_goal(agent):
    with pytest.raises(DataError, match="Supported goals"):
        agent.execute("arbitrary_code")
    assert "support these commands" in agent.command("hello")
