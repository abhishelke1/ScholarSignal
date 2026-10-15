"""Train-only model selection, untouched holdout evaluation, exact NB explanations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB

from student_agent.data import DISPLAY, LABELS, encode, features, targets


@dataclass
class ModelRun:
    model: CategoricalNB
    mode: str
    metrics: dict
    candidates: list[dict]
    train_indices: list[int]
    test_indices: list[int]


def train(frame: pd.DataFrame, mode: str) -> ModelRun:
    x, sizes = encode(frame, mode)
    y = targets(frame.G3).to_numpy()
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    for alpha in (0.5, 1.0, 2.0):
        estimator = CategoricalNB(alpha=alpha, min_categories=sizes)
        scores = cross_val_score(estimator, x[train_ids], y[train_ids], cv=cv, scoring="f1_macro", error_score="raise")
        candidates.append({"alpha": alpha, "cv_macro_f1": float(scores.mean()), "cv_std": float(scores.std())})
    best = max(candidates, key=lambda candidate: candidate["cv_macro_f1"])
    model = CategoricalNB(alpha=best["alpha"], min_categories=sizes).fit(x[train_ids], y[train_ids])
    predicted = model.predict(x[test_ids])
    baseline = DummyClassifier(strategy="most_frequent").fit(x[train_ids], y[train_ids]).predict(x[test_ids])
    metrics = {
        "accuracy": float(accuracy_score(y[test_ids], predicted)),
        "macro_f1": float(f1_score(y[test_ids], predicted, average="macro", zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y[test_ids], baseline)),
        "baseline_macro_f1": float(f1_score(y[test_ids], baseline, average="macro", zero_division=0)),
        "cv_macro_f1": best["cv_macro_f1"], "alpha": best["alpha"],
        "train_rows": len(train_ids), "test_rows": len(test_ids), "seed": 42,
        "confusion_matrix": confusion_matrix(y[test_ids], predicted, labels=LABELS).tolist(),
        "classification_report": classification_report(y[test_ids], predicted, labels=LABELS, output_dict=True, zero_division=0),
        "class_counts": targets(frame.G3).value_counts().reindex(LABELS, fill_value=0).to_dict(),
    }
    return ModelRun(model, mode, metrics, candidates, train_ids.tolist(), test_ids.tolist())


def predict(run: ModelRun, frame: pd.DataFrame) -> pd.DataFrame:
    x, _ = encode(frame, run.mode)
    probabilities = run.model.predict_proba(x)
    result = pd.DataFrame({"prediction": run.model.predict(x)})
    for label in LABELS:
        result[f"p_{label.lower().replace(' ', '_')}"] = probabilities[:, list(run.model.classes_).index(label)]
    result["model_probability"] = probabilities.max(axis=1)
    result["review_recommended"] = (result.prediction == LABELS[0]) | (result.model_probability < 0.6) | frame[features(run.mode)].isna().any(axis=1).to_numpy()
    return result


def explain(run: ModelRun, row: pd.DataFrame) -> dict:
    """Decompose winner-vs-runner-up log posterior odds exactly."""
    x, _ = encode(row.iloc[:1], run.mode)
    probabilities = run.model.predict_proba(x)[0]
    order = np.argsort(probabilities)
    winner, runner = int(order[-1]), int(order[-2])
    evidence = []
    for i, col in enumerate(features(run.mode)):
        contribution = run.model.feature_log_prob_[i][winner, x[0, i]] - run.model.feature_log_prob_[i][runner, x[0, i]]
        evidence.append({"feature": col, "label": DISPLAY[col], "log_odds": float(contribution)})
    return {
        "winner": str(run.model.classes_[winner]), "runner_up": str(run.model.classes_[runner]),
        "prior_log_odds": float(run.model.class_log_prior_[winner] - run.model.class_log_prior_[runner]),
        "evidence": sorted(evidence, key=lambda item: abs(item["log_odds"]), reverse=True),
    }


def study_plan(row: pd.Series, prediction: str) -> list[str]:
    """Transparent support rules, never causal claims from model correlations."""
    plan = []
    if pd.notna(row.get("studytime")) and row.studytime <= 2:
        plan.append("Schedule four 25-minute revision sessions this week; finish each with five practice questions.")
    if pd.notna(row.get("absences")) and row.absences > 5:
        plan.append("Review missed lessons with a teacher and make a catch-up checklist.")
    if pd.notna(row.get("failures")) and row.failures > 0:
        plan.append("Ask a teacher to identify two prerequisite topics to practise before the next assessment.")
    if prediction == LABELS[0]:
        plan.append("Arrange a supportive teacher check-in; review current work before deciding on extra help.")
    plan.append("Complete a short practice quiz at the end of the week and use the result to update the plan.")
    return plan
