"""A bounded, goal-driven tool agent with explicit state and an audit trail.

The controller decides which tools are needed from its goal and current state.
Its predictor learns from data; its planning policy is deterministic, not an LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd

from student_agent.data import DataError, validate
from student_agent.model import ModelRun, explain, predict, study_plan, train


class PerformanceAgent:
    def __init__(self, data: pd.DataFrame, mode: str, source: str):
        self.data = data.copy()
        self.mode = mode
        self.source = source
        self.run: ModelRun | None = None
        self.clean: pd.DataFrame | None = None
        self.events: list[dict] = []
        self.fingerprint = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()

    def log(self, tool: str, status: str, detail: str):
        self.events.append({"step": len(self.events) + 1, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool, "status": status, "detail": detail})
        self.events = self.events[-200:]

    def ensure_model(self):
        if self.run is not None:
            self.log("memory", "ready", "Reused the evaluated model for the current dataset and mode.")
            return
        self.log("planner", "planned", "No trained model: validate → encode → split → cross-validate → fit → evaluate.")
        try:
            # Remove only exact full-row duplicates before splitting, not students
            # with the same selected predictors (they may be different students).
            unique = self.data.drop_duplicates().reset_index(drop=True)
            self.clean, notes = validate(unique, self.mode, training=True)
            self.log("validate_data", "complete", f"Accepted {len(self.clean)} rows; removed {len(self.data) - len(unique)} exact duplicate(s). " + " ".join(notes))
            self.run = train(self.clean, self.mode)
            self.log("train_naive_bayes", "complete", f"Selected alpha={self.run.metrics['alpha']} with five-fold training-only macro F1.")
            self.log("evaluate_holdout", "complete", f"Untouched 20% test set: accuracy {self.run.metrics['accuracy']:.1%}; macro F1 {self.run.metrics['macro_f1']:.3f}.")
            if self.run.metrics["macro_f1"] <= self.run.metrics["baseline_macro_f1"]:
                self.log("quality_check", "review", "Model did not outperform the majority baseline on macro F1. Treat predictions as exploratory.")
            else:
                self.log("quality_check", "complete", "Model exceeded the majority baseline on test macro F1; external validation is still needed.")
        except (DataError, ValueError) as exc:
            self.log("workflow", "blocked", str(exc))
            raise

    def execute(self, goal: str, students: pd.DataFrame | None = None) -> dict:
        if goal not in ("train", "evaluate", "predict", "report"):
            raise DataError("Supported goals: train, evaluate, predict, report.")
        self.log("planner", "goal", f"Requested goal: {goal}.")
        if goal == "predict" and students is None:
            raise DataError("Provide a student profile or prediction CSV.")
        self.ensure_model()
        if goal == "predict":
            try:
                clean, notes = validate(students, self.mode)
            except DataError as exc:
                self.log("validate_students", "blocked", str(exc))
                raise
            result = predict(self.run, clean)
            self.log("predict_students", "complete", f"Predicted {len(result)} row(s); {int(result.review_recommended.sum())} marked for review. " + " ".join(notes))
            output = {"predictions": result, "notes": notes}
            if len(clean) == 1:
                output["explanation"] = explain(self.run, clean)
                output["study_plan"] = study_plan(clean.iloc[0], result.prediction.iloc[0])
                self.log("explain_and_plan", "complete", "Computed exact feature log-odds and a rule-based study checklist.")
            return output
        if goal == "report":
            self.log("export_report", "complete", "Prepared a reproducible report without student records.")
            return self.report()
        return self.run.metrics

    def command(self, message: str) -> str:
        words = set(re.findall(r"[a-z]+", message.lower()))
        if words & {"explain", "algorithm", "bayes"}:
            self.log("explain_method", "complete", "Explained the classifier and controller.")
            return "I use Categorical Naive Bayes: prior class probabilities multiplied by the likelihood of each observed feature. Fixed bins represent grades and absences; additive smoothing avoids zero likelihoods. My controller selects tools from the goal and current state. I am a bounded workflow agent, not an LLM chatbot."
        if words & {"report", "export"}:
            self.execute("report")
            return "Your report is ready. Download it in Model insights; it includes the dataset fingerprint, split, metrics, candidates, and agent trace."
        if words & {"train", "run", "evaluate", "accuracy", "performance", "analyse", "analyze"}:
            metrics = self.execute("evaluate")
            return f"Evaluation complete: accuracy {metrics['accuracy']:.1%}, macro F1 {metrics['macro_f1']:.3f}, majority baseline accuracy {metrics['baseline_accuracy']:.1%}. The test set contains {metrics['test_rows']} students. Use Model insights for class-level results."
        if words & {"predict", "student"}:
            return "Open Predict a student to enter a profile, or Batch predictions to upload a CSV. I will validate the input, train if needed, predict, and flag records for review."
        return "I support these commands: run analysis, evaluate model, explain naive bayes, export report, and predict student. Use the profile form for student values."

    def report(self) -> dict:
        if self.run is None:
            raise DataError("Run an analysis before exporting a report.")
        return {"project": "Student Performance Prediction using Naive Bayes", "agent": "Deterministic goal-driven tool controller; no LLM", "source": self.source, "dataset_sha256": self.fingerprint, "mode": self.mode, "model": "CategoricalNB", "grade_bands": {"Needs support": "0–9", "On track": "10–13", "Strong": "14–20"}, "metrics": self.run.metrics, "candidates": self.run.candidates, "limitations": ["Probabilities are uncalibrated model estimates, not guarantees.", "Correlations and rule-based advice do not establish causation.", "One fixed holdout from a small dataset; no external validation.", "Early-warning mode omits G1/G2 but other fields may be cumulative; temporal validity must be checked.", "Predictions support teacher review, not admissions, grading, or disciplinary decisions."], "trace": self.events.copy()}

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, ensure_ascii=False)
