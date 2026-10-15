"""Command-line workflow for reproducible lab evaluations."""
import argparse
from pathlib import Path

from student_agent.agent import PerformanceAgent
from student_agent.data import DataError, csv_bytes, load_demo, read_csv


def main():
    parser = argparse.ArgumentParser(description="Run the Student Performance Naive Bayes agent.")
    parser.add_argument("--dataset", type=Path, help="Training CSV; default is bundled UCI mathematics data.")
    parser.add_argument("--early-warning", action="store_true", help="Exclude G1 and G2.")
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation_report.json"))
    parser.add_argument("--predict", type=Path, help="Optional student CSV to predict.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()
    try:
        data = read_csv(args.dataset.read_bytes()) if args.dataset else load_demo()
        agent = PerformanceAgent(data, "Early warning" if args.early_warning else "With prior grades", str(args.dataset) if args.dataset else "UCI Student Performance - Mathematics")
        agent.execute("evaluate")
        if args.predict:
            students = read_csv(args.predict.read_bytes())
            predictions = agent.execute("predict", students)["predictions"]
            predictions.insert(0, "csv_line", range(2, len(predictions) + 2))
            if "student_id" in students:
                predictions.insert(1, "student_id", students.student_id.to_numpy())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(csv_bytes(predictions))
        agent.execute("report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(agent.report_json(), encoding="utf-8")
        metrics = agent.run.metrics
        print(f"Accuracy: {metrics['accuracy']:.1%} | Macro F1: {metrics['macro_f1']:.3f} | Test rows: {metrics['test_rows']}")
        print(f"Report: {args.report.resolve()}")
    except (DataError, OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
