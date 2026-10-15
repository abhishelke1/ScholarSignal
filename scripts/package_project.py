"""Bundle the source and public examples, excluding environments and session data."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "artifacts" / "ScholarSignal-AI-Lab.zip"
output.parent.mkdir(exist_ok=True)
files = [ROOT / name for name in ("app.py", "README.md", "requirements.txt", "requirements-dev.txt", "pytest.ini", "start.bat", "setup.bat", ".gitignore", ".gitattributes", ".streamlit/config.toml")]
for directory in ("student_agent", "app_pages", "tests", "docs", "data", "scripts"):
    files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, Path("ScholarSignal") / path.relative_to(ROOT))
    for filename in ("with_prior_grades.json", "early_warning.json", "example_predictions.csv"):
        path = ROOT / "artifacts" / filename
        if path.exists():
            archive.write(path, f"ScholarSignal/sample_results/{filename}")
with ZipFile(output) as archive:
    assert archive.testzip() is None
    assert not any(".venv" in name or "secrets" in name for name in archive.namelist())
    print(f"Packaged {len(archive.namelist())} files: {output}")
