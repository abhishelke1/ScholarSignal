@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run setup.bat to install the project environment.
  pause
  exit /b 1
)
echo Starting ScholarSignal at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
if errorlevel 1 pause
