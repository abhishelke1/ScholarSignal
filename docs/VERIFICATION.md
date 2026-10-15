# Verification record

Verified on 2026-10-15 using Python 3.12.14 on Windows.

- `python -m pytest -q`: **26 passed**, no warnings after the final fix.
- `python -m pip check`: no broken requirements.
- `python -m compileall -q app.py app_pages student_agent tests`: completed successfully.
- CLI training/evaluation executed for both prediction modes and wrote JSON reports.
- CLI batch prediction executed on five bundled sample profiles and wrote a prediction CSV.
- Live Streamlit server started on `127.0.0.1:8501`.
- Browser checked the overview layout, navigated to the student form, submitted the default profile, and verified the prediction, probability and study plan. A preview is saved in `app-preview.jpg`.

Automated interface tests cover the overview command, individual prediction, mode-change reset, batch demonstration, clearing batch results, model insights and lab-guide pages. Model tests cover contracts, target exclusion, held-out partition separation, missing values, exact explanation arithmetic and model reuse.

The live browser check used the available narrow in-app viewport. It is not a cross-browser or full accessibility certification. File parsing is covered in model tests; a native OS file-chooser upload was not separately exercised in the browser.

The restricted test environment blocked Windows loopback sockets required by Streamlit AppTest. The suite passed outside that restriction. This was an environment issue rather than an application failure.
