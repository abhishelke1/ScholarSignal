from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_overview_and_agent_command():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "86.1%"
    app.text_input(key="command").set_value("explain naive bayes")
    app.form_submit_button[0].click().run()
    assert not app.exception
    assert "Categorical Naive Bayes" in app.info[0].value


def test_single_prediction_and_mode_reset():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.switch_page("app_pages/predict.py").run()
    assert not app.exception
    app.number_input(key="G1").set_value(16)
    app.number_input(key="G2").set_value(17)
    app.form_submit_button[0].click().run()
    assert not app.exception
    assert "single_result" in app.session_state
    assert len(app.session_state["single_result"]["study_plan"]) > 0
    app.session_state["mode"] = "Early warning"
    app.run()
    assert not app.exception
    assert "single_result" not in app.session_state
    assert app.session_state["agent"].run.model.n_features_in_ == 9


def test_batch_demo_and_result_clearing():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.switch_page("app_pages/batch.py").run()
    app.checkbox(key="batch_demo").check().run()
    app.button(key="predict_batch").click().run()
    assert not app.exception
    assert len(app.session_state["batch_result"][0]) == 5
    app.checkbox(key="batch_demo").uncheck().run()
    assert "batch_result" not in app.session_state


def test_insights_and_lab_pages():
    app = AppTest.from_file(APP, default_timeout=30).run()
    for page in ("app_pages/insights.py", "app_pages/lab.py"):
        app.switch_page(page).run()
        assert not app.exception
        assert app.title
