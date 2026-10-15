import altair as alt
import pandas as pd
import streamlit as st

from student_agent.data import LABELS
from student_agent.ui import metrics_row, ready_agent, trace

st.title("Evidence behind the estimate")
st.markdown("Inspect the held-out results, class-level errors, and every step the agent took.")
agent = ready_agent()
metrics = agent.run.metrics
metrics_row(metrics)
st.caption(f"Seed 42 · stratified 80/20 split · {metrics['train_rows']} training rows · five-fold training-only selection · selected alpha {metrics['alpha']}")
left, right = st.columns(2, gap="large")
with left, st.container(border=True):
    st.subheader("Where predictions land")
    matrix = pd.DataFrame([{"Actual": actual, "Predicted": predicted, "Students": metrics["confusion_matrix"][i][j]} for i, actual in enumerate(LABELS) for j, predicted in enumerate(LABELS)])
    base = alt.Chart(matrix).encode(x=alt.X("Predicted:N", sort=LABELS, axis=alt.Axis(labelAngle=0)), y=alt.Y("Actual:N", sort=LABELS))
    heatmap = base.mark_rect(cornerRadius=4).encode(color=alt.Color("Students:Q", scale=alt.Scale(range=["#EFF5F0", "#087F73"]), legend=None), tooltip=["Actual", "Predicted", "Students"])
    text = base.mark_text(fontSize=22, fontWeight=600).encode(text="Students:Q", color=alt.condition(alt.datum.Students > matrix.Students.max() / 2, alt.value("white"), alt.value("#182D29")))
    st.altair_chart((heatmap + text).properties(height=270), alt="Confusion matrix: actual versus predicted performance bands on the holdout set")
with right, st.container(border=True):
    st.subheader("Performance by grade band")
    report = pd.DataFrame(metrics["classification_report"]).T.loc[LABELS]
    st.dataframe(report.round(3), alt="Precision, recall, F1 and support for each grade band")
    st.caption("Precision: how often a predicted band is correct. Recall: how many students in a band are found. Macro F1 weights all three bands equally.")
    st.markdown(f"**Majority baseline macro F1:** {metrics['baseline_macro_f1']:.3f}")
st.subheader("How the agent chose the model")
st.dataframe(pd.DataFrame(agent.run.candidates).round(4), hide_index=True, alt="Training cross-validation results for three additive smoothing settings")
st.caption("Alpha is selected by mean training cross-validation macro F1. The test set is evaluated once after selection. The app uses that same training-fitted model for predictions.")
if agent.mode == "Early warning":
    st.warning("Without previous grades, prediction can be much harder. Cumulative absence and support fields must also be checked for availability at the intended prediction date.")
st.info("This is one split of a small educational dataset. Results may change on another school or cohort; model probabilities are uncalibrated.", icon=":material/info:")
st.subheader("Agent activity")
trace(agent)
st.download_button("Download evaluation report", agent.report_json(), "evaluation_report.json", "application/json", type="primary", icon=":material/download:")
