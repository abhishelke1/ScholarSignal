import pandas as pd
import streamlit as st

from student_agent.data import LABELS, csv_bytes, features
from student_agent.ui import band_chart, metrics_row, ready_agent, trace

st.badge("NAIVE BAYES · GOAL-DRIVEN AGENT", icon=":material/auto_awesome:", color="green")
st.title("Understand student performance.")
st.markdown("An explainable learning companion that turns student data into performance predictions and practical study checklists.")
agent = ready_agent()
st.caption(f"{agent.source}  /  {agent.mode}  /  {len(agent.clean)} validated records")
metrics_row(agent.run.metrics)
left, right = st.columns([1.15, 1], gap="large")
with left, st.container(border=True):
    st.subheader("The class at a glance")
    counts = pd.DataFrame({"Performance": LABELS, "Students": [agent.run.metrics["class_counts"][label] for label in LABELS]})
    st.altair_chart(band_chart(counts, "Students"), alt="Actual final-grade band distribution in the training dataset")
    st.caption("Observed final grades: Needs support 0–9 · On track 10–13 · Strong 14–20. Grades are out of 20.")
with right, st.container(border=True):
    st.subheader("Your agent is ready", icon=":material/check_circle:")
    st.markdown("**01 · Inspect**  Validate student records and handle missing features.\n\n**02 · Learn**  Compare three smoothing settings using cross-validation.\n\n**03 · Evaluate**  Check predictions on an untouched test set.\n\n**04 · Support**  Explain predictions and suggest a study checklist.")
    st.page_link("app_pages/predict.py", label="Predict a student", icon=":material/arrow_forward:")
    st.page_link("app_pages/batch.py", label="Work with a class CSV", icon=":material/upload_file:")
st.subheader("Give the agent a task")
with st.form("agent_command"):
    command = st.text_input("Task", placeholder="Try: evaluate model, explain naive bayes, export report", key="command")
    send = st.form_submit_button("Run task", type="primary", icon=":material/play_arrow:")
if send:
    st.session_state.command_response = agent.command(command)
if st.session_state.get("command_response"):
    st.info(st.session_state.command_response, icon=":material/smart_toy:")
st.caption("This agent supports a defined set of commands. Predictions come from Naive Bayes; planning and advice use explicit rules.")
with st.expander("Agent activity"):
    trace(agent)
with st.expander("Explore the dataset and download a template"):
    st.dataframe(agent.clean.head(30), hide_index=True, alt="First thirty validated student records")
    st.download_button("Download training template", csv_bytes(agent.clean), "training_template.csv", "text/csv")
    st.download_button("Download prediction example", csv_bytes(agent.clean[features(agent.mode)].head(5)), "prediction_example.csv", "text/csv")
st.caption("Built for learning and teacher review. A prediction is an estimate, not a student's potential.")
