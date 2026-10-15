import hashlib

import pandas as pd
import streamlit as st

from student_agent.data import DataError, csv_bytes, features, read_csv
from student_agent.ui import ready_agent

st.title("One class. One clear view.")
st.markdown("Upload a CSV to predict multiple students. Missing feature values are supported; invalid values are explained before prediction.")
agent = ready_agent()
st.download_button("Download example CSV", csv_bytes(agent.clean[features(agent.mode)].head(5)), "prediction_example.csv", "text/csv", icon=":material/download:")
st.caption("Required columns: " + ", ".join(features(agent.mode)) + ". Optional student_id is preserved. G3 is never used for prediction.")
upload = st.file_uploader("Student profiles", type="csv", key="batch_upload")
use_example = st.checkbox("Use five bundled example profiles", key="batch_demo")
batch = None
if use_example:
    batch = agent.clean[features(agent.mode)].head(5).copy()
    st.caption("These example profiles come from the training dataset and are for demonstrating the interface, not evaluating accuracy.")
elif upload:
    try:
        batch = read_csv(upload.getvalue())
    except DataError as exc:
        st.error(str(exc))
if batch is not None:
    batch_id = hashlib.sha256(batch.to_csv(index=False).encode()).hexdigest()
    if st.session_state.get("batch_input_id") != batch_id:
        st.session_state.pop("batch_result", None)
        st.session_state.batch_input_id = batch_id
    st.dataframe(batch.head(10), hide_index=True, alt="Preview of uploaded student profiles")
    if st.button("Predict all students", type="primary", icon=":material/play_arrow:", key="predict_batch"):
        try:
            result = agent.execute("predict", batch)
            output = result["predictions"].copy()
            output.insert(0, "csv_line", range(2, len(output) + 2))
            if "student_id" in batch:
                output.insert(1, "student_id", batch.student_id.to_numpy())
            st.session_state.batch_result = (output, result["notes"])
        except DataError as exc:
            st.session_state.pop("batch_result", None)
            st.error(str(exc))
    if "batch_result" in st.session_state:
        output, notes = st.session_state.batch_result
        with st.container(horizontal=True):
            st.metric("Students processed", len(output), border=True)
            st.metric("Needs support", int((output.prediction == "Needs support").sum()), border=True)
            st.metric("Review recommended", int(output.review_recommended.sum()), border=True)
        for note in notes:
            st.warning(note)
        st.dataframe(output, hide_index=True, alt="Student predictions, probabilities, and review flags")
        st.download_button("Download predictions", csv_bytes(output), "student_predictions.csv", "text/csv", type="primary")
        st.caption("Review is flagged for Needs support, model probability below 60%, or missing input values. This is a project heuristic, not a validated intervention threshold.")
else:
    st.session_state.pop("batch_result", None)
    st.info("Choose the example profiles or upload a CSV to begin.", icon=":material/upload_file:")
