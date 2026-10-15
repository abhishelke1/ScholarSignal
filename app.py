"""Streamlit entrypoint. Run: python -m streamlit run app.py"""
import hashlib

import streamlit as st

from student_agent.agent import PerformanceAgent
from student_agent.data import DataError, load_demo, read_csv

st.set_page_config(page_title="ScholarSignal · AI lab", page_icon=":material/school:", layout="wide")


@st.cache_data(max_entries=1)
def demo_data():
    return load_demo()


with st.sidebar:
    st.title("ScholarSignal", anchor=False)
    st.caption("STUDENT PERFORMANCE · AI LAB")
    st.divider()
    mode = st.segmented_control("Prediction mode", ["With prior grades", "Early warning"], default="With prior grades", key="mode") or "With prior grades"
    st.caption("Prior-grade mode uses G1 and G2. Early warning excludes both grades.")
    with st.expander("Training dataset", expanded=False):
        upload = st.file_uploader("Replace the demo dataset", type="csv", key="training_upload", help="UCI-style schema. At least 60 rows, with 10 examples per grade band. Maximum 5 MB.")
        st.caption("Clear the upload to restore the bundled UCI mathematics data.")
    st.divider()
    st.caption("LOCAL BY DEFAULT")
    st.markdown("No API key needed.\n\nYour uploads stay in this app session.")

try:
    payload = upload.getvalue() if upload else None
    data = read_csv(payload) if payload is not None else demo_data()
    source = upload.name if upload else "UCI Student Performance · Mathematics"
    identity = (hashlib.sha256(payload).hexdigest() if payload is not None else "uci-mat-v1", mode)
    if st.session_state.get("identity") != identity:
        st.session_state.agent = PerformanceAgent(data, mode, source)
        st.session_state.identity = identity
        for key in ("single_result", "batch_result", "submitted_profile", "command_response"):
            st.session_state.pop(key, None)
except DataError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

page = st.navigation([
    st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("app_pages/predict.py", title="Predict a student", icon=":material/person_search:"),
    st.Page("app_pages/batch.py", title="Batch predictions", icon=":material/table_view:"),
    st.Page("app_pages/insights.py", title="Model insights", icon=":material/analytics:"),
    st.Page("app_pages/lab.py", title="Lab guide", icon=":material/menu_book:"),
], position="top")
page.run()
