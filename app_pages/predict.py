import pandas as pd
import streamlit as st
import altair as alt

from student_agent.data import BINARY, DISPLAY, LABELS
from student_agent.ui import band_chart, ready_agent

st.title("A clearer picture of one student")
st.markdown("Enter the information available at prediction time. The agent will return a grade band, evidence, and a study checklist.")
agent = ready_agent()
with st.form("student_profile"):
    st.subheader("Learning profile")
    a, b, c = st.columns(3)
    with a:
        studytime = st.selectbox("Weekly study time", [1, 2, 3, 4], index=1, format_func=lambda x: {1: "Less than 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "More than 10 hours"}[x], key="studytime")
        failures = st.number_input("Past class failures", 0, 3, 0, key="failures")
    with b:
        absences = st.number_input("Absences recorded so far", 0, 93, 4, key="absences")
        traveltime = st.selectbox("Travel time to school", [1, 2, 3, 4], format_func=lambda x: {1: "Less than 15 minutes", 2: "15–30 minutes", 3: "30–60 minutes", 4: "More than 60 minutes"}[x], key="traveltime")
    with c:
        if agent.mode == "With prior grades":
            g1 = st.number_input("First-period grade / 20", 0, 20, 12, key="G1")
            g2 = st.number_input("Second-period grade / 20", 0, 20, 12, key="G2")
        else:
            st.info("Early warning uses study and support information without previous grades.")
    st.subheader("Support and access")
    support = {}
    support_columns = st.columns(3)
    for i, field in enumerate(BINARY):
        with support_columns[i % 3]:
            support[field] = st.selectbox(DISPLAY[field], ["yes", "no"], index=1 if field in ("schoolsup", "activities") else 0, key=field)
    submitted = st.form_submit_button("Predict performance", type="primary", icon=":material/auto_awesome:")
if submitted:
    profile = {"studytime": studytime, "failures": failures, "absences": absences, "traveltime": traveltime, **support}
    if agent.mode == "With prior grades":
        profile.update(G1=g1, G2=g2)
    st.session_state.single_result = agent.execute("predict", pd.DataFrame([profile]))
    st.session_state.submitted_profile = profile

if "single_result" in st.session_state:
    result = st.session_state.single_result
    prediction = result["predictions"].iloc[0]
    st.divider()
    left, right = st.columns([1, 1.2], gap="large")
    with left, st.container(border=True):
        st.caption("PREDICTED GRADE BAND")
        st.subheader(prediction.prediction)
        st.metric("Model probability", f"{prediction.model_probability:.1%}")
        if prediction.review_recommended:
            st.warning("Teacher review recommended", icon=":material/person:")
        else:
            st.success("Use this estimate alongside current classwork.", icon=":material/check_circle:")
        st.caption("Uncalibrated model probabilities can be overconfident. These are not guarantees.")
    with right:
        probabilities = pd.DataFrame({"Performance": LABELS, "Probability": [prediction[f"p_{label.lower().replace(' ', '_')}"] for label in LABELS]})
        st.altair_chart(band_chart(probabilities, "Probability"), alt="Predicted probabilities for all three grade bands")
    st.caption("Results reflect the last submitted profile. Submit again after editing the form.")
    with st.expander("View submitted profile"):
        st.json(st.session_state.submitted_profile)
    st.subheader("Why this prediction?")
    explanation = result["explanation"]
    st.caption(f"Evidence for {explanation['winner']} versus {explanation['runner_up']}. Positive values favour the prediction; negative values favour the alternative. This describes the model, not cause and effect.")
    evidence = pd.DataFrame(explanation["evidence"])
    chart = alt.Chart(evidence).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("log_odds:Q", title="Contribution to log odds"),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.condition(alt.datum.log_odds > 0, alt.value("#087F73"), alt.value("#CF8755")),
        tooltip=["label:N", alt.Tooltip("log_odds:Q", format=".3f")],
    ).properties(height=300)
    st.altair_chart(chart, alt="Feature contributions to the model prediction versus its second choice")
    st.caption(f"The class-prior contribution is {explanation['prior_log_odds']:.3f}. Adding it to every feature contribution gives the total log posterior odds.")
    with st.container(border=True):
        st.subheader("A practical plan for this week", icon=":material/checklist:")
        for i, action in enumerate(result["study_plan"], 1):
            st.markdown(f"**{i}.** {action}")
        st.caption("A rule-based checklist for discussion with a teacher; the model has not measured the effects of these actions.")
else:
    st.info("Complete the profile and select Predict performance to see the agent's analysis.", icon=":material/person_search:")
