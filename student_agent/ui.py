"""Shared presentation helpers; no model logic lives in the UI."""
import altair as alt
import pandas as pd
import streamlit as st

from student_agent.data import DataError, LABELS

COLORS = ["#CF8755", "#5E75A8", "#087F73"]


def ready_agent():
    agent = st.session_state.agent
    if agent.run is None:
        try:
            with st.spinner("Agent is checking data, training, and evaluating…"):
                agent.execute("train")
        except (DataError, ValueError) as exc:
            st.error(str(exc), icon=":material/error:")
            st.caption("Correct the training CSV or clear the upload to use the demo.")
            st.stop()
    return agent


def metrics_row(metrics):
    with st.container(horizontal=True):
        st.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}", border=True)
        st.metric("Macro F1", f"{metrics['macro_f1']:.3f}", border=True)
        st.metric("Baseline accuracy", f"{metrics['baseline_accuracy']:.1%}", border=True)
        st.metric("Test students", metrics["test_rows"], border=True)


def band_chart(frame: pd.DataFrame, value: str, height: int = 240):
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=6, size=32).encode(
        x=alt.X(f"{value}:Q", title=value),
        y=alt.Y("Performance:N", sort=LABELS, title=None),
        color=alt.Color("Performance:N", scale=alt.Scale(domain=LABELS, range=COLORS), legend=None),
        tooltip=["Performance:N", alt.Tooltip(f"{value}:Q", format=".3f" if value == "Probability" else "d")],
    ).properties(height=height)


def trace(agent):
    st.caption("A record of tool calls and outcomes. Newest steps appear first.")
    st.dataframe(pd.DataFrame(agent.events[::-1]), hide_index=True, width="stretch", alt="Agent tool execution log")
