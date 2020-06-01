import streamlit as st

from dashboards.challenge_presentation import challenge_presentation
from dashboards.data_exploration_fixed import data_exploration_fixed
from dashboards.data_exploration_dynamic import data_exploration_dynamic
from dashboards.model_training import model_training
from dashboards.predictions import predictions
from dashboards.script_examples import script_examples

from utils.streamlit.login_page import LoginPage

with LoginPage() as session:
    if session:
        dashboards = {}  # Pages keys are ordered
        dashboards["Challenge Presentation"] = challenge_presentation
        dashboards["Data Exploration Report"] = data_exploration_fixed
        dashboards["Data Exploration (Dynamic)"] = data_exploration_dynamic
        dashboards["Model training"] = model_training
        dashboards["Predictions"] = predictions
        dashboards["Some script examples"] = script_examples

        dashboard = dashboards[st.sidebar.selectbox("Page", list(dashboards.keys()))]

        st.sidebar.markdown("---")

        if dashboard is not None:
            dashboard()
