import numpy as np
import streamlit as st

from constants import DATA_PATH, EPSILON
from src.vehicule_model_aggregator import VehiculeModelAggregator
from utils.streamlit import st_load_data, write_df, display_distribution


def data_exploration_dynamic():
    st.write("# Dynamic Data Exploration")

    with st.spinner(f"Load data from {DATA_PATH}"):
        df = st_load_data(DATA_PATH)
        st.write("*Dataset*")
        write_df(df)

    # st.write("## Features distribution")

    df["has_claim"] = df["Claim_Amount"] > 0
    df["log_claim_amount"] = np.log(df["Claim_Amount"] + EPSILON)
    df["age"] = df["Calendar_Year"] - df["Model_Year"]

    vma = VehiculeModelAggregator.from_series(df["Blind_Submodel"], threshold=1000)
    df["Aggregate_Car_Model"] = vma.map(df["Blind_Submodel"])

    st.info("Select some features in the list above to display their distributions.")
    for feature in st.multiselect("Features to explore", sorted(df.columns)):
        display_distribution(
            df=df, title=f"{feature} distribution", column_name=feature
        )
