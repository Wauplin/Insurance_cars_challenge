import pandas as pd
from pathlib import Path
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import PathType
from utils.data_processing import load_data


@st.cache(allow_output_mutation=True)
def st_load_data(path: PathType) -> pd.DataFrame:
    return load_data(path)


def write_df(df: pd.DataFrame, N: int = 1000):
    st.write(df.head(N))
    if len(df) > N:
        st.warning(f"Only {N} rows displayed out of {len(df)}.")


def write_md(md_path: PathType):
    with Path(md_path).open() as f:
        st.write(f.read())


def display_distribution(df: pd.DataFrame, title: str, column_name: str):
    """ convenient method to display the distribution of a feature"""
    st.write(f"*{title} :*")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Histogram(x=df[column_name], name="Full dataset"), secondary_y=False,
    )
    fig.add_trace(
        go.Histogram(x=df[df["has_claim"]][column_name], name="Claims only"),
        secondary_y=True,
    )
    fig.update_layout(barmode="overlay")  # Overlay both histograms
    fig.update_traces(opacity=0.45)  # Reduce opacity to see both histograms
    st.write(fig)
