import streamlit as st

from utils.streamlit import write_md


def script_examples():
    st.write("# Script examples")

    st.write(
        """
    Finally, on this last page, I expose a few script examples to reuse the code and objects implemented for this challenge.
    """
    )

    write_md("doc/examples.md")
