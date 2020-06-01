import streamlit as st

from constants import DAMAGE_PREDICTOR_PICKLE_PATH
from src.damage_predictor import DamagePredictor


@st.cache(allow_output_mutation=True)
def load_model():
    return DamagePredictor.from_pickle(DAMAGE_PREDICTOR_PICKLE_PATH)


def predictions():
    st.write("# Predictions")

    with st.spinner(f"Load model from {DAMAGE_PREDICTOR_PICKLE_PATH}"):
        predictor = load_model()
        vma = predictor.preprocessor.vma
        st.info(f"Model loaded from `{DAMAGE_PREDICTOR_PICKLE_PATH}`")

    st.write(
        """
    On this page, the DamagePredictor model trained previously is used to predict damage based on two parameters : `vehicule year` and `vehicule model`.
    However, this is only a toy example since features filtering is removing those variables to predict the damage amount.
    I could have used other variables as inputs but year and model seemed to me to be easily understandable.
    In a real-life scenario we would have taken the time to run things otherwise but I ran out of time to do so.
    """
    )

    st.write("## Damage predictor")

    vehicule_year = st.number_input(
        "Vehicule model year", min_value=1980, max_value=2007, value=2000
    )
    vehicule_model = st.text_input("Vehicule model", value="AU.14.1")

    aggregated_model = vma.model_mapping[vehicule_model]
    predictions = predictor.predict_vehicule(vehicule_year, vehicule_model)

    st.write(f"**Aggregated car model :** {aggregated_model}")
    st.write(f"**Predicted has_damage** : {bool(predictions['has_damage'])}")
    st.write(
        f"**Predicted damage_amount** *(if any)* : {round(predictions['damage_amount'], 1)}"
    )
