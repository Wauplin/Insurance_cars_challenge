import pandas as pd
import streamlit as st

from constants import DAMAGE_PREDICTOR_PICKLE_PATH, DATA_PATH
from src.vehicule_model_aggregator import VehiculeModelAggregator
from src.damage_predictor import DamagePredictor
from utils.data_processing import train_test_split
from utils.streamlit import st_load_data

import plotly.express as px


def model_training():
    st.write("# Model training")

    with st.spinner(f"Load data from {DATA_PATH}"):
        df = st_load_data(DATA_PATH)

    st.write(
        """
    As seen in the **Data Exploration Report**, the data that we have is not very well correlated to the target value.
    Therefore, the statisticals models that I trained perform poorly. In this section, I will explain the main steps
    I implemented with the ins and outs behind them. There is of course a lot more that can be tested but the focus
    was on functionalities.
    """
    )

    st.write("## Train test split")

    st.write(
        """
    First step is to split the dataset between train and test sets. All preprocessing and training parameters must be
    computed on the train set only and then applied to the test set to evaluate the performance of the model.

    Instead of randomly sampling the rows, I chose to divide the data based on the households in such a way that a
    household can't own a vehicule in train set and a vehicule in the test set. That could have been a (low) source of bias
    for the model. Another method could have been to train the data only on vehicules insured in 2005 and 2006 and apply the
    model on vehicules from the 2007 dataset. By doing this, we would have proved that a model trained on a given year can
    be reused on the next year (which is highly interesting from a business point of vue). Since the results were already
    poor, I did not implemented this second split.

    The train/test ratio is thereafter set to **80% train/20% test**.
    """
    )

    train_test_split(df, test_ratio=0.2)

    st.write("## Preprocessing")

    st.write(
        """
    ### Steps


    Preprocessing steps have already been partially described in the **Data Exploration Report**. I implemented them in a
    `DamagePreprocessor` object that is created from a train DataFrame and that stores statistics useful from preprocessing.

    Following steps are applied :
    - DataFrame is copied to avoid any changes in the initial DataFrame. This can lead to a memory overload if not necessary. It depends on the backend architecture we want.
    - A `has_claim` boolean target feature is computed (`has_claim` is True if `Claim_Amount` is positive).
    - A `log_claim_amount` numerical target feature is computed (see Data Exploration Report).
    - The `age` feature is computed from `Model_Year` and `Calendar_Year`. It is then reduced (substract mean, divide by standard deviation). Mean and std are computed on train set only.
    - The `Aggregate_Car_Model` is computed from the car submodel. A `VehiculeModelAggregator` is implemented to do this work. For further details, see below.
    - Once all features have been created, dummy variables are computed from categorical features.
    - At inference time, some columns can be missing (due to the `get_dummies` function that creates on the fly the dummy column names). Therefore, dataframe is reindexed with original columns (filled with zeros). *Warning : the implementation here can be broken if a new category value is given to the preprocessing. In a production environment, this case must be considered.*
    - The top 20 features correlated with the target values are selected, the other are dropped. This filtering is an attempt to avoid model overfitting. It is of course questionnable and could be done in a more scientific way but I did not spend time to evaluate the effect of it.
    - Finally, features are reordered as they were during training.
    """
    )

    st.write("### VehiculeModelAggregator")

    st.write(
        """
    The `VehiculeModelAggregator` is created as follow :
    - It takes as input a list of submodels (eg. `Make.Model.Submodel` or `X.A.0`) with an associated count.
    - Given a defined threshold (here : `1000`), all submodels with enough vehicules are kept as they are.
    - The remaining submodels are grouped by models.
    - If a model contains enough vehicules, it is kept as `Make.Model.other`.
    - The remaining models are grouped by makers.
    - If a make contains enough vehicules, it is kept as `Make.other`.
    - The remaining vehicules are grouped in a `.other` category.

    Aggregation is computed on the train set only.

    At inference time, the `VMA` map all submodels to its Aggregate_Car_Model. If vehicule submodel is unknown,
    `.other` category is assigned by default.
    """
    )

    st.write("#### Results :")

    with st.spinner(f"Train VehiculeModelAggregator"):
        vma = VehiculeModelAggregator.from_series(df["Blind_Submodel"])

    st.write("*Initial Submodel counts :*")
    st.write(df["Blind_Submodel"].value_counts())

    st.write("*Aggregated models counts :*")
    st.write(
        pd.Series(vma.model_counts, name="Aggregate_Car_Model").sort_values(
            ascending=False
        )
    )

    st.write("## Training")

    with st.spinner(f"Train DamagePredictor"):
        predictor = DamagePredictor.from_dataframe(df)
        scores = predictor.scores()

    with st.spinner("Save DamagePredictor to disk"):
        predictor.to_pickle(DAMAGE_PREDICTOR_PICKLE_PATH)
        st.info(
            f"Predictor has been successfully saved to {DAMAGE_PREDICTOR_PICKLE_PATH} to be reused in **Predictions** tab."
        )

    st.write(
        """
    ### Has_damage model

    The `has_damage_model` is a binary classification task with a very imbalanced dataset (>99% zeros).
    I chose to use the `BalancedRandomForestClassifier` from the **ImbLearn** library. The idea of this model
    is to randomly undersample the over-represented category. This tends to balance the dataset and avoid
    any over-representation of a category over the other at inference time.

    I did not spend time on finetuning the model hyperparameters.
    """
    )

    st.write(
        """
    #### Performances

    In this special case, the model must not be evaluated using an `accuracy` metric. Indeed,
    a dumb model that is always outputting a negative response would still have >99% accuracy.
    However, other methods exists :
    - `precision` : measures how reliable is the model when it predicts a positive value.
    - `recall` : measures how much positive values the model predicts as such.
    - `f1-score` : a geometric average of precision and recall. Useful to evaluate the model based on a single value. `AUC` could have been another good metric.
    -  `confusion_matrix` : not a metric. Matrix that shows immediately the misclassfied elements.
    """
    )

    has_damage_scores = scores["has_damage"]
    metrics = pd.DataFrame(
        [
            [
                has_damage_scores["precision_train"],
                has_damage_scores["recall_train"],
                has_damage_scores["f1_train"],
            ],
            [
                has_damage_scores["precision_test"],
                has_damage_scores["recall_test"],
                has_damage_scores["f1_test"],
            ],
        ],
        columns=["precision", "recall", "f1-score"],
        index=["train", "test"],
    )

    st.write("*Metrics (1.0 is perfect):*")
    st.write(metrics)
    st.write("*Train Confusion Matrix :*")
    st.write(has_damage_scores["confusion_matrix_train"])
    st.write("*Test Confusion Matrix :*")
    st.write(has_damage_scores["confusion_matrix_test"])

    st.write(
        """
    ### Damage_amount model

    The `damage_amount_model` is a regression task on a smaller dataset (only ~1k rows instead of ~100k).
    I chose to use a `LinearRegression` model from the **sklearn** library. An advantage of this model
    is that it is very simple to interpret the results just by looking at its parameters. It is also
    fast to train and therefore is a good first model to use as a baseline. However, it performs poorly
    on datasets that are not linearly correlated.

    The `LinearRegression` is ran on log values in order to regularize the model. Predictions are then exponentially rescaled.

    I did not spend time on finetuning the model hyperparameters.
    """
    )

    st.write(
        """
    #### Performances

    Models performances can be measured using the `mean_squared_error`.
    Another good representation is to draw a Scatter Plot with ground truths values on x-axis and predicted values on y-axis.
    If model performs well, data points must be aligned on a straight line passing through the origin and of slope 1.
    """
    )

    damage_amount_scores = scores["damage_amount"]
    st.write(f"*Train MSE (lower is better) : * {damage_amount_scores['mse_train']}")
    st.write(f"*Test MSE (lower is better) : * {damage_amount_scores['mse_test']}")

    st.write("*Train scatter plot*")
    fig = px.scatter(damage_amount_scores["train_data_points"], x="truth", y="preds")
    st.write(fig)

    st.write("*Test scatter plot*")
    fig = px.scatter(damage_amount_scores["test_data_points"], x="truth", y="preds")
    st.write(fig)
