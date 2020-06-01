import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    DATA_PATH,
    EPSILON,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    TARGET_COLUMNS,
)
import plotly.graph_objects as go

from src.vehicule_model_aggregator import VehiculeModelAggregator
from utils.data_processing import get_model_distribution
from utils.streamlit import st_load_data, write_df, display_distribution


def data_exploration_fixed():
    st.write("# Data Exploration Report")

    with st.spinner(f"Load data from {DATA_PATH}"):
        df = st_load_data(DATA_PATH)
        st.write("*Dataset*")
        write_df(df)

    st.write("## Features")

    st.write("### Available Features")

    st.write(
        """
    The proposed dataset contains car insurances. Each row of the dataframe describes an insured vehicule and (potentially) an associated claim amount.

    In detail, we have :
    - a `row_id`, `household_id` and `vehicle_id`. Row_ids are unique in the dataset while several cars can belong to the same household. `Vehicle_id` is then use to distinguish vehicules in a household.
    - the `year` during which the vehicule was insured.
    - the `model year` of the vehicule
    - the vehicule `make`, `model` and `submodel` of the vehicule. Given the number of submodels (several thousands) and the repartition (some models have thousands of vehicules whereas some models are represented by only one), some extra-work is required before using this information.
    - `12 categorical variables`, `1 ordered categorical variable` and `8 continuous variables` (already normalized) describing the vehicule
    - `1 categorical variable` and `4 continuous variables` (already normalized) not describing the vehivule. We can imagine that those variables describe for instance the driver's age, profession, salary,...
    - The `claim amount`. In most of the cases (>99%), this value is zero meaning no car accident has been declared.


    Given this dataset, the goal is to **predict damage**. I decided to split this problem into 2 models :
    - given a vehicule description, can we predict the fact that he has more chances to have an accident ?
    - if we assume the vehicle has an accident, what will be the damage ?

    As we will see later, both tasks are really hard to solve with the given data. We can assume that getting data more specific to the accidents that occured might help to get more precise predictions.
    """
    )

    st.write("### Hand-made Features")

    st.write(
        """
    In order to tackle this problem, we need to preprocess the raw features.
    """
    )

    st.write("#### Claims features")

    st.write(
        "I defined a `has_claim` boolean feature. This variable will be the one we want to predict in our first task."
    )

    df["has_claim"] = df["Claim_Amount"] > 0

    write_df(df["has_claim"].value_counts())
    display_distribution(
        df=df, title="Claims distribution", column_name="Claim_Amount",
    )

    st.write(
        "As we can see, the claim amount distribution varies a lot with values of different order of magnitude (from less than 50€ up to 5000€). This value spread might cause some unstability to the regression model we want to build. In order to regularize it, I decided to apply a log function to it."
    )

    df["log_claim_amount"] = np.log(df["Claim_Amount"] + EPSILON)
    display_distribution(
        df=df,
        title="Claims distribution (with log-regularization)",
        column_name="log_claim_amount",
    )

    st.write("#### Vehicule years features")

    display_distribution(
        df=df,
        title="Vehicule model year distribution (full dataset)",
        column_name="Model_Year",
    )
    display_distribution(
        df=df,
        title="Insurance year distribution (full dataset)",
        column_name="Calendar_Year",
    )

    st.write(
        """
    Vehicules are mostly recent ones at the time the insurance was taken (around 2005-2007). Some vehicules are older than 10 years and up to 20. In the following, I prefered to process the years as an `age` (difference between insurance year and model_year). Intuitively, a more recent vehicule might get more damages to reimburse but less accidents due newer technologies.
    """
    )

    df["age"] = df["Calendar_Year"] - df["Model_Year"]

    st.warning(
        "**Warning :** the age column appears to be negative in some cases. This comes from incoherent Insurance Year and Model Year. I decided to ignore this problem and keep all values."
        f"\n\n(df['age'] < 0).sum() = {(df['age'] < 0).sum()}"
    )

    st.write("#### Vehicule models features")

    st.write(
        """
    In the dataframe below, we can see that the submodel repartition is very imbalanced with categories having thousands of cars while others are unique. This is a big issue if we want to run statistics on submodels. Indeed, some vehicles might not be representative of their category.

    The solution I propose is to take advantage of the make/model/submodel hierarchy. If a submodel contains enough vehicules I keep the category as it is. Otherwise, I go back to the model category or even to the make if necessary. By recursively applying the method, I ensure that all categories contains at least a certain number of vehicules (1000 in my case). In our case, this reduces the number of categories from 2004 to 43 which implies less parameters in the model. A `VehiculeModelAggregator` object contains code to compute the aggregation but also load from/save to a config file for later reuse (see `Model Training` tab).
    """
    )

    model_distribution = get_model_distribution(df, "Blind_Submodel")
    st.write(
        f"*Distribution before aggregation ({len(model_distribution)} categories):*"
    )
    write_df(model_distribution, N=len(model_distribution))

    vma = VehiculeModelAggregator.from_series(df["Blind_Submodel"])
    df["Aggregate_Car_Model"] = vma.map(df["Blind_Submodel"])

    model_distribution = get_model_distribution(df, "Aggregate_Car_Model")
    st.write(
        f"*Distribution after aggregation ({len(model_distribution)} categories):*"
    )
    write_df(model_distribution, N=len(model_distribution))

    to_drop = [
        "Model_Year",
        "Calendar_Year",
        "Blind_Make",
        "Blind_Model",
        "Blind_Submodel",
    ]
    df = df.drop(to_drop, axis=1)
    st.write(f"Droped columns in the following : `{', '.join(to_drop)}`")

    st.write("#### Continuous features")

    st.write(
        """
    The continuous variables are the most convenient features in a linear model since they can be used as it is.

    To regularize the model, we can normalize the data (mean 0, std 1). In our case, this work has already been done for all features except `age`.
    """
    )

    mean_std = pd.concat(
        [df[CONTINUOUS_FEATURES].mean(), df[CONTINUOUS_FEATURES].std()], axis=1
    )
    mean_std.columns = ["mean", "std"]
    write_df(mean_std)

    df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()

    st.write("#### Categorical Features")

    st.write(
        """
    Only ordered categorical features can be processed as they are. For instance, a variable that describes the size of a vehicule (class 1 -> small cars, class 5 -> trucks) is ordered. In our case we have only one ordered categorical variable.

    The trick to deal with categorical features is to create dummy varables that represent a 1-hot encoding of the initial feature. For instance, a categorical feature with 3 classes (A, B, C) is splitted in `is_A`, `is_B` and `is_C` variables. Pandas has a really helpful method `get_dummies` to do so.
    """
    )

    df_with_dummies = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    st.write("*Dataset with dummy features :*")
    write_df(df_with_dummies, 500)

    st.write("## Features Filtering")

    FEATURES_COLUMNS = [
        col for col in df_with_dummies.columns if col not in ID_COLUMNS + TARGET_COLUMNS
    ]

    st.write(
        f"""
    Now that features are preprocessed, we end up with `{len(FEATURES_COLUMNS)}` features to feed our statistical model.
    However, some of them might by useless (doesn't explain target) while others might be redundant or highly correlated. This leads to heavier and less accurate models.

    To mitigate the risks, a good method is to compute the correlation matrix of our features :
    - if several features are highly correlated, we would like to drop some.
    - if some features are not correlated to the target, we would like to drop them as well.
    """
    )

    st.write("### Correlation Matrix")

    st.write("*Correlation matrix (absolute values, over all rows) :*")
    correlation_matrix = df_with_dummies[FEATURES_COLUMNS + TARGET_COLUMNS].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=np.abs(correlation_matrix),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
        )
    )
    st.write(fig)

    st.write("*Correlation matrix (absolute values, over claims only) :*")
    correlation_matrix_on_claims = df_with_dummies[df["has_claim"]][
        FEATURES_COLUMNS + TARGET_COLUMNS
    ].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=np.abs(correlation_matrix_on_claims),
            x=correlation_matrix_on_claims.columns,
            y=correlation_matrix_on_claims.columns,
        )
    )
    st.write(fig)

    st.write(
        """
    #### Interpretation
    - Continuous features describing the vehicule are highly correlated. A possible explanation is that they represent the height of the vehicule, its weight, its length, its width,... that are naturally related.
    - Some categorical features are also correlated to the 6 continuous features. However, this is less the case for variables that do not describe the vehicle, which makes sense.
    - `Claim_Amount` and `has_claim` are only weakly correlated to the features we have.
    - At least visually, to look at all the dataset or claims only does not seem to change a lot the correlation.
    """
    )

    st.write("### Filtering")

    top_features = np.abs(
        correlation_matrix[["has_claim", "log_claim_amount"]]
    ).sort_values(by="has_claim", ascending=False)
    st.write("*Top Features (ordered to predict `has_claim`, absolute values) :*")
    st.write(top_features)

    st.write(
        """
    - As expected, the features order is (almost) the same between `has_claim` and `log_claim_amount`.
    - Features are not correlated to target values (<2% even for the "best one"). This means that they don't explain well the data. Even without training it, we already know that a Linear Model will be unable to learn anything from it (e.g. it will not converge). At best, it will get some broad trends.
    - From this point, we might want to take only top N features to train the model. Other methods exist. We could compute the p-value for each feature and keep only features with p-value below 5%. Another possibility is to use a Lasso Regularization directly when training the Linear Model. This would force some parameters to zero which would be equivalent to filtering them out.
    - I stopped there for the data exploration part. Another tab of the dashboard (`Dynamic Data Exploration`) is dedicated to the exploration of the dataset and especially features distribution. Next step is to train the models (`Train Models` tab).
    - One last thing : this section is about data exploration only. When training the models, we would like to be more careful on the split between train and test sets. Especially, the mapping between vehicule `Submodel` and `Aggregate_Model` must be computed on the train set only. Same caution must be taken for the normalization of the data (`age`). The `DamagePreprocessor` object takes care of that.
    """
    )
