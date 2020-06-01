import numpy as np
import pandas as pd

from constants import (
    CATEGORICAL_FEATURES,
    EPSILON,
    ID_COLUMNS,
    SPLIT_COLUMN,
    RAW_CATEGORICAL_FEATURES,
    RAW_CONTINUOUS_FEATURES,
    TARGET_COLUMNS,
)
from src.configurable_class import ConfigurableClass
from src.vehicule_model_aggregator import VehiculeModelAggregator


class DamagePreprocessor(ConfigurableClass):

    CONFIG_ARGS = [
        "default_values",
        "age_mean",
        "age_std",
        "vma",
        "all_features",
        "top_features",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.vma, dict):
            self.vma = VehiculeModelAggregator.from_config(self.vma)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        Precompute statistics on the train dataset in order to initialize preprocessing.
        """
        # Get train dataframe only
        if SPLIT_COLUMN in df.columns:
            df = df[df[SPLIT_COLUMN] == "train"]
        df = df.copy()  # Do not want to modify argument

        # Get most common values -> to be reused as default value in inference time
        categorical_modes = df[RAW_CATEGORICAL_FEATURES].mode().iloc[0].to_dict()
        continuous_modes = df[RAW_CONTINUOUS_FEATURES].median().to_dict()
        default_values = dict(continuous_modes, **categorical_modes)

        # Claims features
        df["has_claim"] = df["Claim_Amount"] > 0
        df["log_claim_amount"] = np.log(df["Claim_Amount"] + EPSILON)

        # Age feature
        df["age"] = df["Calendar_Year"] - df["Model_Year"]
        age_mean = df["age"].mean()  # Compute statistics on train dataset
        age_std = df["age"].std()  # Compute statistics on train dataset
        df = df.drop(["Model_Year", "Calendar_Year"], axis=1)

        # Model aggregation
        vma = VehiculeModelAggregator.from_series(df["Blind_Submodel"])
        df["Aggregate_Car_Model"] = vma.map(df["Blind_Submodel"])
        df = df.drop(["Blind_Make", "Blind_Model", "Blind_Submodel"], axis=1)

        # To dummies
        df_with_dummies = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)

        all_features = [
            col
            for col in df_with_dummies.columns
            if col not in ID_COLUMNS + TARGET_COLUMNS + [SPLIT_COLUMN]
        ]

        # /!\ Correlation matrix and top features on Train test only /!\
        correlation_matrix = np.abs(
            df_with_dummies[df["has_claim"]][all_features + TARGET_COLUMNS].corr()
        )
        top_features = list(
            correlation_matrix["log_claim_amount"]
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        top_features = [feat for feat in top_features if feat in all_features]

        return cls(
            default_values=default_values,
            age_mean=age_mean,
            age_std=age_std,
            vma=vma,
            all_features=all_features,
            top_features=top_features,
        )

    def preprocess_vehicule(self, vehicule_year: int, vehicule: str) -> pd.DataFrame:
        vehicule_split = vehicule.split(".")
        if len(vehicule_split) == 1:
            submodel = vehicule
            model = vehicule + ".0"
            make = vehicule + ".0.0"
        elif len(vehicule_split) == 2:
            submodel = vehicule_split[0]
            model = vehicule
            make = vehicule + ".0"
        elif len(vehicule_split) == 3:
            submodel = vehicule_split[0]
            model = vehicule_split[0] + "." + vehicule_split[1]
            make = vehicule
        else:
            raise ValueError(f"Wrong vehicule make/model/submodel given : {vehicule}.")

        df = pd.DataFrame(self.default_values, index=[0])
        df["Model_Year"] = vehicule_year
        df["Blind_Submodel"] = submodel
        df["Blind_Model"] = model
        df["Blind_Make"] = make
        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to given dataset accordingly to pre-computed statistics.
        """
        df = df.copy()  # Do not want to modify argument

        # Claims features
        if "Claim_Amount" in df.columns:
            df["has_claim"] = df["Claim_Amount"] > 0
            df["log_claim_amount"] = np.log(df["Claim_Amount"] + EPSILON)
            target_columns = ["has_claim", "Claim_Amount", "log_claim_amount"]
        else:
            target_columns = []

        # Age feature
        df["age"] = df["Calendar_Year"] - df["Model_Year"]
        df["age"] = (
            df["age"] - self.age_mean
        ) / self.age_std  # Apply statistics on full dataset
        df = df.drop(["Model_Year", "Calendar_Year"], axis=1)

        # Model aggregation
        df["Aggregate_Car_Model"] = self.vma.map(df["Blind_Submodel"])
        df = df.drop(["Blind_Make", "Blind_Model", "Blind_Submodel"], axis=1)

        # To dummies
        df_with_dummies = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
        df_with_dummies = df_with_dummies.reindex(
            columns=self.all_features + target_columns + [SPLIT_COLUMN], fill_value=0
        )

        # Keep top features only
        df_with_dummies = df_with_dummies[
            self.top_features + target_columns + [SPLIT_COLUMN]
        ]

        # Return Preprocessed DataFrame
        return df_with_dummies
