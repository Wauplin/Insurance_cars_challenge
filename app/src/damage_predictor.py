import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel as LinearModelType
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_squared_error,
)

from constants import PathType, SPLIT_COLUMN, TARGET_COLUMNS
from src.damage_preprocessor import DamagePreprocessor
from utils.data_processing import train_test_split
from utils.sql import sql_to_df


class DamagePredictor:
    def __init__(
        self,
        preprocessor: DamagePreprocessor,
        has_damage_model: LinearModelType,
        damage_amount_model: LinearModelType,
        initial_df: Optional[pd.DataFrame] = None,
        preprocessed_df: Optional[pd.DataFrame] = None,
    ):
        self.preprocessor = preprocessor
        self.has_damage_model = has_damage_model
        self.damage_amount_model = damage_amount_model
        self.initial_df = initial_df
        self.preprocessed_df = preprocessed_df

    # Constructors
    @classmethod
    def from_sql(cls, table_name: str):
        return cls.from_dataframe(sql_to_df(table_name))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        initial_df = train_test_split(df, test_ratio=0.2)
        preprocessor = DamagePreprocessor.from_dataframe(df)
        preprocessed_df = preprocessor.preprocess(df)
        has_damage_model = cls._train_has_damage(preprocessed_df)
        damage_amount_model = cls._train_damage_amount(preprocessed_df)
        return cls(
            preprocessor=preprocessor,
            has_damage_model=has_damage_model,
            damage_amount_model=damage_amount_model,
            initial_df=initial_df,
            preprocessed_df=preprocessed_df,
        )

    @classmethod
    def from_pickle(cls, pickle_path: PathType):
        with Path(pickle_path).open("rb") as pickle_file:
            return pickle.load(pickle_file)

    def to_pickle(self, pickle_path: PathType):
        with Path(pickle_path).open("wb") as pickle_file:
            pickle.dump(self, pickle_file)

    # Main methods
    def predict_vehicule(
        self, vehicule_year: int, vehicule: str
    ) -> Dict[str, Union[float, bool]]:
        preprocessed = self.preprocessor.preprocess_vehicule(
            vehicule_year=vehicule_year, vehicule=vehicule
        )
        has_damage = self._predict_has_damage(preprocessed)[0]
        damage_amount = self._predict_damage_amount(preprocessed)[0]
        return {
            "has_damage": has_damage,
            "damage_amount": damage_amount,
        }

    def scores(self):
        return {
            "has_damage": self._scores_has_damage(),
            "damage_amount": self._scores_damage_amount(),
        }

    # Internal methods
    def _predict_has_damage(self, X: pd.DataFrame):
        return self.has_damage_model.predict(self.__to_features(X))

    def _predict_damage_amount(self, X: pd.DataFrame):
        # Predicted value is log-regularized
        return np.exp(self.damage_amount_model.predict(self.__to_features(X)))

    def _scores_has_damage(self):
        if self.preprocessed_df is None:
            raise ValueError("Preprocessed DF not set. Cannot get training scores.")
        X_train, X_test, Y_train, Y_test = self.get_X_Y_split(
            self.preprocessed_df, "has_claim"
        )
        train_preds = self._predict_has_damage(X_train)
        test_preds = self._predict_has_damage(X_test)

        return {
            "f1_train": f1_score(Y_train, train_preds),
            "f1_test": f1_score(Y_test, test_preds),
            "confusion_matrix_train": confusion_matrix(Y_train, train_preds),
            "confusion_matrix_test": confusion_matrix(Y_test, test_preds),
            "precision_train": precision_score(Y_train, train_preds),
            "precision_test": precision_score(Y_test, test_preds),
            "recall_train": recall_score(Y_train, train_preds),
            "recall_test": recall_score(Y_test, test_preds),
        }

    def _scores_damage_amount(self):
        if self.preprocessed_df is None:
            raise ValueError("Preprocessed DF not set. Cannot get training scores.")
        preprocessed_claims = self.preprocessed_df[self.preprocessed_df["has_claim"]]

        X_train, X_test, Y_train, Y_test = self.get_X_Y_split(
            preprocessed_claims, "Claim_Amount"
        )

        train_preds = self._predict_damage_amount(X_train)
        test_preds = self._predict_damage_amount(X_test)

        return {
            "mse_train": mean_squared_error(Y_train, train_preds),
            "mse_test": mean_squared_error(Y_test, test_preds),
            "train_data_points": pd.DataFrame(
                np.stack([Y_train.values, train_preds], axis=1),
                columns=["truth", "preds"],
            ),
            "test_data_points": pd.DataFrame(
                np.stack([Y_test.values, test_preds], axis=1),
                columns=["truth", "preds"],
            ),
        }

    @classmethod
    def _train_has_damage(cls, preprocessed_df: pd.DataFrame) -> LinearModelType:
        X_train, X_test, Y_train, Y_test = cls.get_X_Y_split(
            preprocessed_df, "has_claim"
        )
        model = BalancedRandomForestClassifier()
        model.fit(X_train, Y_train)
        return model

    @classmethod
    def _train_damage_amount(cls, preprocessed_df: pd.DataFrame) -> LinearModelType:
        preprocessed_claims = preprocessed_df[
            preprocessed_df["has_claim"]
        ]  # Only claims
        X_train, X_test, Y_train, Y_test = cls.get_X_Y_split(
            preprocessed_claims, "log_claim_amount"
        )
        model = LinearRegression()
        model.fit(X_train, Y_train)
        return model

    @classmethod
    def get_X_Y_split(cls, df: pd.DataFrame, target: str):
        feature_columns = cls.__get_feature_columns(df)
        df_train = df[df[SPLIT_COLUMN] == "train"]
        df_test = df[df[SPLIT_COLUMN] == "test"]

        X_train = df_train[feature_columns]
        X_test = df_test[feature_columns]
        Y_train = df_train[target]
        Y_test = df_test[target]

        return X_train, X_test, Y_train, Y_test

    @classmethod
    def __to_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df[cls.__get_feature_columns(df)]

    @staticmethod
    def __get_feature_columns(df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if col not in TARGET_COLUMNS + [SPLIT_COLUMN]]
