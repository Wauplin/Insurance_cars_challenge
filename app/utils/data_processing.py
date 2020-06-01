import numpy as np
import pandas as pd
import random

from constants import PathType


def load_data(path: PathType) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna(
        "?"
    )  # In practice : only some categorical values are missing. Otherwise, would need more care.
    df = df.drop_duplicates()  # In practice : no duplicate (even if we remove row_id)
    return df


def get_model_distribution(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    distribution = (
        df.groupby(column_name)
        .agg(
            nb_vehicules=pd.NamedAgg(column="Row_ID", aggfunc="count"),
            mean_age=pd.NamedAgg(column="age", aggfunc=np.mean),
            median_age=pd.NamedAgg(column="age", aggfunc=np.median),
            std_age=pd.NamedAgg(column="age", aggfunc=np.std),
            nb_claims=pd.NamedAgg(column="has_claim", aggfunc="sum"),
            mean_claim_amount=pd.NamedAgg(
                column="Claim_Amount", aggfunc=lambda x: np.sum(x) / np.count_nonzero(x)
            ),
        )
        .sort_values("nb_vehicules", ascending=False)
    )
    distribution["nb_claim_per_100_vehicules"] = (
        100 * distribution["nb_claims"] / distribution["nb_vehicules"]
    )
    return distribution


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2) -> pd.DataFrame:
    """
    Train test split that separate households between train and test sets.
    """
    assert "Household_ID" in df
    households = list(df["Household_ID"].unique())
    test_households = random.sample(households, int(test_ratio * len(households)))
    df["split"] = "train"
    df.loc[df["Household_ID"].isin(test_households), "split"] = "test"
    return df
