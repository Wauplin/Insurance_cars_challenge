import numpy as np
from pathlib import Path
from typing import Union

PathType = Union[str, Path]

TABLE_NAME = "podatki"
DATA_PATH = Path("podatki/podatki.csv")

STORE_DIRPATH = Path("tmp_store")
STORE_DIRPATH.mkdir(parents=True, exist_ok=True)
VMA_CONFIG_PATH = STORE_DIRPATH / "vma_config.json"
DAMAGE_PREPROCESSOR_CONFIG_PATH = STORE_DIRPATH / "damage_preprocessor_config.json"
DAMAGE_PREDICTOR_PICKLE_PATH = STORE_DIRPATH / "damage_predictor.pkl"

EPSILON = np.finfo(np.float32).eps

ID_COLUMNS = [
    "Row_ID",
    "Household_ID",
    "Vehicle",
]

SPLIT_COLUMN = "split"

TARGET_COLUMNS = [
    "Claim_Amount",
    "has_claim",
    "log_claim_amount",
]

RAW_CONTINUOUS_FEATURES = [
    "Model_Year",
    "Calendar_Year",
    "Var1",
    "Var2",
    "Var3",
    "Var4",
    "Var5",
    "Var6",
    "Var7",
    "Var8",
    "NVVar1",
    "NVVar2",
    "NVVar3",
    "NVVar4",
]

CONTINUOUS_FEATURES = [
    "age",
    "Var1",
    "Var2",
    "Var3",
    "Var4",
    "Var5",
    "Var6",
    "Var7",
    "Var8",
    "NVVar1",
    "NVVar2",
    "NVVar3",
    "NVVar4",
]

RAW_CATEGORICAL_FEATURES = [
    "Blind_Make",
    "Blind_Model",
    "Blind_Submodel",
    "Cat1",
    "Cat2",
    "Cat3",
    "Cat4",
    "Cat5",
    "Cat6",
    "Cat7",
    "Cat8",
    "Cat9",
    "Cat10",
    "Cat11",
    "Cat12",
    "NVCat",
]

CATEGORICAL_FEATURES = [
    "Aggregate_Car_Model",
    "Cat1",
    "Cat2",
    "Cat3",
    "Cat4",
    "Cat5",
    "Cat6",
    "Cat7",
    "Cat8",
    "Cat9",
    "Cat10",
    "Cat11",
    "Cat12",
    "NVCat",
]
