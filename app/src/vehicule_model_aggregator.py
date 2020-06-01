from collections import defaultdict
import pandas as pd
from typing import Dict, List

from src.configurable_class import ConfigurableClass

ModelCountsType = Dict[str, int]
ModelMappingType = Dict[str, str]


def lambda_other_fun():  # Avoid conflicts with pickle
    return ".other"


class VehiculeModelAggregator(ConfigurableClass):
    CONFIG_ARGS = ["threshold", "model_counts", "model_mapping"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_mapping = defaultdict(lambda_other_fun, self.model_mapping)

    def map(self, series: pd.Series):
        return series.map(self.model_mapping)

    @classmethod
    def from_series(cls, series: pd.Series, threshold: int = 1000):
        models = series.value_counts().to_dict()
        model_counts = cls._compute_model_aggregation(models, threshold=threshold)
        model_mapping = cls._compute_models_mapping(list(models.keys()), model_counts)
        return cls(
            threshold=threshold, model_counts=model_counts, model_mapping=model_mapping
        )

    @staticmethod
    def _model_parent(submodel: str) -> str:
        return ".".join(submodel.rstrip(".other").split(".")[:-1]) + ".other"

    @classmethod
    def _compute_model_aggregation(
        cls, models_counts: ModelCountsType, threshold: int
    ) -> ModelCountsType:
        aggregate = defaultdict(int)
        for model, count in models_counts.items():
            if count > threshold:
                aggregate[model] = count
            else:
                aggregate[cls._model_parent(model)] += count
        if aggregate.keys() == models_counts.keys():
            ordered_aggregate = {}
            for key in sorted(list(aggregate.keys())):
                ordered_aggregate[key] = aggregate[key]
            return ordered_aggregate
        else:
            return cls._compute_model_aggregation(aggregate, threshold=threshold)

    @classmethod
    def _compute_models_mapping(
        cls, models_list: List[str], model_counts: ModelCountsType
    ) -> ModelMappingType:
        if not models_list:
            return {}
        mapping = {}
        imodel = 0
        model = models_list[imodel]
        current_model = model
        while True:
            if current_model in model_counts:
                mapping[model] = current_model
                if imodel >= len(models_list):
                    break
                else:
                    model = models_list[imodel]
                    current_model = model
                    imodel += 1
            else:
                current_model = cls._model_parent(current_model)
        return mapping
