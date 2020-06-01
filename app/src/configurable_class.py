import json
from typing import List, Dict
from pathlib import Path

from constants import PathType


class ConfigurableClass:
    CONFIG_ARGS: List[str] = []

    def __init__(self, **kwargs):
        for key in self.CONFIG_ARGS:
            assert key in kwargs

        for key, value in kwargs.items():
            assert key in self.CONFIG_ARGS
            setattr(self, key, value)

    @classmethod
    def from_config_file(cls, config_path: PathType):
        with Path(config_path).open("r") as config_file:
            return cls.from_config(json.load(config_file))

    def to_config_file(self, config_path: PathType) -> None:
        with Path(config_path).open("w") as config_file:
            json.dump(self.to_config(), config_file)

    @classmethod
    def from_config(cls, config: Dict):
        return cls(**config)

    def to_config(self) -> Dict:
        config = {key: getattr(self, key) for key in self.CONFIG_ARGS}
        for key, value in config.items():
            if isinstance(value, ConfigurableClass):
                config[key] = value.to_config()
        return config
