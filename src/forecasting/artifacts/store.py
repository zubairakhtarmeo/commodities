from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump, load


@dataclass
class ArtifactStore:
    root: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save_model(self, *, asset_id: str, model_name: str, estimator) -> Path:
        self.ensure()
        path = self.root / asset_id / f"{model_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(estimator, path)
        return path

    def load_model(self, *, asset_id: str, model_name: str):
        path = self.root / asset_id / f"{model_name}.joblib"
        return load(path)

    def save_metrics(self, *, asset_id: str, model_name: str, df: pd.DataFrame) -> Path:
        self.ensure()
        path = self.root / asset_id / f"{model_name}_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def save_importances(self, *, asset_id: str, model_name: str, df: pd.DataFrame) -> Path:
        self.ensure()
        path = self.root / asset_id / f"{model_name}_importances.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
