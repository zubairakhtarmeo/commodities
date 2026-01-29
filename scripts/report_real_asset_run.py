from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecasting.config import load_config
from forecasting.data import build_series_registry
from forecasting.dataset import build_supervised_dataset
from forecasting.features import FeatureBuilder
from forecasting.models import build_model
from forecasting.time import align_asset_frame
from forecasting.validation import WalkForwardSplitter


@dataclass(frozen=True)
class PredRow:
    asof: pd.Timestamp
    target_time: pd.Timestamp
    y_true: float
    y_pred: float


def _build_dataset(cfg_path: Path, asset_id: str):
    cfg = load_config(cfg_path)
    asset = next(a for a in cfg.assets if a.asset_id == asset_id)

    registry = build_series_registry(asset)
    base = align_asset_frame(registry, rule=cfg.frequency.rule)

    fb = FeatureBuilder.from_configs(cfg.features.packs)
    X = fb.build(base)

    ds = build_supervised_dataset(features=X, target=base["target"], cfg=cfg.dataset)
    return cfg, base, ds


def _collect_walk_forward_predictions(*, cfg, ds, base_times: pd.DatetimeIndex, model_name: str, horizon_steps: int):
    spec = next(m for m in cfg.model_ladder if m.name == model_name)
    built = build_model(spec, horizon_count=len(ds.horizons))

    splitter = WalkForwardSplitter(cfg.validation, max_horizon=max(ds.horizons))

    # Map requested horizon to column index
    horizons = list(ds.horizons)
    if horizon_steps not in horizons:
        raise ValueError(f"horizon_steps={horizon_steps} not in config horizons {horizons}")
    j = horizons.index(horizon_steps)
    ycol = f"y_h{horizon_steps}"

    rows: list[PredRow] = []
    skipped = 0
    for split in splitter.split(ds.times):
        X_train = ds.X.iloc[split.train_idx]
        y_train = ds.y.iloc[split.train_idx]
        X_test = ds.X.iloc[split.test_idx]
        y_test = ds.y.iloc[split.test_idx]

        est = built.estimator
        est.fit(X_train, y_train)
        pred = est.predict(X_test)

        # Align predictions/targets by asof timestamps (row timestamps)
        y_pred = pd.Series(pred[:, j], index=y_test.index, name="y_pred")
        y_true = y_test[ycol].astype(float)

        # Convert (asof t) -> (target_time t+h) using the full canonical grid
        # (ds.times may be filtered and shorter than the required horizon lookup).
        for asof, yt, yp in zip(y_true.index, y_true.to_numpy(), y_pred.to_numpy()):
            try:
                pos = base_times.get_loc(asof)
                tgt_time = base_times[pos + horizon_steps]
            except (KeyError, IndexError):
                skipped += 1
                continue
            rows.append(PredRow(asof=asof, target_time=pd.Timestamp(tgt_time), y_true=float(yt), y_pred=float(yp)))

    if skipped:
        print(f"\n[report] Skipped {skipped} rows that could not be mapped to target_time")

    return pd.DataFrame([r.__dict__ for r in rows]).sort_values("target_time")


def main() -> int:
    ap = argparse.ArgumentParser(description="Report metrics/importances and plot forecast-vs-actual for a trained real asset run.")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--model-name", default="linear_ridge")
    ap.add_argument("--horizon", type=int, default=1, help="Horizon steps to plot (must be in dataset.horizon_steps)")
    ap.add_argument("--out", default=None, help="Optional output PNG path. Default: artifacts/<asset>/forecast_vs_actual_<model>_h<h>.png")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg, base, ds = _build_dataset(cfg_path, args.asset_id)

    # Load latest saved artifacts (metrics + importances) if present
    art_dir = Path(cfg.output.artifacts_dir) / args.asset_id
    metrics_path = art_dir / f"{args.model_name}_metrics.csv"
    importances_path = art_dir / f"{args.model_name}_importances.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        print("\n=== Metrics (saved) ===")
        print(metrics.tail(5).to_string(index=False))
        # quick aggregate
        numeric_cols = [c for c in metrics.columns if "/" in c]
        if numeric_cols:
            means = metrics[numeric_cols].mean(numeric_only=True).sort_values()
            print("\n=== Mean metrics across folds ===")
            print(means.to_string())
    else:
        print(f"No metrics found at {metrics_path} (run training first).")

    if importances_path.exists():
        imp = pd.read_csv(importances_path)
        print("\n=== Feature importance (saved; top 20) ===")
        sort_col = "importance_mean" if "importance_mean" in imp.columns else imp.columns[0]
        show = imp.sort_values(sort_col, ascending=False).head(20)
        print(show.to_string(index=False))
    else:
        print(f"No importances found at {importances_path} (ensure explainability enabled).")

    # Build and plot predictions vs actual (no core code changes)
    preds = _collect_walk_forward_predictions(
        cfg=cfg, ds=ds, base_times=base.index, model_name=args.model_name, horizon_steps=args.horizon
    )

    out_path = Path(args.out) if args.out else (art_dir / f"forecast_vs_actual_{args.model_name}_h{args.horizon}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(preds["target_time"], preds["y_true"], label="actual", linewidth=1.5)
        plt.plot(preds["target_time"], preds["y_pred"], label="forecast", linewidth=1.5)
        plt.title(f"{args.asset_id} | {args.model_name} | horizon={args.horizon}")
        plt.xlabel("target_time")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"\nWrote plot: {out_path}")
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; install it to generate the plot:")
        print("  python -m pip install matplotlib")

    preds_out = art_dir / f"predictions_{args.model_name}_h{args.horizon}.csv"
    preds.to_csv(preds_out, index=False)
    print(f"Wrote predictions: {preds_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
