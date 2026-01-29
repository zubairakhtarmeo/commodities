from __future__ import annotations

import argparse
from pathlib import Path

from forecasting import load_config
from forecasting.pipeline import ForecastingEngine
from forecasting.utils import parse_date


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    engine = ForecastingEngine(cfg)
    trained = engine.train()

    for asset_id, models in trained.items():
        print(f"[{asset_id}] trained: {', '.join(models)}")
        print(f"  artifacts -> {Path(cfg.output.artifacts_dir) / asset_id}")

    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    engine = ForecastingEngine(cfg)

    asof = parse_date(args.asof)
    df = engine.predict(asset_id=args.asset_id, model_name=args.model_name, asof=asof)
    print(df.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train + evaluate models")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Predict at a given as-of timestamp")
    p_pred.add_argument("--config", required=True, help="Path to YAML config")
    p_pred.add_argument("--asset-id", required=True)
    p_pred.add_argument("--model-name", required=True)
    p_pred.add_argument("--asof", required=True, help="ISO date, e.g. 2025-12-31")
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
