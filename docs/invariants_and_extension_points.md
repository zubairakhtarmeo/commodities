# Invariants and extension points

This framework is designed to be **asset-agnostic** and **leakage-safe**.

The fastest way to break those guarantees is to “just tweak code for one asset”.
Instead, *all asset-specific behavior must live in config* (or in pluggable extension implementations that remain generic).

## What "invariant" means
An **invariant core module** encodes time semantics, leakage guards, train/infer parity, or evaluation correctness.

- Do **not** fork or edit these modules per asset.
- If an asset needs different behavior, expose it as **config** or as a **new plugin** that implements an existing interface.

## Invariant core modules (must never change per asset)

### Pipeline / orchestration
- `src/forecasting/pipeline/engine.py`
  - Owns the canonical end-to-end flow (ingest → align → features → dataset → split → train/eval → artifacts, and predict with the same features).
  - Asset-specific changes here typically introduce train/serve skew or as-of leakage.

### Time semantics
- `src/forecasting/time/alignment.py`
  - Defines canonical resampling and the **as-of availability** rule (including `availability_lag_steps`).
  - Asset-specific backfilling, custom resampling, or “use next available” logic is a common leakage vector.

### Dataset semantics
- `src/forecasting/dataset/builder.py`
  - Defines the meaning of a row timestamp $t$ and labels at $t+h$.
  - Asset-specific label shifting or target construction makes horizon metrics incomparable and risks lookahead.

### Validation
- `src/forecasting/validation/walk_forward.py`
  - Encodes purge/embargo logic preventing overlapping label horizons.
  - Per-asset relaxation here creates optimistic validation and invalid comparisons.

### Training
- `src/forecasting/training/trainer.py`
  - Defines fold-level fit/predict and keeps preprocessing inside estimators/pipelines.
  - Per-asset changes often lead to fitting scalers/imputers on full data (leakage).

### Feature system (generic packs)
- `src/forecasting/features/base.py`
- `src/forecasting/features/builder.py`
- `src/forecasting/features/packs.py`
  - Must remain generic “roles not heuristics”.
  - Asset-specific conditions here become hardcoded alpha and are not portable.

### Models (ladder + preprocessing)
- `src/forecasting/models/factory.py`
- `src/forecasting/models/naive.py`
  - Must remain generic and comparable across assets.
  - Per-asset preprocessing outside pipelines is a common source of train/test contamination.

### Ingestion contracts
- `src/forecasting/connectors/base.py`
- `src/forecasting/connectors/csv_connector.py`
- `src/forecasting/data/registry.py`
  - Define the contract for timestamps, values, and role mapping.
  - Per-asset cleaning rules here silently change “what was knowable when”.

### Evaluation + explainability
- `src/forecasting/evaluation/metrics.py`
- `src/forecasting/explainability/permutation.py`
  - Must not depend on future labels or test data.
  - Per-asset metrics/explanations lead to cherry-picked comparisons.

### Artifacts
- `src/forecasting/artifacts/store.py`
  - Must remain stable for automation (promotion, inference, monitoring).

### Monitoring (optional but should stay generic)
- `src/forecasting/monitoring/quality.py`
- `src/forecasting/monitoring/drift.py`
  - Must not require future labels to run in production.

## Config-driven behavior (safe per-asset changes)
All asset-level customization should happen through YAML config:

- `configs/*.yml`
  - Assets, connectors, roles
  - Frequency
  - Feature packs on/off + parameters
  - Dataset horizons
  - Walk-forward settings (initial train, test window, purge/embargo)
  - Model ladder enablement + parameters
  - Explainability enablement

## Explicit extension points (allowed vs forbidden)

### Allowed (safe and intended)
- **Add a new connector** by implementing `BaseConnector` and referencing it via config.
  - Example: database connector, API connector, Parquet connector.
- **Add a new feature pack** by implementing `FeaturePack` and registering it in the feature builder.
  - Must use only information available at or before the row timestamp $t$.
- **Add a new model type** inside the model factory.
  - Preprocessing must live inside the estimator/pipeline so it is fit per-fold on training data.
- **Add new metrics** that are horizon-correct and computed strictly on test folds.
- **Add monitoring checks** that rely only on inputs/features available at inference time.

### Forbidden (leakage-prone)
- Editing any invariant module “just for one asset”.
- Using forward-fill/backfill that makes future values appear earlier than they were known.
- Computing features with centered windows, negative shifts (except label creation), or any join that looks ahead.
- Fitting scalers/imputers/encoders on full data before splitting.
- Validating with random splits for time series.
- Relaxing purge/embargo to increase fold sizes without re-auditing label overlap.

## Practical checklist for onboarding a real asset (config only)
- Define target + exogenous as roles, set `availability_lag_steps` if the feed is delayed.
- Choose frequency and confirm timestamps represent that period correctly.
- Enable only feature packs that are compatible with available series.
- Ensure `purge_steps` is conservative relative to max lag/window depth.
- Confirm walk-forward produces folds and that train label horizons never overlap test.
