"""Load PaySim transactions and adapt them into a UPI-shaped dataset."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "hparams.yaml"
EXPECTED_OUTPUT_COLUMNS = [
    "txn_id",
    "src_upi",
    "dst_upi",
    "amount_clipped",
    "timestamp",
    "merchant_type",
    "mcc_weight",
    "is_fraud",
]
USER_VPA_PROVIDERS = ("okaxis", "oksbi", "okhdfcbank", "okicici", "ybl")
MERCHANT_VPA_PROVIDERS = ("oksbi", "okhdfcbank", "paytm", "ibl")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration from YAML."""

    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with resolved_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_paysim_as_upi(
    csv_path: str | Path,
    config_path: str | Path | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read a PaySim CSV and return a cleaned UPI-like transaction frame."""

    read_kwargs: dict[str, Any] = {}
    if max_rows is not None:
        read_kwargs["nrows"] = int(max_rows)
    raw_frame = pd.read_csv(csv_path, **read_kwargs)
    return adapt_paysim_to_upi(raw_frame=raw_frame, config_path=config_path)


def adapt_paysim_to_upi(
    raw_frame: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Transform a PaySim-style dataframe into the Phase 1 UPI schema.

    Design choices:
    - `TRANSFER` is treated as the P2P flow and is the only transaction type
      expanded into multiple destination VPAs when it exceeds the NPCI cap.
    - Merchant categories are assigned per destination account so the same
      merchant node keeps a stable MCC risk baseline across rows.
    - Timestamps are rebuilt over a fixed 30-day window using deterministic
      cosine-shaped hour profiles to encode legitimate peak hours and fraud
      off-hour bursts.
    """

    config = load_config(config_path)
    prepared = _prepare_raw_frame(raw_frame)
    expanded = _expand_rows(prepared, config=config)
    expanded["is_fraud"] = _calibrate_fraud_labels(expanded, config=config)
    expanded["merchant_type"] = _assign_merchant_types(expanded, config=config)
    expanded["mcc_weight"] = expanded["merchant_type"].map(
        config["data"]["merchant_risk_weights"]
    )
    expanded["timestamp"] = _generate_timestamps(expanded, config=config)

    cleaned = expanded.loc[:, EXPECTED_OUTPUT_COLUMNS].copy()
    cleaned["amount_clipped"] = cleaned["amount_clipped"].round(2)
    cleaned["is_fraud"] = cleaned["is_fraud"].astype(int)
    cleaned = cleaned.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)

    validate_loader_output(cleaned, config=config)
    return cleaned


def validate_loader_output(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    """Validate the Phase 1 loader guarantees."""

    active_config = config or load_config()
    data_config = active_config["data"]
    fraud_rate = frame["is_fraud"].mean()
    min_rate = data_config["fraud_rate"]["min"]
    max_rate = data_config["fraud_rate"]["max"]
    cap = data_config["npci"]["p2p_cap_inr"]
    expected_span = pd.Timedelta(days=data_config["simulation_window_days"])
    actual_span = frame["timestamp"].max() - frame["timestamp"].min()

    if not min_rate <= fraud_rate <= max_rate:
        raise AssertionError(
            f"Fraud rate {fraud_rate:.6f} is outside [{min_rate:.6f}, {max_rate:.6f}]"
        )
    if frame["amount_clipped"].gt(cap).any():
        raise AssertionError(f"Found transaction amount above NPCI cap {cap}")
    if actual_span != expected_span:
        raise AssertionError(
            f"Timestamp span {actual_span} does not match {expected_span}"
        )


def _prepare_raw_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"type", "amount", "nameOrig", "nameDest"}
    missing = required.difference(raw_frame.columns)
    if missing:
        raise ValueError(f"Missing required PaySim columns: {sorted(missing)}")

    prepared = raw_frame.copy()
    if "step" not in prepared.columns:
        prepared["step"] = np.arange(len(prepared), dtype=int)
    if "isFraud" not in prepared.columns:
        prepared["isFraud"] = 0
    if "isFlaggedFraud" not in prepared.columns:
        prepared["isFlaggedFraud"] = 0

    prepared["type"] = prepared["type"].astype(str).str.upper()
    prepared["amount"] = pd.to_numeric(prepared["amount"], errors="coerce").fillna(0.0)
    prepared["step"] = pd.to_numeric(prepared["step"], errors="coerce").fillna(0).astype(int)
    prepared["nameOrig"] = prepared["nameOrig"].astype(str)
    prepared["nameDest"] = prepared["nameDest"].astype(str)
    prepared["isFraud"] = prepared["isFraud"].fillna(0).astype(int).clip(lower=0, upper=1)
    prepared["isFlaggedFraud"] = (
        prepared["isFlaggedFraud"].fillna(0).astype(int).clip(lower=0, upper=1)
    )
    return prepared.reset_index(drop=True)


def _expand_rows(raw_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cap = float(config["data"]["npci"]["p2p_cap_inr"])
    records: list[dict[str, Any]] = []

    for row_index, row in raw_frame.iterrows():
        txn_type = row["type"]
        original_amount = max(float(row["amount"]), 0.0)
        is_p2p = txn_type == "TRANSFER"
        split_count = math.ceil(original_amount / cap) if is_p2p and original_amount > cap else 1
        remaining = original_amount
        txn_base = f"TXN_{row_index:07d}"

        for split_index in range(split_count):
            if split_count > 1:
                amount_clipped = min(cap, remaining)
                remaining -= amount_clipped
                txn_id = f"{txn_base}_{split_index + 1:02d}"
            else:
                amount_clipped = min(original_amount, cap)
                txn_id = txn_base

            records.append(
                {
                    "txn_id": txn_id,
                    "src_upi": _make_upi_handle(row["nameOrig"], role="user"),
                    "dst_upi": _make_upi_handle(
                        row["nameDest"],
                        role="user" if is_p2p else "merchant",
                        alias_index=split_index if split_count > 1 else None,
                    ),
                    "amount_clipped": float(amount_clipped),
                    "txn_type": txn_type,
                    "nameDest": row["nameDest"],
                    "step": int(row["step"]),
                    "is_p2p": is_p2p,
                    "base_is_fraud": int(row["isFraud"]),
                    "flagged_fraud": int(row["isFlaggedFraud"]),
                    "original_amount": original_amount,
                    "split_count": split_count,
                }
            )

    return pd.DataFrame.from_records(records)


def _calibrate_fraud_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    fraud_config = config["data"]["fraud_rate"]
    min_rate = fraud_config["min"]
    max_rate = fraud_config["max"]
    target_rate = fraud_config["target"]
    labels = frame["base_is_fraud"].astype(int).copy()
    current_rate = labels.mean()

    if min_rate <= current_rate <= max_rate:
        return labels

    target_count = max(1, int(round(target_rate * len(frame))))
    scores = _fraud_risk_scores(frame, config=config)
    tie_breaker = frame["txn_id"].map(lambda value: _stable_fraction(value, "fraud"))

    if labels.sum() < target_count:
        candidates = frame.index[labels.eq(0)]
        ordered = sorted(
            candidates,
            key=lambda idx: (scores.loc[idx], tie_breaker.loc[idx]),
            reverse=True,
        )
        labels.loc[ordered[: target_count - int(labels.sum())]] = 1
    elif labels.sum() > target_count:
        candidates = frame.index[labels.eq(1)]
        ordered = sorted(
            candidates,
            key=lambda idx: (scores.loc[idx], tie_breaker.loc[idx]),
        )
        labels.loc[ordered[: int(labels.sum()) - target_count]] = 0

    return labels.astype(int)


def _fraud_risk_scores(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    cap = float(config["data"]["npci"]["p2p_cap_inr"])
    amount_component = np.log1p(frame["original_amount"].clip(lower=0.0)) / np.log1p(cap)
    transfer_component = frame["txn_type"].isin(["TRANSFER", "CASH_OUT"]).astype(float)
    split_component = frame["split_count"].gt(1).astype(float)

    return (
        0.50 * frame["base_is_fraud"].astype(float)
        + 0.18 * frame["flagged_fraud"].astype(float)
        + 0.14 * transfer_component
        + 0.10 * split_component
        + 0.08 * amount_component
    )


def _assign_merchant_types(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    categories = pd.Series("peer_to_peer", index=frame.index, dtype="object")
    merchant_rows = frame.loc[~frame["is_p2p"]]
    if merchant_rows.empty:
        return categories

    grouped = merchant_rows.groupby("nameDest", sort=False).agg(
        txn_type=("txn_type", lambda values: values.mode().iat[0]),
        fraud_exposure=("is_fraud", "max"),
    )
    merchant_types: dict[str, str] = {}
    for name_dest, values in grouped.iterrows():
        mix_key = _merchant_mix_key(
            txn_type=str(values["txn_type"]),
            fraud_exposure=int(values["fraud_exposure"]),
        )
        probabilities = config["data"]["merchant_category_mix"][mix_key]
        merchant_types[name_dest] = _select_from_distribution(
            distribution=probabilities,
            key_parts=(name_dest, mix_key),
        )

    categories.loc[merchant_rows.index] = merchant_rows["nameDest"].map(merchant_types)
    return categories


def _merchant_mix_key(txn_type: str, fraud_exposure: int) -> str:
    normalized = txn_type.lower().replace("-", "_")
    if normalized not in {"payment", "debit", "cash_out", "cash_in"}:
        normalized = "payment"
    suffix = "fraud" if fraud_exposure else "legit"
    return f"{normalized}_{suffix}"


def _select_from_distribution(
    distribution: dict[str, float],
    key_parts: tuple[str, str],
) -> str:
    threshold = _stable_fraction(*key_parts)
    cumulative = 0.0
    last_key = next(reversed(distribution))
    for category, probability in distribution.items():
        cumulative += float(probability)
        if threshold <= cumulative:
            return category
    return last_key


def _generate_timestamps(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    data_config = config["data"]
    start_timestamp = pd.Timestamp(data_config["start_timestamp"])
    total_days = int(data_config["simulation_window_days"])

    if len(frame) == 1:
        return pd.Series([start_timestamp], index=frame.index, dtype="datetime64[ns]")

    relative_steps = _relative_steps(frame)
    day_offsets = np.minimum(
        np.floor(relative_steps * total_days).astype(int),
        total_days - 1,
    )

    legit_cdf = _build_velocity_cdf(data_config["velocity_profile"]["legit"])
    fraud_cdf = _build_velocity_cdf(data_config["velocity_profile"]["fraud"])

    timestamps = []
    for index, row in frame.iterrows():
        cdf = fraud_cdf if int(row["is_fraud"]) else legit_cdf
        minute_index = _sample_from_cdf(cdf, key_parts=(row["txn_id"], "minute"))
        second = int(_stable_fraction(row["txn_id"], "second") * 60)
        timestamp = (
            start_timestamp
            + pd.Timedelta(days=int(day_offsets[index]))
            + pd.Timedelta(minutes=int(minute_index))
            + pd.Timedelta(seconds=second)
        )
        timestamps.append(timestamp)

    timestamp_series = pd.Series(timestamps, index=frame.index, dtype="datetime64[ns]")
    timestamp_series.loc[frame.index.min()] = start_timestamp
    timestamp_series.loc[frame.index.max()] = start_timestamp + pd.Timedelta(days=total_days)
    return timestamp_series


def _relative_steps(frame: pd.DataFrame) -> np.ndarray:
    steps = frame["step"].astype(float).to_numpy()
    if np.unique(steps).size == 1:
        return np.linspace(0.0, 1.0, num=len(frame), dtype=float)

    step_min = steps.min()
    step_span = steps.max() - step_min
    return (steps - step_min) / step_span


def _build_velocity_cdf(profile: dict[str, Any]) -> np.ndarray:
    minute_grid = np.arange(24 * 60, dtype=float) / 60.0
    weights = np.full(minute_grid.shape, float(profile["baseline"]), dtype=float)

    for peak in profile["peaks"]:
        weights += float(peak["amplitude"]) * _cosine_window(
            hour_values=minute_grid,
            center_hour=float(peak["center_hour"]),
            width_hours=float(peak["width_hours"]),
        )

    weights /= weights.sum()
    return np.cumsum(weights)


def _cosine_window(
    hour_values: np.ndarray,
    center_hour: float,
    width_hours: float,
) -> np.ndarray:
    wrapped_distance = np.abs(hour_values - center_hour)
    wrapped_distance = np.minimum(wrapped_distance, 24.0 - wrapped_distance)
    window = np.zeros_like(hour_values, dtype=float)
    inside = wrapped_distance <= width_hours
    window[inside] = 0.5 * (
        1.0 + np.cos(np.pi * wrapped_distance[inside] / width_hours)
    )
    return window


def _sample_from_cdf(cdf: np.ndarray, key_parts: tuple[str, str]) -> int:
    threshold = _stable_fraction(*key_parts)
    return int(np.searchsorted(cdf, threshold, side="right"))


def _make_upi_handle(
    identifier: str,
    role: str,
    alias_index: int | None = None,
) -> str:
    providers = USER_VPA_PROVIDERS if role == "user" else MERCHANT_VPA_PROVIDERS
    provider = providers[_stable_int(identifier, role, modulus=len(providers))]
    slug = _slugify(identifier)
    prefix = "u" if role == "user" else "m"
    if alias_index is not None:
        slug = f"{slug}s{alias_index + 1:02d}"
    return f"{prefix}{slug}@{provider}"


def _slugify(identifier: str) -> str:
    clean = "".join(character.lower() for character in str(identifier) if character.isalnum())
    if not clean:
        return "acct"
    return clean[-12:]


def _stable_int(*parts: Any, modulus: int) -> int:
    return int(_stable_fraction(*parts) * modulus) % modulus


def _stable_fraction(*parts: Any) -> float:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)
