"""Alert publishing for high-risk transactions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.paysim_loader import PROJECT_ROOT

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore


@dataclass
class FraudAlert:
    txn_id: str
    risk_score: float
    fraud_pattern: str
    timestamp: str

    def to_message(self) -> dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "risk_score": round(self.risk_score, 4),
            "fraud_pattern": self.fraud_pattern,
            "timestamp": self.timestamp,
        }


class AlertingService:
    """Publish fraud alerts to SNS and structured logs."""

    def __init__(
        self,
        *,
        topic_arn: str | None = None,
        region_name: str | None = None,
        log_path: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.topic_arn = topic_arn
        self.region_name = region_name
        self.enabled = enabled
        self.log_path = (
            Path(log_path)
            if log_path is not None
            else PROJECT_ROOT / "data" / "processed" / "alerts.jsonl"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = (
            boto3.client("sns", region_name=region_name)
            if enabled and topic_arn and boto3 is not None
            else None
        )
        self.published_alerts: list[FraudAlert] = []

    def publish(self, alert: FraudAlert) -> None:
        """Synchronously publish and log an alert."""

        if not self.enabled:
            return

        payload = alert.to_message()
        if self.client is not None and self.topic_arn is not None:
            self.client.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(payload),
                Subject="sentinel-upi-fraud-alert",
            )

        self._write_structured_log(payload)
        self.published_alerts.append(alert)

    async def publish_async(self, alert: FraudAlert) -> None:
        """Fire-and-forget async publish path for the API hot path."""

        if not self.enabled:
            return
        await asyncio.to_thread(self.publish, alert)

    def dispatch_background(self, alert: FraudAlert) -> asyncio.Task[Any] | None:
        """Schedule alert publishing without awaiting it in the request path."""

        if not self.enabled:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(self.publish_async(alert))

    @staticmethod
    def build_alert(
        *,
        txn_id: str,
        risk_score: float,
        fraud_pattern: str,
        timestamp: datetime | None = None,
    ) -> FraudAlert:
        resolved_timestamp = timestamp or datetime.now(timezone.utc)
        return FraudAlert(
            txn_id=txn_id,
            risk_score=risk_score,
            fraud_pattern=fraud_pattern,
            timestamp=resolved_timestamp.isoformat(),
        )

    def _write_structured_log(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
