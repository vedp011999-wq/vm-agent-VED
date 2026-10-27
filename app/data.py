"""Data access: load the monitoring export, validate it, and turn raw samples into per-VM summaries.

Bad records are skipped and reported as issues instead of crashing the app.
"""
import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

SIZES = ["small", "medium", "large", "xlarge"]
MONTHLY_COST = {"small": 50, "medium": 100, "large": 200, "xlarge": 400}
Size = Literal["small", "medium", "large", "xlarge"]


class Sample(BaseModel):
    ts: datetime
    cpu: float = Field(ge=0, le=100)
    mem: float = Field(ge=0, le=100)


class VMRecord(BaseModel):  # one raw record from the source
    id: str = Field(pattern=r"^vm-\d{3}$")
    name: str
    environment: Literal["prod", "staging", "dev"]
    size: Size
    description: str = ""
    samples: list[Sample]


class VM(BaseModel):  # what the API, UI and agent see
    id: str
    name: str
    environment: str
    size: Size
    description: str
    monthly_cost: int
    cpu_avg: float | None
    cpu_peak: float | None
    mem_avg: float | None
    mem_peak: float | None
    last_sample: datetime | None
    is_stale: bool
    signal: Literal["underutilized", "overutilized", "normal", "unknown"]


class DataError(Exception):
    """The whole source is unreadable."""


def utilization_signal(cpu_peak, mem_peak, is_stale) -> str:
    if cpu_peak is None or is_stale:
        return "unknown"  # no fresh data -> no opinion
    if cpu_peak >= 85 or mem_peak >= 90:
        return "overutilized"
    if cpu_peak < 30 and mem_peak < 40:
        return "underutilized"
    return "normal"


def summarize(rec: VMRecord, now: datetime, max_age_hours: int) -> VM:
    cpu = [s.cpu for s in rec.samples]
    mem = [s.mem for s in rec.samples]
    last = max((s.ts for s in rec.samples), default=None)
    stale = last is not None and now - last > timedelta(hours=max_age_hours)
    cpu_peak = max(cpu) if cpu else None
    mem_peak = max(mem) if mem else None
    return VM(
        id=rec.id, name=rec.name, environment=rec.environment, size=rec.size, description=rec.description,
        monthly_cost=MONTHLY_COST[rec.size],
        cpu_avg=round(sum(cpu) / len(cpu), 1) if cpu else None, cpu_peak=cpu_peak,
        mem_avg=round(sum(mem) / len(mem), 1) if mem else None, mem_peak=mem_peak,
        last_sample=last, is_stale=stale, signal=utilization_signal(cpu_peak, mem_peak, stale),
    )


def load_vms(path: str, now: datetime, max_age_hours: int = 24) -> tuple[list[VM], list[str]]:
    """Returns (vms, issues). Issues are human-readable data-quality problems."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DataError(f"cannot read {path}: {e}") from e

    vms, issues, seen = [], [], set()
    for i, item in enumerate(raw.get("vms", [])):
        try:
            rec = VMRecord.model_validate(item)
        except ValidationError as e:
            err = e.errors()[0]
            issues.append(f"record {i}: invalid {'.'.join(map(str, err['loc']))} ({err['msg']})")
            continue
        if rec.id in seen:
            issues.append(f"{rec.id}: duplicate id, skipped")
            continue
        seen.add(rec.id)
        vm = summarize(rec, now, max_age_hours)
        if not rec.samples:
            issues.append(f"{rec.id}: no metric samples")
        elif vm.is_stale:
            issues.append(f"{rec.id}: stale data, last sample {vm.last_sample:%Y-%m-%d %H:%M}")
        vms.append(vm)
    return vms, issues
