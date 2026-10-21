"""Deterministic runtime diagnostics and invariant verification for `utilitrack`."""

from __future__ import annotations

import time
from typing import Any

from .telemetry import compute_degree_days, calculate_scope2_emissions
from .scheduler import euclidean_distance_km

def _domain_probe(seed: int) -> dict:
    hdd, cdd = compute_degree_days(85.0, base_temp_f=65.0)
    report = calculate_scope2_emissions(
        total_kwh=1350000.0, gross_floor_area_sqft=100000.0,
        egrid_subregion="ERCT", building_type="commercial_office"
    )
    dist = euclidean_distance_km((0.0, 0.0), (3.0, 4.0))
    return {
        "hdd": hdd,
        "cdd": cdd,
        "scope2_metric_tons_co2e": round(report.scope2_metric_tons_co2e, 2),
        "distance_km": round(dist, 3),
        "invariants_ok": (hdd == 0.0 and cdd == 20.0 and abs(report.scope2_metric_tons_co2e - 518.4) < 0.1 and dist == 5.0),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `utilitrack-cbre` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "utilitrack-cbre",
        "package": "utilitrack",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `utilitrack`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "utilitrack-cbre",
        "package": "utilitrack",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
