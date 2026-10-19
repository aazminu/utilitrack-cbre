"""Commercial building energy telemetry, weather-normalized EUI baseline regression,
CUSUM/EWMA equipment degradation drift detection, and EPA eGRID Scope 2 carbon accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

# EPA eGRID subregion CO2e emission rates (kg CO2e per kWh)
EGRID_EMISSION_FACTORS_KG_PER_KWH: dict[str, float] = {
    "ERCT": 0.384,  # ERCOT All (Texas / DFW region)
    "CAMX": 0.226,  # WECC California
    "RFCW": 0.485,  # ReliabilityFirst West
    "SRMV": 0.362,  # SERC Mississippi Valley
    "NYUP": 0.115,  # NPCC Upstate NY
    "US_AVG": 0.386,  # US National Average
}

# ASHRAE Standard 100 / CBECS annual EUI targets (kWh / sq.ft / year)
ASHRAE_EUI_BENCHMARKS_KWH_SQFT: dict[str, float] = {
    "commercial_office": 14.8,
    "data_center": 145.0,
    "retail_mall": 18.2,
    "hospital_inpatient": 31.5,
    "warehouse_refrigerated": 24.0,
    "education_campus": 13.4,
}


@dataclass(frozen=True)
class MeterReading:
    """Interval smart-meter reading paired with outdoor ambient dry-bulb temperature."""

    timestamp: str
    kwh: float
    outdoor_temp_f: float
    occupancy_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.kwh < 0:
            raise ValueError("Meter kWh cannot be negative.")
        if not (0.0 <= self.occupancy_ratio <= 1.5):
            raise ValueError("occupancy_ratio must be in [0.0, 1.5].")


@dataclass(frozen=True)
class EUIBaselineModel:
    """Three-parameter change-point weather-normalized Energy Use Intensity (EUI) model.

    Models interval EUI (kWh/sqft) as:
        EUI = baseload_kwh_per_sqft
              + cooling_slope_kwh_per_cdd_sqft * CDD(T)
              + heating_slope_kwh_per_hdd_sqft * HDD(T)
    """

    baseload_kwh_per_sqft: float
    cooling_slope_kwh_per_cdd_sqft: float
    heating_slope_kwh_per_hdd_sqft: float
    r_squared: float
    residual_std_kwh_per_sqft: float
    base_temp_f: float = 65.0

    def predict_eui(self, outdoor_temp_f: float) -> float:
        """Predict interval EUI (kWh/sqft) for a given outdoor temperature (F)."""
        hdd, cdd = compute_degree_days(outdoor_temp_f, self.base_temp_f)
        pred = (
            self.baseload_kwh_per_sqft
            + self.cooling_slope_kwh_per_cdd_sqft * cdd
            + self.heating_slope_kwh_per_hdd_sqft * hdd
        )
        return max(0.0, pred)

    def predict_kwh(self, outdoor_temp_f: float, gross_floor_area_sqft: float) -> float:
        """Predict total interval energy consumption (kWh) for the building."""
        if gross_floor_area_sqft <= 0:
            raise ValueError("gross_floor_area_sqft must be positive.")
        return self.predict_eui(outdoor_temp_f) * gross_floor_area_sqft


@dataclass(frozen=True)
class DriftAlert:
    """Equipment power-draw anomaly detection state using tabular CUSUM and EWMA."""

    reading_index: int
    timestamp: str
    observed_kwh: float
    expected_kwh: float
    residual_z: float
    cusum_pos: float
    cusum_neg: float
    ewma_z: float
    is_anomalous: bool
    severity: str


@dataclass(frozen=True)
class CarbonFootprintReport:
    """Scope 2 greenhouse gas accounting and ASHRAE EUI benchmark comparison."""

    total_kwh: float
    egrid_subregion: str
    emission_factor_kg_co2e_per_kwh: float
    scope2_kg_co2e: float
    scope2_metric_tons_co2e: float
    building_eui_kwh_per_sqft: float
    ashrae_benchmark_delta_pct: float


def compute_degree_days(
    outdoor_temp_f: float, base_temp_f: float = 65.0
) -> tuple[float, float]:
    """Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD) relative to base_temp_f.

    Returns:
        Tuple of (hdd, cdd).
    """
    hdd = max(0.0, base_temp_f - outdoor_temp_f)
    cdd = max(0.0, outdoor_temp_f - base_temp_f)
    return hdd, cdd


def _solve_3x3(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve a 3x3 linear system A * x = b via Gaussian elimination with partial pivoting."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = 3
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            M[col][col] += 1e-8  # Ridge regularization for collinear degree days
        else:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def fit_weather_normalized_eui(
    readings: Sequence[MeterReading],
    gross_floor_area_sqft: float,
    base_temp_f: float = 65.0,
) -> EUIBaselineModel:
    """Fit a weather-normalized EUI baseline model via Ordinary Least Squares (OLS).

    Follows ASHRAE Guideline 14 three-parameter / five-parameter degree-day regression
    for commercial HVAC/chiller plants.
    """
    if len(readings) < 3:
        raise ValueError("At least 3 meter readings are required to fit EUI regression.")
    if gross_floor_area_sqft <= 0:
        raise ValueError("gross_floor_area_sqft must be strictly positive.")

    xs: list[tuple[float, float, float]] = []
    ys: list[float] = []
    for r in readings:
        hdd, cdd = compute_degree_days(r.outdoor_temp_f, base_temp_f)
        xs.append((1.0, cdd, hdd))
        ys.append(r.kwh / gross_floor_area_sqft)

    # Normal equations (X^T X + lambda * I) beta = X^T y
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for x_row, y_val in zip(xs, ys):
        for i in range(3):
            xty[i] += x_row[i] * y_val
            for j in range(3):
                xtx[i][j] += x_row[i] * x_row[j]

    # Tiny Tikhonov regularization on slopes to ensure stability if only cooling or heating is present
    xtx[1][1] += 1e-9
    xtx[2][2] += 1e-9

    beta0, beta_cdd, beta_hdd = _solve_3x3(xtx, xty)
    beta0 = max(0.0, beta0)
    beta_cdd = max(0.0, beta_cdd)
    beta_hdd = max(0.0, beta_hdd)

    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    residuals = [
        y - (beta0 + beta_cdd * x[1] + beta_hdd * x[2]) for x, y in zip(xs, ys)
    ]
    ss_res = sum(res**2 for res in residuals)

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    r_squared = max(0.0, min(1.0, r_squared))
    dof = max(1, len(ys) - 3)
    residual_std = math.sqrt(ss_res / dof)
    residual_std = max(residual_std, 1e-4)

    return EUIBaselineModel(
        baseload_kwh_per_sqft=beta0,
        cooling_slope_kwh_per_cdd_sqft=beta_cdd,
        heating_slope_kwh_per_hdd_sqft=beta_hdd,
        r_squared=r_squared,
        residual_std_kwh_per_sqft=residual_std,
        base_temp_f=base_temp_f,
    )


def detect_equipment_drift(
    readings: Sequence[MeterReading],
    baseline: EUIBaselineModel,
    gross_floor_area_sqft: float,
    cusum_slack_k: float = 0.5,
    cusum_threshold_h: float = 4.0,
    ewma_lambda: float = 0.25,
    ewma_threshold_z: float = 2.5,
) -> list[DriftAlert]:
    """Detect HVAC/chiller degradation and simultaneous heating/cooling via CUSUM + EWMA.

    Maintains two-sided tabular CUSUM accumulators (S_t^+, S_t^-) and an Exponentially
    Weighted Moving Average (EWMA) statistic over standardized EUI residuals.
    """
    if gross_floor_area_sqft <= 0:
        raise ValueError("gross_floor_area_sqft must be positive.")
    if not (0.0 < ewma_lambda <= 1.0):
        raise ValueError("ewma_lambda must be in (0, 1].")

    alerts: list[DriftAlert] = []
    s_pos = 0.0
    s_neg = 0.0
    ewma_val = 0.0
    sigma = max(baseline.residual_std_kwh_per_sqft, 1e-4)

    for idx, reading in enumerate(readings):
        t = idx + 1
        expected_kwh = baseline.predict_kwh(reading.outdoor_temp_f, gross_floor_area_sqft)
        observed_eui = reading.kwh / gross_floor_area_sqft
        expected_eui = expected_kwh / gross_floor_area_sqft
        z_t = (observed_eui - expected_eui) / sigma

        s_pos = max(0.0, s_pos + z_t - cusum_slack_k)
        s_neg = max(0.0, s_neg - z_t - cusum_slack_k)

        ewma_val = ewma_lambda * z_t + (1.0 - ewma_lambda) * ewma_val
        ewma_se = math.sqrt(
            (ewma_lambda / (2.0 - ewma_lambda)) * (1.0 - (1.0 - ewma_lambda) ** (2 * t))
        )
        ewma_z = ewma_val / max(ewma_se, 1e-6)

        cusum_breach = max(s_pos, s_neg) >= cusum_threshold_h
        ewma_breach = abs(ewma_z) >= ewma_threshold_z
        is_anomalous = cusum_breach or ewma_breach

        if s_pos >= cusum_threshold_h * 1.5 or abs(ewma_z) >= ewma_threshold_z * 1.4:
            severity = "CRITICAL"
        elif is_anomalous:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        alerts.append(
            DriftAlert(
                reading_index=idx,
                timestamp=reading.timestamp,
                observed_kwh=reading.kwh,
                expected_kwh=expected_kwh,
                residual_z=z_t,
                cusum_pos=s_pos,
                cusum_neg=s_neg,
                ewma_z=ewma_z,
                is_anomalous=is_anomalous,
                severity=severity,
            )
        )

    return alerts


def calculate_scope2_emissions(
    total_kwh: float,
    gross_floor_area_sqft: float,
    egrid_subregion: str = "ERCT",
    building_type: str = "commercial_office",
) -> CarbonFootprintReport:
    """Calculate EPA eGRID Scope 2 carbon emissions and ASHRAE Standard 100 benchmark delta."""
    if total_kwh < 0:
        raise ValueError("total_kwh must be non-negative.")
    if gross_floor_area_sqft <= 0:
        raise ValueError("gross_floor_area_sqft must be positive.")

    subregion_key = egrid_subregion.upper()
    if subregion_key not in EGRID_EMISSION_FACTORS_KG_PER_KWH:
        raise KeyError(f"Unsupported eGRID subregion: {egrid_subregion}")
    if building_type not in ASHRAE_EUI_BENCHMARKS_KWH_SQFT:
        raise KeyError(f"Unsupported building_type: {building_type}")

    factor = EGRID_EMISSION_FACTORS_KG_PER_KWH[subregion_key]
    kg_co2e = total_kwh * factor
    tons_co2e = kg_co2e / 1000.0

    building_eui = total_kwh / gross_floor_area_sqft
    benchmark_eui = ASHRAE_EUI_BENCHMARKS_KWH_SQFT[building_type]
    delta_pct = ((building_eui - benchmark_eui) / benchmark_eui) * 100.0

    return CarbonFootprintReport(
        total_kwh=total_kwh,
        egrid_subregion=subregion_key,
        emission_factor_kg_co2e_per_kwh=factor,
        scope2_kg_co2e=kg_co2e,
        scope2_metric_tons_co2e=tons_co2e,
        building_eui_kwh_per_sqft=building_eui,
        ashrae_benchmark_delta_pct=delta_pct,
    )
