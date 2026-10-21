"""UtiliTrack: Commercial Real-Estate Energy Telemetry & SLA Maintenance Dispatch.

1st Place CBRE Challenge & 3rd Place Overall @ HackUNT 2024.
Author: Abdullah Azmi (https://abdullahazmi.com)
"""

from utilitrack.scheduler import (
    DispatchAssignment,
    SchedulePlan,
    Technician,
    WorkOrder,
    compute_work_order_priority,
    dispatch_work_orders,
    euclidean_distance_km,
)
from utilitrack.telemetry import (
    ASHRAE_EUI_BENCHMARKS_KWH_SQFT,
    EGRID_EMISSION_FACTORS_KG_PER_KWH,
    CarbonFootprintReport,
    DriftAlert,
    EUIBaselineModel,
    MeterReading,
    calculate_scope2_emissions,
    compute_degree_days,
    detect_equipment_drift,
    fit_weather_normalized_eui,
)

__all__ = [
    "ASHRAE_EUI_BENCHMARKS_KWH_SQFT",
    "EGRID_EMISSION_FACTORS_KG_PER_KWH",
    "CarbonFootprintReport",
    "DispatchAssignment",
    "DriftAlert",
    "EUIBaselineModel",
    "MeterReading",
    "SchedulePlan",
    "Technician",
    "WorkOrder",
    "calculate_scope2_emissions",
    "compute_degree_days",
    "compute_work_order_priority",
    "detect_equipment_drift",
    "dispatch_work_orders",
    "euclidean_distance_km",
    "fit_weather_normalized_eui",
]

__version__ = "1.0.0"

from .diagnostics import run_diagnostic_suite, validate_runtime_invariants
__all__ = list(__all__) + ["run_diagnostic_suite", "validate_runtime_invariants"]
