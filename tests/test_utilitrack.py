"""Unit and integration test suite for UtiliTrack (CBRE Challenge @ HackUNT 2024)."""

from __future__ import annotations

import pytest

from utilitrack.scheduler import (
    Technician,
    WorkOrder,
    compute_work_order_priority,
    dispatch_work_orders,
    euclidean_distance_km,
)
from utilitrack.telemetry import (
    MeterReading,
    calculate_scope2_emissions,
    compute_degree_days,
    detect_equipment_drift,
    fit_weather_normalized_eui,
)


def _generate_synthetic_bms_readings() -> list[MeterReading]:
    """Generate synthetic commercial building interval readings with known degree-day slopes."""
    temps = [40.0, 48.0, 55.0, 62.0, 65.0, 70.0, 76.0, 82.0, 88.0, 95.0, 100.0]
    gfa = 100_000.0
    baseload_eui = 0.020  # 2,000 kWh baseload
    cdd_slope = 0.0015  # 150 kWh per CDD
    hdd_slope = 0.0008  # 80 kWh per HDD
    readings: list[MeterReading] = []
    for i, temp in enumerate(temps):
        hdd, cdd = compute_degree_days(temp, 65.0)
        eui = baseload_eui + cdd_slope * cdd + hdd_slope * hdd
        readings.append(
            MeterReading(
                timestamp=f"2024-07-{i+1:02d}T14:00:00Z",
                kwh=eui * gfa,
                outdoor_temp_f=temp,
                occupancy_ratio=0.95,
            )
        )
    return readings


def test_compute_degree_days_cooling_heating_and_neutral() -> None:
    hdd_cool, cdd_cool = compute_degree_days(85.0, base_temp_f=65.0)
    assert hdd_cool == 0.0
    assert cdd_cool == pytest.approx(20.0)

    hdd_heat, cdd_heat = compute_degree_days(42.0, base_temp_f=65.0)
    assert hdd_heat == pytest.approx(23.0)
    assert cdd_heat == 0.0

    hdd_neutral, cdd_neutral = compute_degree_days(65.0, base_temp_f=65.0)
    assert hdd_neutral == 0.0
    assert cdd_neutral == 0.0


def test_meter_reading_validation() -> None:
    with pytest.raises(ValueError, match="negative"):
        MeterReading(timestamp="2024-01-01T00:00:00Z", kwh=-5.0, outdoor_temp_f=70.0)
    with pytest.raises(ValueError, match="occupancy_ratio"):
        MeterReading(
            timestamp="2024-01-01T00:00:00Z",
            kwh=100.0,
            outdoor_temp_f=70.0,
            occupancy_ratio=2.0,
        )


def test_fit_weather_normalized_eui_recovers_exact_parameters() -> None:
    readings = _generate_synthetic_bms_readings()
    model = fit_weather_normalized_eui(readings, gross_floor_area_sqft=100_000.0)

    assert model.baseload_kwh_per_sqft == pytest.approx(0.020, abs=1e-4)
    assert model.cooling_slope_kwh_per_cdd_sqft == pytest.approx(0.0015, abs=1e-4)
    assert model.heating_slope_kwh_per_hdd_sqft == pytest.approx(0.0008, abs=1e-4)
    assert model.r_squared == pytest.approx(1.0, abs=1e-5)


def test_eui_model_predict_kwh_and_validation() -> None:
    readings = _generate_synthetic_bms_readings()
    model = fit_weather_normalized_eui(readings, gross_floor_area_sqft=100_000.0)

    # At 85F (20 CDD), expected kWh = (0.020 + 0.0015 * 20) * 100,000 = 5,000 kWh
    pred_kwh = model.predict_kwh(85.0, 100_000.0)
    assert pred_kwh == pytest.approx(5000.0, rel=1e-3)

    with pytest.raises(ValueError, match="positive"):
        model.predict_kwh(85.0, 0.0)


def test_fit_weather_normalized_eui_requires_minimum_readings() -> None:
    readings = _generate_synthetic_bms_readings()[:2]
    with pytest.raises(ValueError, match="At least 3"):
        fit_weather_normalized_eui(readings, gross_floor_area_sqft=50_000.0)


def test_detect_equipment_drift_nominal_vs_chiller_degradation() -> None:
    baseline_readings = _generate_synthetic_bms_readings()
    model = fit_weather_normalized_eui(baseline_readings, gross_floor_area_sqft=100_000.0)

    # Nominal operation: zero drift alerts
    nominal_alerts = detect_equipment_drift(
        baseline_readings, model, gross_floor_area_sqft=100_000.0
    )
    assert all(not a.is_anomalous for a in nominal_alerts)
    assert all(a.severity == "NORMAL" for a in nominal_alerts)

    # Inject economizer damper stuck / chiller fouling (+35% kWh spike over 4 intervals)
    degraded_readings = list(baseline_readings) + [
        MeterReading("2024-07-12T14:00:00Z", kwh=6800.0, outdoor_temp_f=85.0),
        MeterReading("2024-07-13T14:00:00Z", kwh=7100.0, outdoor_temp_f=85.0),
        MeterReading("2024-07-14T14:00:00Z", kwh=7400.0, outdoor_temp_f=85.0),
    ]
    drift_alerts = detect_equipment_drift(
        degraded_readings, model, gross_floor_area_sqft=100_000.0
    )
    tail_alerts = drift_alerts[-3:]
    assert any(a.is_anomalous for a in tail_alerts)
    assert tail_alerts[-1].severity == "CRITICAL"
    assert tail_alerts[-1].cusum_pos > 4.0


def test_calculate_scope2_emissions_ercot_and_benchmark_delta() -> None:
    # 1,350,000 kWh over a 100,000 sqft office in ERCOT (13.5 kWh/sqft vs 14.8 benchmark)
    report = calculate_scope2_emissions(
        total_kwh=1_350_000.0,
        gross_floor_area_sqft=100_000.0,
        egrid_subregion="ERCT",
        building_type="commercial_office",
    )
    assert report.egrid_subregion == "ERCT"
    assert report.scope2_kg_co2e == pytest.approx(1_350_000.0 * 0.384)
    assert report.scope2_metric_tons_co2e == pytest.approx(518.4)
    assert report.building_eui_kwh_per_sqft == pytest.approx(13.5)
    assert report.ashrae_benchmark_delta_pct < 0.0  # 8.78% more efficient than benchmark


def test_calculate_scope2_emissions_invalid_inputs() -> None:
    with pytest.raises(KeyError, match="Unsupported eGRID"):
        calculate_scope2_emissions(10_000.0, 5_000.0, egrid_subregion="INVALID")
    with pytest.raises(KeyError, match="Unsupported building_type"):
        calculate_scope2_emissions(10_000.0, 5_000.0, building_type="castle")


def test_work_order_priority_ordering() -> None:
    critical_chiller = WorkOrder(
        order_id="WO-101",
        asset_id="CHILLER-01",
        required_skill="hvac_chiller",
        severity="CRITICAL",
        sla_hours_remaining=1.0,
        location_xy=(2.0, 3.0),
        estimated_duration_hours=2.0,
        est_energy_waste_kw=45.0,
    )
    routine_lighting = WorkOrder(
        order_id="WO-102",
        asset_id="LIGHT-PANEL-B",
        required_skill="electrical",
        severity="LOW",
        sla_hours_remaining=24.0,
        location_xy=(1.0, 1.0),
        estimated_duration_hours=1.0,
        est_energy_waste_kw=0.5,
    )
    assert compute_work_order_priority(critical_chiller) > compute_work_order_priority(
        routine_lighting
    )


def test_dispatch_work_orders_skill_and_proximity_matching() -> None:
    orders = [
        WorkOrder(
            order_id="WO-CRIT-1",
            asset_id="AHU-4",
            required_skill="hvac",
            severity="CRITICAL",
            sla_hours_remaining=2.0,
            location_xy=(10.0, 10.0),
            estimated_duration_hours=2.0,
            est_energy_waste_kw=32.0,
        ),
        WorkOrder(
            order_id="WO-ELEC-2",
            asset_id="VFD-2",
            required_skill="electrical",
            severity="HIGH",
            sla_hours_remaining=4.0,
            location_xy=(2.0, 2.0),
            estimated_duration_hours=1.5,
            est_energy_waste_kw=12.0,
        ),
    ]
    techs = [
        Technician(
            tech_id="T-NEAR-HVAC",
            name="Marcus Vance",
            skills=frozenset({"hvac", "controls"}),
            current_xy=(9.0, 10.0),
            available_hours=8.0,
        ),
        Technician(
            tech_id="T-FAR-HVAC",
            name="Elena Rostova",
            skills=frozenset({"hvac"}),
            current_xy=(50.0, 50.0),
            available_hours=8.0,
        ),
        Technician(
            tech_id="T-ELEC",
            name="Devon Brooks",
            skills=frozenset({"electrical"}),
            current_xy=(1.0, 2.0),
            available_hours=8.0,
        ),
    ]

    plan = dispatch_work_orders(orders, techs, avg_speed_kmh=40.0)
    assert len(plan.assignments) == 2
    assert plan.unassigned_order_ids == []
    assert plan.total_energy_waste_mitigated_kw == pytest.approx(44.0)
    assert plan.sla_compliance_rate == pytest.approx(1.0)

    by_order = {a.order_id: a for a in plan.assignments}
    assert by_order["WO-CRIT-1"].tech_id == "T-NEAR-HVAC"
    assert by_order["WO-ELEC-2"].tech_id == "T-ELEC"


def test_dispatch_work_orders_handles_unassigned_when_shift_capacity_exceeded() -> None:
    orders = [
        WorkOrder(
            order_id="WO-LONG",
            asset_id="BOILER-1",
            required_skill="boiler",
            severity="HIGH",
            sla_hours_remaining=3.0,
            location_xy=(5.0, 5.0),
            estimated_duration_hours=6.0,
            est_energy_waste_kw=20.0,
        )
    ]
    techs = [
        Technician(
            tech_id="T-SHORT-SHIFT",
            name="Samir Patel",
            skills=frozenset({"boiler"}),
            current_xy=(5.0, 5.0),
            available_hours=3.5,  # Less than 6.0h job requirement
        )
    ]
    plan = dispatch_work_orders(orders, techs)
    assert len(plan.assignments) == 0
    assert plan.unassigned_order_ids == ["WO-LONG"]
    assert plan.sla_compliance_rate == 0.0


def test_dispatch_work_orders_updates_technician_location_across_multi_stop_route() -> None:
    orders = [
        WorkOrder(
            order_id="WO-STOP-1",
            asset_id="CHILLER-A",
            required_skill="hvac",
            severity="CRITICAL",
            sla_hours_remaining=1.5,
            location_xy=(10.0, 0.0),
            estimated_duration_hours=1.0,
            est_energy_waste_kw=25.0,
        ),
        WorkOrder(
            order_id="WO-STOP-2",
            asset_id="CHILLER-B",
            required_skill="hvac",
            severity="MEDIUM",
            sla_hours_remaining=6.0,
            location_xy=(13.0, 4.0),  # 5 km from STOP-1 (3-4-5 triangle)
            estimated_duration_hours=1.5,
            est_energy_waste_kw=10.0,
        ),
    ]
    techs = [
        Technician(
            tech_id="T-SOLO",
            name="Alex Rivera",
            skills=frozenset({"hvac"}),
            current_xy=(0.0, 0.0),
            available_hours=8.0,
        )
    ]
    plan = dispatch_work_orders(orders, techs, avg_speed_kmh=50.0)
    assert len(plan.assignments) == 2
    assert plan.assignments[0].travel_distance_km == pytest.approx(10.0)
    assert plan.assignments[1].travel_distance_km == pytest.approx(5.0)


def test_euclidean_distance_km() -> None:
    assert euclidean_distance_km((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_empty_work_order_dispatch() -> None:
    plan = dispatch_work_orders([], [])
    assert plan.assignments == []
    assert plan.sla_compliance_rate == 1.0
