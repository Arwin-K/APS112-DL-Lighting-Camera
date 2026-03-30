"""Carbon estimation helpers derived from electricity use."""

from __future__ import annotations


def estimate_power_from_lux(
    avg_lux: float,
    floor_area_m2: float,
    watts_per_lux_m2: float,
) -> float:
    """Converts average illuminance to an approximate lighting power estimate."""

    return avg_lux * floor_area_m2 * watts_per_lux_m2


def estimate_interval_energy_kwh(power_w: float, interval_hours: float) -> float:
    """Converts power in watts to energy over an interval."""

    return (power_w / 1000.0) * interval_hours


def estimate_interval_carbon_kg(
    energy_kwh: float,
    carbon_factor_kg_per_kwh: float,
) -> float:
    """Converts energy use into interval carbon emissions."""

    return energy_kwh * carbon_factor_kg_per_kwh


def summarize_carbon_from_lux(
    avg_lux: float,
    floor_area_m2: float,
    watts_per_lux_m2: float,
    interval_hours: float,
    carbon_factor_kg_per_kwh: float,
) -> dict[str, float]:
    """Builds a compact power, energy, and carbon report."""

    estimated_power_w = estimate_power_from_lux(
        avg_lux=avg_lux,
        floor_area_m2=floor_area_m2,
        watts_per_lux_m2=watts_per_lux_m2,
    )
    interval_energy_kwh = estimate_interval_energy_kwh(
        power_w=estimated_power_w,
        interval_hours=interval_hours,
    )
    interval_carbon_kg = estimate_interval_carbon_kg(
        energy_kwh=interval_energy_kwh,
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
    )
    return {
        "estimated_power_w": estimated_power_w,
        "interval_energy_kwh": interval_energy_kwh,
        "interval_carbon_kg": interval_carbon_kg,
    }
