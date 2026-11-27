import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

from simulation_engine import SimulationEngine, SimulationResult, SimulationOutcome
from aeb_agent import AEBAgent


@dataclass
class ScenarioConfig:
    scenario_id: str
    description: str
    initial_speed_kmh: float
    mass_kg: float
    initial_distance_m: float
    relative_speed_kmh: float
    friction: float
    road_environment: str
    weather_condition: str


@dataclass
class ComparisonRecord:
    scenario_id: str
    description: str
    road_env: str
    friction: float
    weather: str
    speed_kmh: float
    initial_gap_m: float
    # AEB Disabled results
    coll_disabled: bool
    impact_speed_disabled_kmh: float
    final_gap_disabled_m: float
    # AEB Enabled results
    coll_enabled: bool
    impact_speed_enabled_kmh: float
    final_gap_enabled_m: float
    # Comparison delta
    collision_prevented: bool
    impact_speed_reduction_kmh: float
    # Relative collision impact speeds (closing velocity at impact)
    impact_rel_speed_disabled_kmh: float = 0.0
    impact_rel_speed_enabled_kmh: float = 0.0
    rel_speed_reduction_kmh: float = 0.0


def generate_benchmark_scenarios() -> List[ScenarioConfig]:
    """Generates a standardized suite of 20 diverse collision scenarios."""
    scenarios = [
        # 1-6: Urban Scenarios (30 - 50 km/h)
        ScenarioConfig("URBAN_01", "City dry road moderate gap", 40.0, 1400.0, 25.0, 40.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("URBAN_02", "City dry road short gap", 40.0, 1400.0, 15.0, 40.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("URBAN_03", "City wet road short gap", 45.0, 1500.0, 18.0, 45.0, 0.55, "Wet", "Rain"),
        ScenarioConfig("URBAN_04", "City icy road moderate gap", 30.0, 1300.0, 20.0, 30.0, 0.20, "Icy", "Clear"),
        ScenarioConfig("URBAN_05", "City wet road sudden obstacle", 50.0, 1450.0, 22.0, 50.0, 0.50, "Wet", "Fog"),
        ScenarioConfig("URBAN_06", "City dry road close obstacle", 35.0, 1200.0, 12.0, 35.0, 0.80, "Dry", "Clear"),

        # 7-12: Suburban / Arterial Roads (60 - 75 km/h)
        ScenarioConfig("SUBURBAN_01", "Arterial dry road long gap", 60.0, 1500.0, 45.0, 60.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("SUBURBAN_02", "Arterial dry road medium gap", 60.0, 1500.0, 28.0, 60.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("SUBURBAN_03", "Arterial wet road medium gap", 60.0, 1600.0, 32.0, 60.0, 0.55, "Wet", "Rain"),
        ScenarioConfig("SUBURBAN_04", "Arterial icy road long gap", 60.0, 1500.0, 40.0, 60.0, 0.22, "Icy", "Fog"),
        ScenarioConfig("SUBURBAN_05", "Arterial dry road heavy vehicle", 70.0, 1900.0, 40.0, 70.0, 0.80, "Dry", "Clear"),
        ScenarioConfig("SUBURBAN_06", "Arterial wet road heavy vehicle", 70.0, 1900.0, 35.0, 70.0, 0.50, "Wet", "Rain"),

        # 13-16: Highway Scenarios (80 - 100 km/h)
        ScenarioConfig("HIGHWAY_01", "Highway dry road high speed", 90.0, 1500.0, 55.0, 90.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("HIGHWAY_02", "Highway dry road sudden stationary", 90.0, 1500.0, 35.0, 90.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("HIGHWAY_03", "Highway wet road high speed", 85.0, 1650.0, 48.0, 85.0, 0.55, "Wet", "Rain"),
        ScenarioConfig("HIGHWAY_04", "Highway icy road emergency", 80.0, 1500.0, 30.0, 80.0, 0.18, "Icy", "Rain"),

        # 17-20: Dynamic Moving Lead Vehicle Scenarios
        ScenarioConfig("MOVING_01", "Highway car-following mild braking", 80.0, 1400.0, 35.0, 30.0, 0.85, "Dry", "Clear"),
        ScenarioConfig("MOVING_02", "Highway car-following sudden lead brake", 90.0, 1500.0, 25.0, 50.0, 0.80, "Dry", "Clear"),
        ScenarioConfig("MOVING_03", "Wet road car-following moderate gap", 70.0, 1500.0, 25.0, 35.0, 0.55, "Wet", "Rain"),
        ScenarioConfig("MOVING_04", "Safe following distant vehicle", 60.0, 1400.0, 50.0, 10.0, 0.85, "Dry", "Clear")
    ]
    return scenarios


class AEBComparator:
    """Runs paired comparative evaluations between AEB-enabled and AEB-disabled simulations."""

    def __init__(self, time_step: float = 0.01, max_duration: float = 8.0):
        # Instantiate a single shared agent to avoid reloading the pickle for every run
        self.agent = AEBAgent()
        self.engine = SimulationEngine(time_step=time_step, max_duration=max_duration, agent=self.agent)

    def evaluate_scenario(self, config: ScenarioConfig) -> ComparisonRecord:
        """Runs the scenario under both AEB disabled and enabled states."""
        # 1. Run without AEB (baseline)
        res_disabled = self.engine.run_scenario(
            initial_speed_kmh=config.initial_speed_kmh,
            mass_kg=config.mass_kg,
            initial_distance_m=config.initial_distance_m,
            relative_speed_kmh=config.relative_speed_kmh,
            friction=config.friction,
            road_environment=config.road_environment,
            weather_condition=config.weather_condition,
            aeb_enabled=False
        )

        # 2. Run with AEB
        res_enabled = self.engine.run_scenario(
            initial_speed_kmh=config.initial_speed_kmh,
            mass_kg=config.mass_kg,
            initial_distance_m=config.initial_distance_m,
            relative_speed_kmh=config.relative_speed_kmh,
            friction=config.friction,
            road_environment=config.road_environment,
            weather_condition=config.weather_condition,
            aeb_enabled=True
        )

        coll_prev = (res_disabled.collision_occurred and not res_enabled.collision_occurred)
        speed_red = max(0.0, res_disabled.impact_speed_kmh - res_enabled.impact_speed_kmh)
        rel_speed_red = max(0.0, res_disabled.impact_relative_speed_kmh - res_enabled.impact_relative_speed_kmh)

        return ComparisonRecord(
            scenario_id=config.scenario_id,
            description=config.description,
            road_env=config.road_environment,
            friction=config.friction,
            weather=config.weather_condition,
            speed_kmh=config.initial_speed_kmh,
            initial_gap_m=config.initial_distance_m,
            coll_disabled=res_disabled.collision_occurred,
            impact_speed_disabled_kmh=round(res_disabled.impact_speed_kmh, 2),
            final_gap_disabled_m=round(res_disabled.final_gap_m, 2),
            coll_enabled=res_enabled.collision_occurred,
            impact_speed_enabled_kmh=round(res_enabled.impact_speed_kmh, 2),
            final_gap_enabled_m=round(res_enabled.final_gap_m, 2),
            collision_prevented=coll_prev,
            impact_speed_reduction_kmh=round(speed_red, 2),
            impact_rel_speed_disabled_kmh=round(res_disabled.impact_relative_speed_kmh, 2),
            impact_rel_speed_enabled_kmh=round(res_enabled.impact_relative_speed_kmh, 2),
            rel_speed_reduction_kmh=round(rel_speed_red, 2)
        )

    def run_benchmark(self, scenarios: Optional[List[ScenarioConfig]] = None) -> Dict[str, Any]:
        """Runs all scenarios and calculates quantitative summary statistics."""
        if scenarios is None:
            scenarios = generate_benchmark_scenarios()

        records: List[ComparisonRecord] = []
        for sc in scenarios:
            records.append(self.evaluate_scenario(sc))

        total = len(records)
        collisions_disabled = sum(1 for r in records if r.coll_disabled)
        collisions_enabled = sum(1 for r in records if r.coll_enabled)
        prevented = sum(1 for r in records if r.collision_prevented)

        reduction_pct = (
            ((collisions_disabled - collisions_enabled) / collisions_disabled * 100.0)
            if collisions_disabled > 0 else 0.0
        )

        # Ground impact speed metrics across all scenarios
        avg_impact_disabled = (
            sum(r.impact_speed_disabled_kmh for r in records) / total
        )
        avg_impact_enabled = (
            sum(r.impact_speed_enabled_kmh for r in records) / total
        )

        # Relative collision impact speeds (closing velocity at impact)
        avg_rel_impact_disabled = (
            sum(r.impact_rel_speed_disabled_kmh for r in records) / total
        )
        avg_rel_impact_enabled = (
            sum(r.impact_rel_speed_enabled_kmh for r in records) / total
        )

        # Collision-conditioned averages (averaging only over scenarios that actually collided)
        avg_collision_rel_disabled = (
            sum(r.impact_rel_speed_disabled_kmh for r in records if r.coll_disabled) / collisions_disabled
            if collisions_disabled > 0 else 0.0
        )
        avg_collision_rel_enabled = (
            sum(r.impact_rel_speed_enabled_kmh for r in records if r.coll_enabled) / collisions_enabled
            if collisions_enabled > 0 else 0.0
        )

        # Impact speed reduction in cases that still collided (mitigation)
        unavoidable_records = [r for r in records if r.coll_disabled and r.coll_enabled]
        avg_mitigation_kmh = (
            sum(r.impact_speed_reduction_kmh for r in unavoidable_records) / len(unavoidable_records)
            if unavoidable_records else 0.0
        )

        summary = {
            "total_scenarios": total,
            "collisions_disabled": collisions_disabled,
            "collision_rate_disabled_pct": round((collisions_disabled / total) * 100.0, 2),
            "collisions_enabled": collisions_enabled,
            "collision_rate_enabled_pct": round((collisions_enabled / total) * 100.0, 2),
            "collisions_prevented": prevented,
            "collision_reduction_pct": round(reduction_pct, 2),
            "avg_impact_speed_disabled_kmh": round(avg_impact_disabled, 2),
            "avg_impact_speed_enabled_kmh": round(avg_impact_enabled, 2),
            "avg_rel_impact_speed_disabled_kmh": round(avg_rel_impact_disabled, 2),
            "avg_rel_impact_speed_enabled_kmh": round(avg_rel_impact_enabled, 2),
            "avg_collision_rel_disabled_kmh": round(avg_collision_rel_disabled, 2),
            "avg_collision_rel_enabled_kmh": round(avg_collision_rel_enabled, 2),
            "avg_impact_mitigation_kmh": round(avg_mitigation_kmh, 2),
            "records": records
        }

        return summary

    def save_to_csv(self, records: List[ComparisonRecord], filename: str = "aeb_comparison_results.csv") -> None:
        """Saves detailed per-scenario comparison records to CSV."""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario_id", "description", "road_env", "friction", "weather",
                "initial_speed_kmh", "initial_gap_m",
                "collision_disabled", "impact_speed_disabled_kmh", "impact_rel_speed_disabled_kmh", "final_gap_disabled_m",
                "collision_enabled", "impact_speed_enabled_kmh", "impact_rel_speed_enabled_kmh", "final_gap_enabled_m",
                "collision_prevented", "impact_speed_reduction_kmh", "rel_speed_reduction_kmh"
            ])
            for r in records:
                writer.writerow([
                    r.scenario_id, r.description, r.road_env, r.friction, r.weather,
                    r.speed_kmh, r.initial_gap_m,
                    r.coll_disabled, r.impact_speed_disabled_kmh, r.impact_rel_speed_disabled_kmh, r.final_gap_disabled_m,
                    r.coll_enabled, r.impact_speed_enabled_kmh, r.impact_rel_speed_enabled_kmh, r.final_gap_enabled_m,
                    r.collision_prevented, r.impact_speed_reduction_kmh, r.rel_speed_reduction_kmh
                ])


def main():
    print("==================================================================================")
    print("           AEB-ENABLED VS. AEB-DISABLED COMPARATIVE BENCHMARK")
    print("==================================================================================")
    comparator = AEBComparator()
    results = comparator.run_benchmark()

    comparator.save_to_csv(results["records"], "aeb_comparison_results.csv")
    print(f"\nSaved detailed results to: aeb_comparison_results.csv")

    print("\n------------------------------- SCENARIO BREAKDOWN -------------------------------")
    print(f"{'ID':<13} {'Speed':<6} {'Gap':<5} {'Cond':<5} {'No AEB (Rel)':<14} {'AEB (Rel)':<12} {'Prevented?':<11} {'Mitigation':<10}")
    print("-" * 88)
    for r in results["records"]:
        no_aeb_str = f"{r.impact_rel_speed_disabled_kmh} km/h" if r.coll_disabled else "No Collision"
        aeb_str = f"{r.impact_rel_speed_enabled_kmh} km/h" if r.coll_enabled else "Safe Stop"
        prev_str = "YES" if r.collision_prevented else ("N/A (Safe)" if not r.coll_disabled else "NO")
        red_str = f"-{r.rel_speed_reduction_kmh} km/h" if r.rel_speed_reduction_kmh > 0 else "0.0"
        cond_str = f"{r.road_env[:3]}/{r.friction}"
        print(f"{r.scenario_id:<13} {r.speed_kmh:<6.0f} {r.initial_gap_m:<5.0f} {cond_str:<5} {no_aeb_str:<14} {aeb_str:<12} {prev_str:<11} {red_str:<10}")

    print("\n============================= AGGREGATE SUMMARY =============================")
    print(f"Total Scenarios Evaluated          : {results['total_scenarios']}")
    print(f"Collisions without AEB             : {results['collisions_disabled']} ({results['collision_rate_disabled_pct']}%)")
    print(f"Collisions with AEB                : {results['collisions_enabled']} ({results['collision_rate_enabled_pct']}%)")
    print(f"Total Collisions Prevented         : {results['collisions_prevented']}")
    print(f"Overall Collision Reduction Rate   : {results['collision_reduction_pct']}%")
    print("--- Ground Impact Speeds (Ego Ground Velocity) ---")
    print(f"Avg Ground Impact Speed without AEB: {results['avg_impact_speed_disabled_kmh']} km/h")
    print(f"Avg Ground Impact Speed with AEB   : {results['avg_impact_speed_enabled_kmh']} km/h")
    print("--- Relative Impact Speeds (Closing Velocity at Impact) ---")
    print(f"Avg Relative Impact without AEB    : {results['avg_rel_impact_speed_disabled_kmh']} km/h")
    print(f"Avg Relative Impact with AEB       : {results['avg_rel_impact_speed_enabled_kmh']} km/h")
    print(f"Avg Crash Severity in Collisions   : {results['avg_collision_rel_disabled_kmh']} km/h -> {results['avg_collision_rel_enabled_kmh']} km/h")
    print(f"Avg Mitigation in Unavoidable      : -{results['avg_impact_mitigation_kmh']} km/h")
    print("==============================================================================\n")


if __name__ == "__main__":
    main()
