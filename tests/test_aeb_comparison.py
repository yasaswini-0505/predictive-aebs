import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_aeb_comparison import (
    generate_benchmark_scenarios,
    AEBComparator,
    ScenarioConfig,
    ComparisonRecord
)


def test_generate_benchmark_scenarios():
    """Verify that the benchmark generator yields 20 diverse standardized test scenarios."""
    scenarios = generate_benchmark_scenarios()
    assert len(scenarios) == 20, f"Expected 20 scenarios, found {len(scenarios)}"
    assert all(isinstance(s, ScenarioConfig) for s in scenarios)

    # Verify ID format and diversity
    ids = [s.scenario_id for s in scenarios]
    assert len(set(ids)) == 20, "Scenario IDs must be unique"
    assert any("URBAN" in i for i in ids), "Must include urban scenarios"
    assert any("SUBURBAN" in i for i in ids), "Must include suburban scenarios"
    assert any("HIGHWAY" in i for i in ids), "Must include highway scenarios"
    assert any("MOVING" in i for i in ids), "Must include moving lead vehicle scenarios"


def test_aeb_comparator_evaluation_physics():
    """Verify that AEBComparator executes paired evaluations and records physical deltas."""
    comparator = AEBComparator()
    # Moderate city scenario
    sc = ScenarioConfig("TEST_URB", "City test", 40.0, 1400.0, 25.0, 40.0, 0.85, "Dry", "Clear")
    record = comparator.evaluate_scenario(sc)

    assert isinstance(record, ComparisonRecord)
    assert record.scenario_id == "TEST_URB"
    assert record.coll_disabled is True, "Baseline driver inattention should collide"
    assert record.coll_enabled is False, "AEB intervention should safely prevent collision"
    assert record.collision_prevented is True
    assert record.impact_speed_reduction_kmh > 0.0


def test_aeb_comparator_relative_speed_metrics():
    """Verify ground impact speed vs relative impact speed accounting in paired runs."""
    comparator = AEBComparator()
    # Moving lead vehicle scenario: v_ego = 80, v_rel = 30 => v_obs = 50 km/h
    sc = ScenarioConfig("TEST_MOV", "Moving lead", 80.0, 1400.0, 10.0, 30.0, 0.85, "Dry", "Clear")
    rec = comparator.evaluate_scenario(sc)

    assert rec.impact_rel_speed_disabled_kmh <= rec.impact_speed_disabled_kmh
    assert rec.rel_speed_reduction_kmh >= 0.0


def test_csv_export(tmp_path):
    """Verify that evaluation records are exported correctly to CSV with all required headers."""
    comparator = AEBComparator()
    scenarios = generate_benchmark_scenarios()[:3]
    records = [comparator.evaluate_scenario(sc) for sc in scenarios]

    out_file = tmp_path / "test_comparison.csv"
    comparator.save_to_csv(records, str(out_file))

    assert out_file.exists(), "CSV output file was not created"
    import pandas as pd
    df = pd.read_csv(out_file)
    assert len(df) == 3
    required_cols = [
        "scenario_id", "description", "road_env", "friction", "weather",
        "initial_speed_kmh", "initial_gap_m", "collision_disabled",
        "impact_speed_disabled_kmh", "final_gap_disabled_m", "collision_enabled",
        "impact_speed_enabled_kmh", "final_gap_enabled_m", "collision_prevented"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing expected column {c} in exported CSV"

