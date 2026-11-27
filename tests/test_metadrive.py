"""Runs real MetaDrive simulations headless (about 10 s in total)."""
import json

import pandas as pd
import pytest

import config as C
import scenarios as S
from evaluation.runner import GroupContext, run_scenario


@pytest.fixture(scope="module")
def dry():
    ctx = GroupContext("Dry", 1500.0)
    yield ctx
    ctx.close()


def test_controlled_scene_two_cars_same_lane(dry):
    sc = S.get("S02_sudden")
    r = run_scenario(dry, sc, "no_aeb")
    first = r.telemetry[0]
    assert abs(first["gap_m"] - sc.gap_m) < 0.5
    assert abs(first["ego_kmh"] - sc.ego_speed_kmh) < 1.0 and abs(first["lead_kmh"] - sc.lead_speed_kmh) < 1.0
    assert r.lead_lost_steps == 0
    lead_before = [x for x in r.telemetry if x["t"] < sc.lead_event_time_s]
    lead_after = [x for x in r.telemetry if x["t"] > sc.lead_event_time_s + 0.5]
    assert lead_after[-1]["lead_kmh"] < lead_before[-1]["lead_kmh"] - 10     # lead actually decelerates


def test_no_aeb_collides_and_hybrid_avoids(dry):
    sc = S.get("S02_sudden")
    base, hyb = run_scenario(dry, sc, "no_aeb"), run_scenario(dry, sc, "hybrid")
    assert base.collision and base.first_brake_time_s is None
    assert not hyb.collision and hyb.outcome == "STOPPED" and hyb.min_gap_m > 2.0
    speeds = [x["ego_kmh"] for x in hyb.telemetry]
    assert speeds[-1] < 1.0 < speeds[0]                                     # ego visibly slowed to a stop
    assert hyb.brake_applied_time_s - hyb.first_brake_time_s == pytest.approx(0.2, abs=0.021)


def test_braking_reaches_requested_deceleration(dry):
    r = run_scenario(dry, S.get("S02_sudden"), "hybrid")
    tel = r.telemetry
    i = next(k for k, x in enumerate(tel) if x["decel_req"] >= 7.9)
    v0, v1 = tel[i + 5]["ego_kmh"] / 3.6, tel[i + 15]["ego_kmh"] / 3.6
    measured = (v0 - v1) / (10 * C.STEP_DT)
    assert measured == pytest.approx(8.0, rel=0.1)


def test_safe_scenarios_do_not_brake(dry):
    for sid in ("S11_safe", "S12_lead_accel"):
        for m in C.MODES[1:]:
            r = run_scenario(dry, S.get(sid), m)
            assert not r.collision and r.first_brake_time_s is None, (sid, m)


def test_identical_initial_conditions_and_reset(dry):
    sc = S.get("S05_short_gap")
    runs = [run_scenario(dry, sc, m) for m in C.MODES] + [run_scenario(dry, sc, "hybrid")]
    starts = {(round(r.telemetry[0]["gap_m"], 3), round(r.telemetry[0]["ego_kmh"], 3)) for r in runs}
    assert len(starts) == 1
    assert runs[C.MODES.index("hybrid")].row() == runs[-1].row()            # reset gives a repeatable run


def test_result_export(dry, tmp_path, monkeypatch):
    dry.close()                         # MetaDrive allows one engine per process
    from evaluation import report
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    folder, df, summary = report.run_and_report([S.get("S02_sudden")], ["no_aeb", "hybrid"], out_name="t")
    for f in ("per_scenario_results.csv", "summary.csv", "telemetry.jsonl", "run_info.json", "comparison.png",
              "timeseries_S02_sudden.png"):
        assert (folder / f).exists(), f
    assert len(pd.read_csv(folder / "per_scenario_results.csv")) == 2
    assert "MetaDrive" in json.loads((folder / "run_info.json").read_text())["simulator"]
