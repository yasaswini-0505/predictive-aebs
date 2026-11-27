"""Runs scenarios in MetaDrive and records what happened, step by step."""
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import config as C
from simulation.environment import make_env, LeadVehicle
from perception.detector import LeadDetector
from control.aeb_controller import AEBController
from control.brake_mapper import BrakeMapper
from control.calibrate import brake_table, COMMANDS


@dataclass
class RunResult:
    scenario_id: str
    mode: str
    surface: str
    mass_kg: float
    category: str
    ego_speed_init_kmh: float
    lead_speed_init_kmh: float
    gap_init_m: float
    lead_accel_ms2: float
    outcome: str                          # COLLISION / STOPPED / STABLE / TIMEOUT
    collision: bool
    impact_closing_speed_kmh: float
    impact_ego_speed_kmh: float
    min_gap_m: float
    final_gap_m: float
    final_ego_speed_kmh: float
    mean_ego_speed_kmh: float
    first_brake_time_s: Optional[float]   # first PARTIAL/FULL decision
    brake_applied_time_s: Optional[float] # first step with deceleration > 0 (after the 0.20 s delay)
    max_stage: str
    max_probability: Optional[float]
    false_braking: bool                   # braking in a scenario marked safe
    metadrive_crash_flag: bool
    duration_s: float
    lead_lost_steps: int
    telemetry: List[dict] = field(default_factory=list, repr=False)

    def row(self):
        d = asdict(self); d.pop("telemetry"); return d


class GroupContext:
    """One MetaDrive world per (surface, mass): tyre friction and mass are fixed at creation."""

    def __init__(self, surface, mass_kg, render=False):
        self.env = make_env(surface, mass_kg, render=render)
        self.env.reset()
        cached = {}
        if C.CALIBRATION_FILE.exists():
            cached = json.loads(C.CALIBRATION_FILE.read_text()).get("brake_tables", {})
        key = f"{surface}_{int(mass_kg)}"
        if key in cached:                 # measured earlier by `main.py calibrate` with identical settings
            cmds, decs, self.table_source = cached[key]["commands"], cached[key]["decels"], "calibration file"
        else:                             # measure now (same procedure)
            cmds, decs, self.table_source = COMMANDS, brake_table(self.env), "measured at start"
        self.mapper = BrakeMapper(cmds, decs)
        self.surface, self.mass = surface, mass_kg

    def close(self):
        if self.env is not None:      # safe to call twice
            self.env.close()
            self.env = None


STAGE_ORDER = ["OFF", "PARTIAL", "FULL"]


def run_scenario(ctx: GroupContext, sc: C.Scenario, mode: str, record=True, on_step=None, hold_after_s=0.0):
    env = ctx.env
    env.reset()
    ego = env.agent
    ego.set_velocity(ego.heading, sc.ego_speed_kmh / 3.6)
    if env.config["use_render"]:
        env.engine.main_camera.track(ego)          # keep the chase camera on the ego car after every reset
    lead = LeadVehicle(env, ego, sc)
    detector = LeadDetector()
    ctrl = AEBController(mode, sc.mass_kg, sc.surface, ctx.mapper, sc.ego_speed_kmh / 3.6)
    tel, t, steps = [], 0.0, 0
    min_gap = float("inf"); outcome = "TIMEOUT"; impact = (0.0, 0.0); max_stage = "OFF"; max_p = None
    crash_flag, applied, speeds, lost = False, None, [], 0
    state = detector.update(0.0, ego, [lead.vehicle])
    try:
        while t < C.MAX_DURATION:
            out = ctrl.step(t, state)
            lead_a = lead.command(t)
            env.step(out.action)
            t = round(t + C.STEP_DT, 6); steps += 1
            crash_flag = crash_flag or bool(ego.crash_vehicle)
            state = detector.update(t, ego, [lead.vehicle])
            if not state.lead_found:
                lost += 1
            if STAGE_ORDER.index(out.stage) > STAGE_ORDER.index(max_stage):
                max_stage = out.stage
            if out.probability is not None:
                max_p = out.probability if max_p is None else max(max_p, out.probability)
            if out.decel_request > 0 and applied is None:
                applied = t - C.STEP_DT
            speeds.append(state.ego_speed_kmh)
            gap = state.distance_m if state.lead_found else None
            if gap is not None:
                min_gap = min(min_gap, gap)
            if record:
                tel.append(dict(t=t, ego_kmh=round(state.ego_speed_kmh, 3),
                                lead_kmh=None if gap is None else round(state.lead_speed_kmh, 3),
                                lead_cmd_kmh=round(lead.speed * 3.6, 3), lead_cmd_accel=round(lead_a, 3),
                                gap_m=None if gap is None else round(gap, 3),
                                rel_kmh=None if gap is None else round(state.relative_speed_kmh, 3),
                                lead_accel_est=None if gap is None else round(state.lead_accel_mps2, 3),
                                ttc=state.ttc_s, pttc=state.projected_ttc_s, brake_light=state.brake_light_detected,
                                stage=out.stage, prob=out.probability, decel_req=round(out.decel_request, 3),
                                action=round(out.action[1], 4), reason=out.reason))
            if on_step:
                on_step(t, state, out, sc, mode)
            # ---- end conditions (mirroring the original dynamic engine) ----
            if (gap is not None and gap <= 0.0) or ego.crash_vehicle:
                outcome = "COLLISION"
                impact = (max(0.0, state.relative_speed_kmh or 0.0), state.ego_speed_kmh)
                break
            if state.ego_speed_mps <= 0.05 and steps > 5:
                outcome = "STOPPED"; break
            if (gap is not None and state.relative_speed_mps <= 0.0 and lead.accel >= 0
                    and t > sc.lead_event_time_s + 0.5 and ctrl.brakes.target_stage == ctrl.brakes.target_stage.OFF
                    and (sc.lead_accel_ms2 == 0 or abs(lead.speed - sc.lead_target_speed_kmh / 3.6) < 1e-6)):
                outcome = "STABLE"; break
        # Demo only: keep the final scene on screen. After a collision both cars are held still and the
        # panel keeps showing the state at the moment of impact (post-impact physics values are meaningless).
        final_state = state
        for _ in range(int(hold_after_s / C.STEP_DT)):
            if outcome == "COLLISION":
                ego.set_velocity(ego.heading, 0.0); lead.vehicle.set_velocity(lead.vehicle.heading, 0.0)
                env.step([0.0, -1.0])
                shown = final_state
            else:
                lead.command(t)
                env.step([0.0, -1.0] if outcome == "STOPPED" else [0.0, ctrl.driver.action(ego.speed)])
                shown = detector.update(t, ego, [lead.vehicle])
            t = round(t + C.STEP_DT, 6)
            if on_step:
                on_step(t, shown, None, sc, mode, outcome)
    finally:
        lead.remove()
    final_gap = 0.0 if outcome == "COLLISION" else (state.distance_m if state.lead_found else float("nan"))
    return RunResult(
        scenario_id=sc.scenario_id, mode=mode, surface=sc.surface, mass_kg=sc.mass_kg, category=sc.category,
        ego_speed_init_kmh=sc.ego_speed_kmh, lead_speed_init_kmh=sc.lead_speed_kmh, gap_init_m=sc.gap_m,
        lead_accel_ms2=sc.lead_accel_ms2, outcome=outcome, collision=outcome == "COLLISION",
        impact_closing_speed_kmh=round(impact[0], 2), impact_ego_speed_kmh=round(impact[1], 2),
        min_gap_m=round(max(0.0, min_gap if outcome != "COLLISION" else 0.0), 3), final_gap_m=round(final_gap, 3),
        final_ego_speed_kmh=round(state.ego_speed_kmh, 2), mean_ego_speed_kmh=round(sum(speeds) / max(1, len(speeds)), 2),
        first_brake_time_s=ctrl.first_brake_time, brake_applied_time_s=None if applied is None else round(applied, 3),
        max_stage=max_stage, max_probability=None if max_p is None else round(max_p, 4),
        false_braking=bool(sc.expect_no_braking and ctrl.first_brake_time is not None),
        metadrive_crash_flag=crash_flag, duration_s=round(t, 3), lead_lost_steps=lost, telemetry=tel)


def run_many(scenarios, modes, render=False, progress=True):
    """Groups scenarios by (surface, mass) so each MetaDrive world is created once."""
    results = []
    groups = {}
    for sc in scenarios:
        groups.setdefault((sc.surface, sc.mass_kg), []).append(sc)
    calibrations = {}
    for (surface, mass), scs in groups.items():
        ctx = GroupContext(surface, mass, render=render)
        calibrations[f"{surface}_{int(mass)}kg"] = dict(wheel_friction=ctx.env.aeb_wheel_friction, source=ctx.table_source, **ctx.mapper.to_dict())
        try:
            for sc in scs:
                for m in modes:
                    t0 = time.time()
                    r = run_scenario(ctx, sc, m)
                    results.append(r)
                    if progress:
                        print(f"  {sc.scenario_id:<32} {m:<8} {r.outcome:<9} min gap {r.min_gap_m:6.2f} m"
                              f"  brake@{r.first_brake_time_s}  ({time.time() - t0:.1f}s)")
        finally:
            ctx.close()
    return results, calibrations
