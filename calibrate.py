"""
Measures MetaDrive braking so BrakingController decelerations can be converted to actions.

1. For Wet and Icy roads, finds a MetaDrive tyre friction (wheel_friction) whose maximum
   braking is close to mu * g of that surface (bisection). Dry keeps MetaDrive's 0.9.
2. For a (surface, mass) pair, measures the deceleration produced by each brake command.
"""
import json

import numpy as np

import config as C
from simulation.environment import make_env

COMMANDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
G = 9.81
TABLE_MASSES = [1150.0, 1500.0, 2200.0]   # masses used by the scenarios


def measure(env, command, v0=15.0, settle_s=1.5, brake_s=0.6):
    """Steady-state deceleration (m/s^2) for one brake command.

    The car first drives for `settle_s` at constant speed (wheels rolling, as in a real run),
    then brakes; the first 0.1 s of braking is skipped and the rest is averaged.
    """
    from control.aeb_controller import CruiseDriver
    env.reset()
    ego = env.agent
    ego.set_velocity(ego.heading, v0)
    driver = CruiseDriver(v0)
    for _ in range(int(settle_s / C.STEP_DT)):
        env.step([0.0, driver.action(ego.speed)])
    for _ in range(int(0.1 / C.STEP_DT)):
        env.step([0.0, -command])
    start, n = ego.speed, int((brake_s - 0.1) / C.STEP_DT)
    for i in range(n):
        env.step([0.0, -command])
        if ego.speed < 1.0:                     # stopped early: use the time actually spent
            n = i + 1
            break
    return (start - ego.speed) / (n * C.STEP_DT)


def brake_table(env, commands=COMMANDS):
    return [round(measure(env, c), 3) for c in commands]


def _max_decel_with(wf, mass):
    env = make_env("Dry", mass, wheel_friction=wf)
    try:
        return measure(env, 1.0)
    finally:
        env.close()


def find_wheel_friction(target_decel, mass=1500.0, lo=0.001, hi=0.9, iters=12):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _max_decel_with(mid, mass) > target_decel:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 4)


def run_calibration():
    result = {"wheel_friction": {"Dry": 0.9}, "note": "MetaDrive tyre friction chosen so max braking ~= mu*g"}
    for surface in ("Wet", "Icy"):
        target = C.SURFACES[surface]["mu"] * G
        wf = find_wheel_friction(target)
        result["wheel_friction"][surface] = wf
        print(f"{surface}: mu*g = {target:.2f} m/s^2 -> wheel_friction {wf}, measured max {_max_decel_with(wf, 1500.0):.2f}")
    C.CALIBRATION_FILE.write_text(json.dumps(result, indent=2))   # tyre friction must exist before tables
    result["brake_tables"] = {}
    for surface in C.SURFACES:
        for mass in TABLE_MASSES:
            env = make_env(surface, mass)
            try:
                result["brake_tables"][f"{surface}_{int(mass)}"] = dict(commands=COMMANDS, decels=brake_table(env))
            finally:
                env.close()
    C.CALIBRATION_FILE.write_text(json.dumps(result, indent=2))
    print("Saved", C.CALIBRATION_FILE)
    return result
