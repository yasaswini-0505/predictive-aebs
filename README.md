# MetaDrive integration of the BTP AEB system

A software simulation, built on MetaDrive 0.4.3. No real vehicle, radar, camera or LiDAR is involved.

## What it does

Two cars run in one lane of a straight MetaDrive road. The lead car follows a script: it cruises, then brakes, or accelerates, at a set time. The ego car is controlled by your **original** BTP code:

```
MetaDrive vehicle states ──> perception/detector.py      (nearest car ahead in the ego lane, gap, closing speed,
                                                          TTC, projected TTC, estimated lead acceleration)
                         ──> BTP agent.evaluate_state()   (predictive_aeb_agent.py / KinematicAEBAgent)
                         ──> BTP BrakingController        (OFF / PARTIAL 40% / FULL, 0.20 s delay, min(F/m, mu*g))
                         ──> control/brake_mapper.py      (deceleration -> MetaDrive brake command, from measurements)
                         ──> env.step([steering, throttle_brake])   every 0.02 s
```

Nothing is retrained, copied or reimplemented. The modes are:

| Mode | Code used |
|---|---|
| `no_aeb` | Cruise driver only, who never brakes |
| `physics` | `KinematicAEBAgent` (experiment_close_obstacle_baseline.py) |
| `ml` | `PredictiveAEBAgent(enable_physics_fallback=False)` with `aeb_predictive_rf_model.pkl` |
| `static_ml` | `UltraFastComparativeAEBAgent(enable_physics_fallback=False)` with the initial `aeb_random_forest_model.pkl` |
| `static_hybrid` | `UltraFastComparativeAEBAgent(enable_physics_fallback=True)` with the initial model |
| `hybrid` | `PredictiveAEBAgent(enable_physics_fallback=True)` |

## Commands

Run these on a Mac, from the package folder:

```bash
cd /Users/udaykiran/Desktop/BTP/metadrive_aeb_integration
```
```bash
~/metadrive_env/bin/python main.py check                       # model loads; feature order matches
```
```bash
~/metadrive_env/bin/python main.py demo --scenario S02_sudden --all      # 3D window: all 6 approaches + scoreboard
```
```bash
~/metadrive_env/bin/python main.py demo --scenario S02_sudden --mode hybrid --compare   # No AEB vs one approach
```
```bash
~/metadrive_env/bin/python main.py compare --suite all         # 6 modes x 63 scenarios, saved in results/
```
```bash
~/metadrive_env/bin/python -m pytest tests -v                  # 19 tests
```

`main.py calibrate` re-measures the braking. You need it only if you change vehicle settings.

To run a demo scenario, pass any id from `scenarios.py`: S01 to S12, or a `G_...` id from the grid suite.

## What is measured and what is simulated

- **Read from MetaDrive:** positions, velocities and headings of both cars, plus the ego lane index and lane width.
- **Calculated:** gap (bumper to bumper, along the ego heading), closing speed, TTC (`None` when the closing speed is 0.05 m/s or less), and projected TTC (the project's `compute_projected_ttc`).
- **Estimated:** lead acceleration, from the change in lead speed over 0.1 s.
- **Simulated:** the brake light, ON when estimated lead acceleration is below -0.8 m/s², the same rule as the original engine. There is no camera.
- **Given to the agent:** friction, which is the surface's μ (Dry 0.85, Wet 0.50, Icy 0.20), and the road and weather labels (Dry/Clear, Wet/Rain, Icy/Fog). MetaDrive's graphics don't show rain or fog.
- **Fixed for repeatability:** ego mass, brake force (150), engine force (800) and tyre friction, through `VaryingDynamicsEnv` with min = max. Runs are deterministic, and two full runs produced identical CSVs.

## Calibration (control/brake_calibration.json)

MetaDrive's brake command does not mean "fraction of a stated deceleration". So I measured it on a rolling car, for each surface and mass, and invert that table. For Dry at 1500 kg, a command of 0.4 gives 7.96 m/s², and the controller's FULL request is 8.0 m/s². Wet and Icy tyre friction were chosen by search, so that MetaDrive's maximum braking equals μ·g: 4.90 and 1.96 m/s².

## Known limitations

- MetaDrive cuts engine force above 80 km/h, so ego speeds stay at 80 km/h or below.
- The lead car is moved by setting its velocity each step. It is not driven by its own brakes.
- The AEB runs every 0.02 s. The original engine ran every 0.01 s.
- The run ends when the ego car stops, collides, settles behind a car that is no longer braking, or reaches 12 s. This mirrors the original engine.
- MetaDrive's own crash flag is recorded, but a collision is declared when the gap reaches 0 m.
- **These are new MetaDrive results.** They must not be merged with the original 500-scenario custom-engine results (385, 320, 238, 88, 87).
