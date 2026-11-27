"""Writes per-scenario CSV, a summary table, telemetry and plots for one experiment run."""
import json
import math
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from evaluation.runner import run_many

COLOURS = {"no_aeb": "#d62728", "static_ml": "#8c564b", "static_hybrid": "#e377c2", "physics": "#ff7f0e",
           "ml": "#9467bd", "hybrid": "#1f77b4"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[df["mode"] == "no_aeb"]["collision"].sum() if "no_aeb" in set(df["mode"]) else None
    for m in [x for x in C.MODES if x in set(df["mode"])]:
        d = df[df["mode"] == m]
        safe = d[d["category"] == "safe"]
        n_coll = int(d["collision"].sum())
        rows.append({
            "mode": C.MODE_LABELS[m],
            "scenarios": len(d),
            "collisions": n_coll,
            "collision_rate_pct": round(100 * n_coll / len(d), 2),
            "reduction_vs_no_aeb_pct": None if not base else round(100 * (base - n_coll) / base, 2),
            "avoided_where_no_aeb_collided": None,
            "stopped_safely": int((d["outcome"] == "STOPPED").sum()),
            "aeb_activated": int(d["first_brake_time_s"].notna().sum()),
            "false_braking_in_safe": f"{int(safe['false_braking'].sum())}/{len(safe)}",
            "avg_min_gap_m": round(d["min_gap_m"].mean(), 2),
            "avg_impact_closing_kmh": round(d.loc[d["collision"], "impact_closing_speed_kmh"].mean(), 2) if n_coll else 0.0,
            "avg_ego_speed_kmh": round(d["mean_ego_speed_kmh"].mean(), 2),
            "avg_first_brake_time_s": round(d["first_brake_time_s"].dropna().mean(), 3) if d["first_brake_time_s"].notna().any() else None,
        })
    out = pd.DataFrame(rows)
    if "no_aeb" in set(df["mode"]):
        crashed = set(df[(df["mode"] == "no_aeb") & df["collision"]]["scenario_id"])
        for i, m in enumerate([x for x in C.MODES if x in set(df["mode"])]):
            d = df[(df["mode"] == m) & df["scenario_id"].isin(crashed)]
            out.loc[i, "avoided_where_no_aeb_collided"] = f"{int((~d['collision']).sum())}/{len(crashed)}"
    return out


def plot_summary(df, summary, folder):
    modes = [m for m in C.MODES if m in set(df["mode"])]
    labels = [C.short_label(m) for m in modes]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    vals = [int(df[df["mode"] == m]["collision"].sum()) for m in modes]
    axes[0].bar(labels, vals, color=[COLOURS[m] for m in modes]); axes[0].set_title("Collisions")
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    rates = [100 * v / len(df[df["mode"] == m]) for v, m in zip(vals, modes)]
    axes[1].bar(labels, rates, color=[COLOURS[m] for m in modes]); axes[1].set_title("Collision rate (%)")
    axes[2].bar(labels, [df[df["mode"] == m]["min_gap_m"].mean() for m in modes], color=[COLOURS[m] for m in modes])
    axes[2].set_title("Average minimum gap (m)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"MetaDrive AEB comparison: {len(df) // len(modes)} scenarios (software simulation)")
    fig.tight_layout(); fig.savefig(folder / "comparison.png", dpi=130); plt.close(fig)


def plot_timeseries(results, scenario_id, folder):
    runs = [r for r in results if r.scenario_id == scenario_id]
    if not runs:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    ax = axes.ravel()
    for r in runs:
        tel = pd.DataFrame(r.telemetry)
        c, lab = COLOURS[r.mode], C.MODE_LABELS[r.mode]
        ax[0].plot(tel.t, tel.ego_kmh, color=c, label=f"Ego: {lab}")
        ax[1].plot(tel.t, tel.gap_m, color=c, label=lab)
        ax[2].plot(tel.t, tel.ttc.astype(float), color=c, label=lab)
        if r.mode in C.ML_MODES:
            ax[3].plot(tel.t, tel.prob.astype(float), color=c, label=lab)
        ax[4].step(tel.t, tel.stage.map({"OFF": 0, "PARTIAL": 1, "FULL": 2}), color=c, where="post", label=lab)
        ax[5].plot(tel.t, -tel.action.clip(upper=0), color=c, label=lab)
    lead = pd.DataFrame(runs[0].telemetry)
    ax[0].plot(lead.t, lead.lead_cmd_kmh, "k--", label="Lead car (no AEB run)")
    titles = ["Speed (km/h)", "Bumper gap (m)", "TTC (s, blank = not closing)", "ML collision probability",
              "AEB stage (0 OFF, 1 PARTIAL, 2 FULL)", "MetaDrive brake command (0-1)"]
    for a, tt in zip(ax, titles):
        a.set_title(tt); a.grid(alpha=.3)
    ax[2].set_ylim(0, 10); ax[3].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    for a in ax[4:]:
        a.set_xlabel("Simulation time (s)")
    fig.suptitle(f"{scenario_id}: MetaDrive simulation")
    fig.tight_layout(); fig.savefig(folder / f"timeseries_{scenario_id}.png", dpi=120); plt.close(fig)


def run_and_report(scenarios, modes, out_name="named"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = C.RESULTS_DIR / f"{out_name}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(scenarios)} scenarios x {len(modes)} modes in MetaDrive (headless)...")
    results, calib = run_many(scenarios, modes)
    df = pd.DataFrame([r.row() for r in results])
    df.to_csv(folder / "per_scenario_results.csv", index=False)
    summary = summarise(df)
    summary.to_csv(folder / "summary.csv", index=False)
    with open(folder / "telemetry.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({"scenario_id": r.scenario_id, "mode": r.mode, "telemetry": r.telemetry},
                               default=lambda x: None if isinstance(x, float) and math.isnan(x) else x) + "\n")
    (folder / "run_info.json").write_text(json.dumps(dict(
        simulator="MetaDrive 0.4.3 (VaryingDynamicsEnv, fixed dynamics)", created=stamp,
        scenarios=[s.to_dict() for s in scenarios], modes=modes, step_s=C.STEP_DT,
        braking_calibration=calib, note="New MetaDrive results. Not comparable one-to-one with the original 500-scenario custom-engine results."),
        indent=2))
    plot_summary(df, summary, folder)
    for sid in [s.scenario_id for s in scenarios][:12]:
        plot_timeseries(results, sid, folder)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + summary.to_string(index=False))
    print(f"\nSaved to {folder}")
    return folder, df, summary
