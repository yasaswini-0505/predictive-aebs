"""
3D MetaDrive demonstration with a live on-screen panel (Panda3D OnscreenText).

Every number on screen comes from the running simulation step: detector output,
the BTP agent's decision and probability, and the action sent to MetaDrive.
Simulated signals (brake light, lead acceleration estimate) are labelled as such.
"""
from pathlib import Path

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Filename

import config as C
from evaluation.runner import GroupContext, run_scenario

GREEN, YELLOW, RED, WHITE, BLUE = (0.2, 0.9, 0.3, 1), (1, 0.85, 0.1, 1), (1, 0.25, 0.25, 1), (1, 1, 1, 1), (0.35, 0.65, 1, 1)


class Hud:
    def __init__(self, engine, shots_dir=None):
        self.engine = engine
        a2d = engine.aspect2d
        common = dict(parent=a2d, mayChange=True, align=TextNode.ALeft, shadow=(0, 0, 0, 1))
        self.title = OnscreenText(text="", pos=(-1.75, 0.9), scale=0.055, fg=WHITE, **common)
        self.panel = OnscreenText(text="", pos=(-1.75, 0.8), scale=0.045, fg=WHITE, bg=(0, 0, 0, 0.55), **common)
        self.stage = OnscreenText(text="", pos=(0.55, 0.8), scale=0.06, fg=GREEN, bg=(0, 0, 0, 0.55), **common)
        self.banner = OnscreenText(text="", pos=(0, -0.75), scale=0.085, fg=RED, parent=a2d, mayChange=True,
                                   align=TextNode.ACenter, shadow=(0, 0, 0, 1), bg=(0, 0, 0, 0.6))
        self.footer = OnscreenText(text="Software simulation in MetaDrive. Distances/speeds read from the simulator; "
                                        "brake light and lead deceleration are simulated estimates.",
                                   pos=(0, -0.95), scale=0.033, fg=(0.85, 0.85, 0.85, 1), parent=a2d, align=TextNode.ACenter)
        self.shots = Path(shots_dir) if shots_dir else None
        self._shot_taken = set()
        self.lead = None
        self.last_out = None
        self.run_label = ""            # e.g. "Run 3 of 6: Physics-only AEB"
        self.board = None

    @staticmethod
    def _f(v, fmt, unit, none="--"):
        return none if v is None else f"{v:{fmt}} {unit}"

    def update(self, t, s, out, sc, mode, outcome=None):
        if out is None:                 # run finished: keep showing the last decision
            out = self.last_out
        self.last_out = out
        label = C.MODE_LABELS[mode] + ("   (run finished)" if outcome else "")
        self.title.setText(f"{label}   |   {sc.description}   ({sc.surface} road, {int(sc.mass_kg)} kg)")
        ttc = "not closing" if s.ttc_s is None else f"{s.ttc_s:.2f} s"
        pttc = "--" if s.projected_ttc_s is None else f"{s.projected_ttc_s:.2f} s"
        prob = "n/a (no ML in this mode)" if (out is None or out.probability is None) else f"{100 * out.probability:.0f} %"
        lead_state = "BRAKING" if s.brake_light_detected else ("Accelerating" if (s.lead_accel_mps2 or 0) > 0.3 else "Normal")
        lines = [
            f"Time            {t:5.2f} s",
            f"Ego speed       {max(0.0, s.ego_speed_kmh):6.1f} km/h",
            f"Lead speed      {self._f(None if s.lead_speed_kmh is None else max(0.0, s.lead_speed_kmh), '6.1f', 'km/h')}",
            f"Distance        {self._f(s.distance_m, '6.1f', 'm')}",
            f"Closing speed   {self._f(s.relative_speed_kmh, '6.1f', 'km/h')}",
            f"TTC             {ttc}",
            f"Projected TTC   {pttc}",
            f"Lead accel est. {self._f(s.lead_accel_mps2, '5.1f', 'm/s2')}",
            f"Lead state      {lead_state}  (sim. brake light {'ON' if s.brake_light_detected else 'off'})",
            f"ML probability  {prob}",
        ]
        if out is not None:
            brake = f"{100 * out.brake_command:.0f} % of max" if out.brake_command > 0 else (
                "waiting 0.20 s actuator delay" if out.stage != "OFF" else "off (driver throttle)")
            lines += [f"Brake command   {brake}", f"Decel request   {out.decel_request:4.1f} m/s2"]
        self.panel.setText("\n".join(lines))
        stage = out.stage if out is not None else "OFF"
        self.stage.setText(f" AEB: {stage} ")
        self.stage.setFg({"OFF": GREEN, "PARTIAL": YELLOW, "FULL": RED}[stage])
        if self.lead is not None:   # tint the lead car red while its (simulated) brake lights are on
            self.lead.origin.setColorScale((1, 0.45, 0.45, 1) if s.brake_light_detected else (1, 1, 1, 1))
        if outcome == "COLLISION":
            self.banner.setText(" COLLISION "); self.banner.setFg(RED); self._shot("collision", mode, sc)
        elif outcome in ("STOPPED", "STABLE", "TIMEOUT"):
            gap = "" if s.distance_m is None else f": {s.distance_m:.1f} m to the lead car"
            self.banner.setText(f" NO COLLISION{gap} "); self.banner.setFg(GREEN); self._shot("result", mode, sc)
        elif stage == "FULL":
            self.banner.setText(" COLLISION RISK DETECTED  -  FULL EMERGENCY BRAKING "); self.banner.setFg(RED)
            self._shot("full_braking", mode, sc, delay_key=t)
        elif stage == "PARTIAL":
            self.banner.setText(" COLLISION RISK DETECTED  -  PARTIAL BRAKING ACTIVATED "); self.banner.setFg(YELLOW)
        elif s.brake_light_detected:
            self.banner.setText(" Lead vehicle braking "); self.banner.setFg(BLUE)
        elif self.run_label:
            self.banner.setText(f" {self.run_label} "); self.banner.setFg(WHITE)
        else:
            self.banner.setText("")

    def scoreboard(self, sc, results):
        """Final table built from the runs that just happened in this window."""
        for w in (self.panel, self.stage, self.banner, self.title):
            w.setText("")
        head = f"{'Approach':<24}{'Result':<12}{'Closest gap':>12}{'Brake decision':>16}{'Impact speed':>14}"
        rows = [head, "-" * len(head)]
        for r in results:
            res = "COLLISION" if r.collision else "avoided"
            brake = "never" if r.first_brake_time_s is None else f"{r.first_brake_time_s:.2f} s"
            hit = f"{r.impact_closing_speed_kmh:.1f} km/h" if r.collision else "-"
            rows.append(f"{C.short_label(r.mode):<24}{res:<12}{r.min_gap_m:>10.1f} m{brake:>16}{hit:>14}")
        lead = next((r for r in results if r.mode == "no_aeb"), None)
        foot = f"Lead car starts braking at {sc.lead_event_time_s:.1f} s." if sc.lead_accel_ms2 < 0 else ""
        text = (f"SAME SCENARIO, {len(results)} APPROACHES\n{sc.description}\n"
                f"({sc.surface} road, ego {sc.ego_speed_kmh:.0f} km/h, gap {sc.gap_m:.0f} m)\n\n"
                + "\n".join(rows) + f"\n\n{foot}  Values measured in this MetaDrive run.")
        from panda3d.core import TextNode as TN
        self.board = OnscreenText(text=text, pos=(0, 0.55), scale=0.05, fg=WHITE, bg=(0, 0, 0, 0.8),
                                  parent=self.engine.aspect2d, align=TN.ACenter, mayChange=True, font=self._mono())
        if self.shots is not None:
            self._shot_taken.discard(("scoreboard", "all"))
            self._shot("scoreboard", "all", sc)

    def _mono(self):
        """A fixed-width font so the scoreboard columns line up (falls back to the default font)."""
        for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
            try:
                f = self.engine.loader.loadFont(path)
                if f is not None and f.isValid():
                    return f
            except Exception:
                pass
        return None

    def _shot(self, name, mode, sc, delay_key=None):
        key = (name, mode)
        if self.shots is None or key in self._shot_taken:
            return
        self._shot_taken.add(key)
        self.shots.mkdir(parents=True, exist_ok=True)
        self.engine.graphicsEngine.renderFrame()
        self.engine.win.saveScreenshot(Filename.fromOsSpecific(str(self.shots / f"{sc.scenario_id}_{mode}_{name}.png")))


class VideoRecorder:
    """Writes what the MetaDrive window shows to an MP4 file (every 2nd step -> 25 frames per second)."""

    def __init__(self, engine, path, every=2):
        import imageio_ffmpeg
        self.engine, self.every, self.n, self.path = engine, every, 0, path
        size = (engine.win.getXSize(), engine.win.getYSize())
        self.writer = imageio_ffmpeg.write_frames(str(path), size, fps=int(1 / (C.STEP_DT * every)),
                                                  quality=8, macro_block_size=1)
        self.writer.send(None)

    def capture(self):
        import numpy as np
        self.n += 1
        if self.n % self.every:
            return
        self.engine.graphicsEngine.renderFrame()
        tex = self.engine.win.getScreenshot()
        if tex is None:
            return
        img = np.frombuffer(tex.getRamImageAs("RGB"), dtype=np.uint8)
        img = img.reshape(tex.getYSize(), tex.getXSize(), 3)[::-1]      # Panda3D images are stored bottom-up
        self.writer.send(np.ascontiguousarray(img))

    def close(self):
        self.writer.close()
        print(f"Video saved: {self.path}")


def run_demo(sc, modes, shots_dir=None, hold_s=2.5, board_s=8.0, video=None):
    """Plays `modes` one after another on the same scenario, then shows a scoreboard."""
    ctx = GroupContext(sc.surface, sc.mass_kg, render=True)
    hud = Hud(ctx.env.engine, shots_dir)
    rec = VideoRecorder(ctx.env.engine, video) if video else None
    results = []
    try:
        for k, m in enumerate(modes, 1):
            hud.run_label = f"Run {k} of {len(modes)}: {C.MODE_LABELS[m]}" if len(modes) > 1 else ""

            def on_step(t, s, out, sc_, mode_, outcome=None):
                if hud.lead is None:
                    others = [v for v in ctx.env.engine.get_objects().values()
                              if v is not ctx.env.agent and hasattr(v, "origin") and type(v).__name__ == "LVehicle"]
                    hud.lead = others[0] if others else None
                hud.update(t, s, out, sc_, mode_, outcome)
                if rec:
                    rec.capture()
            hud.lead = None; hud.last_out = None
            r = run_scenario(ctx, sc, m, on_step=on_step, hold_after_s=hold_s)
            results.append(r)
            print(f"{C.MODE_LABELS[m]:<24}: {r.outcome:<9} min gap {r.min_gap_m:6.2f} m, first brake decision "
                  f"{r.first_brake_time_s} s, impact closing speed {r.impact_closing_speed_kmh} km/h")
        if len(results) > 1:
            hud.scoreboard(sc, results)
            ego = ctx.env.agent
            for _ in range(int(board_s / C.STEP_DT)):
                ego.set_velocity(ego.heading, 0.0)
                ctx.env.step([0.0, -1.0])
                if rec:
                    rec.capture()
    finally:
        if rec:
            rec.close()
        ctx.close()
    return results
