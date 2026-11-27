"""Fill the professor's template with our content. Keeps the template's frame (title, step number,
time chip, tracker) and replaces prompts, worked examples and notes. Numbers match the report."""
import copy
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
D = json.loads((HERE / "team_details.json").read_text())
prs = Presentation(HERE / "template_src" / "template.pptx")

TIMES = ["1:00", "1:15", "0:45", "1:30", "1:30", "1:45", "1:00", "0:15"]   # total 9:00


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def set_paras(tf, lines, size=None):
    """Replace paragraphs, reusing the first paragraph's formatting. '**x** y' makes x bold."""
    ps = tf.paragraphs
    proto = copy.deepcopy(ps[0]._p)
    for p in ps[1:]:
        p._p.getparent().remove(p._p)
    body = ps[0]._p.getparent()
    body.remove(ps[0]._p)
    for line in lines:
        p = copy.deepcopy(proto)
        runs = p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r")
        for r in runs[1:]:
            p.remove(r)
        body.append(p)
    for para, line in zip(tf.paragraphs, lines):
        parts = re.split(r"\*\*(.+?)\*\*", line)
        run0 = para.runs[0] if para.runs else para.add_run()
        first = True
        if not line:
            run0.text = ""
        for i, part in enumerate(parts):
            if not part:
                continue
            r = run0 if first else para.add_run()
            if not first:
                r._r.insert(0, copy.deepcopy(run0._r.rPr)) if run0._r.rPr is not None else None
            r.text = part
            r.font.bold = True if i % 2 == 1 else None
            if size:
                r.font.size = Pt(size)
            first = False


def remove(slide, name):
    s = shape(slide, name)
    s._element.getparent().remove(s._element)


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None,
                                    Inches(h) if h else None)


def card_image(slide, path, y=1.6, maxh=4.5, x=7.6, w=5.1):
    """Replace the worked-example card with an image that fits inside it."""
    from PIL import Image
    iw, ih = Image.open(path).size
    width = w
    if width * ih / iw > maxh:
        width = maxh * iw / ih
    remove(slide, "Text 6"); remove(slide, "Text 7")
    picture(slide, path, x + (w - width) / 2, y, width)


def card_text_image(slide, label, lines, path, caption):
    from PIL import Image
    card_text(slide, label, lines, size=13)
    t7 = shape(slide, "Text 7"); t7.height = Inches(0.85)
    w = 4.6; iw, ih = Image.open(path).size; h = w * ih / iw
    picture(slide, path, 7.85, 3.3, w)
    cap = slide.shapes.add_textbox(Inches(7.85), Inches(3.3 + h + 0.02), Inches(w), Inches(0.3))
    cap.text_frame.text = caption
    r = cap.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(10); r.font.italic = True


def frame(slide, idx, heading, bullets, notes, size=None):
    set_paras(shape(slide, "Text 3").text_frame, [heading])
    body = [s for s in slide.placeholders if s.placeholder_format.idx != 0 and s.name == "Text 0"][-1]
    set_paras(body.text_frame, bullets, size)
    bp = body.text_frame._txBody.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr")
    for c in list(bp):
        if c.tag.endswith("Autofit"):
            bp.remove(c)
    slide.notes_slide.notes_text_frame.text = notes


def card_text(slide, label, lines, size=None):
    set_paras(shape(slide, "Text 6").text_frame, [label])
    set_paras(shape(slide, "Text 7").text_frame, lines, size)


s = prs.slides
# 1 Title -----------------------------------------------------------------------------------------
t = s[0]
set_paras(shape(t, "Text 0").text_frame if False else t.shapes[0].text_frame,
          ["Automatic Emergency Braking Using Machine Learning and Physics-Based Collision Prevention"])
team = [f"{m['name']}  |  {m['roll']}" for m in D["members"]]
set_paras(t.shapes[1].text_frame, team + ["Project type: BTP", "Evaluation: Second",
                                          f"Mentor: {D['mentor']}"])
set_paras(shape(t, "Text 3").text_frame, ["Video order (about 9 minutes)"])
names = ["Introduction", "Literature Review", "Quick Recap", "Methodology", "Implementation Details",
         "Results and Analysis", "Conclusion and Future Work", "References"]
set_paras(shape(t, "Text 4").text_frame, [f"{i}  {n}  ({TIMES[i-1]})" for i, n in enumerate(names, 1)])
t.notes_slide.notes_text_frame.text = (
    "Opening (all four members on camera, about 15 s before slide 2): each member says their name and roll number. "
    "Say: this is a software simulation project; no real vehicle, camera or radar was used.")

# 2 Introduction ---------------------------------------------------------------------------------
frame(s[1], 0, "Problem statement and motivation", [
    "**Problem:** identify the vehicle in front, compute distance and time-to-collision (TTC), and brake automatically before a crash.",
    "**Why it matters:** rear-end crashes are common; AEB systems cut front-to-rear crash rates by about 50% in insurance data [3].",
    "**Objective:** a hybrid ML + physics AEB that brakes early, avoids false braking, and is tested in a 3-D simulator.",
], "Introduce the problem: the system must find the car ahead, measure gap and TTC, and brake on its own. "
   "Explain that fixed thresholds react late when the lead car brakes hard. State the objective. "
   "Point at the picture: without AEB, the ego car hits the braking lead car in MetaDrive.")
card_text_image(s[1], "Our problem in one line", [
    "When the car ahead brakes suddenly, a late reaction causes a rear-end crash.",
    "**Objective:** detect the car ahead, compute gap and TTC, and brake early enough to avoid it.",
], FIG / "S02_sudden_no_aeb_collision.png", "Without AEB: collision in MetaDrive")

# 3 Literature review ----------------------------------------------------------------------------
frame(s[2], 1, "Related work", [
    "**Coelingh et al., 2010 [1]:** Volvo production AEB; rule-based thresholds, vehicle-specific.",
    "**Lee and Peng, 2005 [2]:** compared warning/avoidance algorithms; fixed rules trade late alarms vs false alarms.",
    "**Cicchino, 2017 [3]:** crash data shows AEB works; gives no algorithm.",
    "**Hayward, 1972 [4]:** TTC as danger measure; assumes constant speeds.",
    "**MetaDrive [5] / CARLA [6]:** open simulators for repeatable tests; no AEB method included.",
], "Go through the five works in one line each: what it does and where it falls short. Then read the gap on the right.", size=17)
card_text(s[2], "The gap our work fills", [
    "Classic AEB uses fixed thresholds on the current state only.",
    "",
    "We add a learned risk model that uses lead-vehicle behaviour (deceleration, projected TTC), keep a physics check as a safety net, and compare six approaches under identical, repeatable conditions in two simulators.",
])

# 4 Quick recap ----------------------------------------------------------------------------------
frame(s[3], 2, "Work from our mid evaluation", [
    "**Last time:** collision-risk classifier on 10,000 simulated scenarios (7 inputs); Random Forest best at 97.70% accuracy. It predicted risk only, with no braking.",
    f"**Feedback:** {D['interim_feedback']}",
    "**Since then:** closed-loop braking controller, predictive model, and the full system running in the MetaDrive 3-D simulator with a recorded demo video.",
], "Recap the mid evaluation: the classifier and its accuracy. Say the mentor asked for visible output such as a video. "
   "Then say what we built in response: braking control and the MetaDrive simulation with a demo video.")
card_text(s[3], "Mid evaluation result (2,000 test samples)", [
    "Logistic Regression: 96.30%",
    "Decision Tree: 97.45%",
    "Random Forest: 97.70% (F1 95.15%)",
    "",
    "**Now:** visible 3-D simulation and video, 55 → 2 collisions in MetaDrive.",
])

# 5 Methodology ----------------------------------------------------------------------------------
frame(s[4], 3, "Proposed approach", [
    "**Detection:** nearest vehicle ahead in the same lane; bumper-to-bumper gap.",
    "**Features:** closing speed, TTC, projected TTC, lead deceleration, brake light.",
    "**ML:** Random Forest (100 trees) gives P(collision).",
    "**Physics check:** stopping distance with 0.20 s delay and a = min(F/m, μg).",
    "**Hybrid:** take the more urgent of ML and physics: OFF / PARTIAL 40% / FULL.",
    "**Why:** RF handles mixed inputs and gives probabilities; physics catches ML misses.",
], "Walk through the block diagram from left to right: simulator, detection, features, the two decision paths, "
   "the hybrid rule, and the braking controller. Explain why we keep both ML and physics.", size=16)
card_image(s[4], FIG / "fig_block_diagram.png", y=2.0)

# 6 Implementation -------------------------------------------------------------------------------
frame(s[5], 4, "How we built it", [
    "**Software:** Python 3.12, scikit-learn 1.9.1, pandas, matplotlib, pytest, MetaDrive 0.4.3 (Panda3D/Bullet).",
    "**Data (synthetic):** 10,000 + 1,000 rows (static model); 36,358 trajectory snapshots (predictive model).",
    "**Simulators:** custom 1-D (0.01 s step) and MetaDrive 3-D (0.02 s step, brake calibrated for dry/wet/icy).",
    "**Hardware:** no physical vehicle, camera or sensors used.",
    "**Testing:** 99 automated tests, all passing.",
], "List tools and data. Then switch to the live MetaDrive demo or the recorded video: "
   "show the HUD with gap, TTC, ML probability and the FULL braking banner.", size=16)
card_text_image(s[5], "Our result in MetaDrive: a close call avoided", [
    "Wet road, ego at 60 km/h, lead car only 15 m ahead brakes suddenly at 6 m/s².",
    "Without AEB: crash at 47.8 km/h. Our Predictive Hybrid AEB applies full braking and stops:",
], FIG / "close_call_hybrid_stop.png", "Predictive Hybrid AEB: no collision, only 3.3 m to the lead car.")

# 7 Results --------------------------------------------------------------------------------------
frame(s[6], 5, "Mid evaluation vs now", [
    "**Sudden braking:** Hybrid brakes 0.5 s earlier than physics-only and stops 14.0 m short (vs 3.4 m).",
    "**Failure:** on ice (about 2 m/s² braking) impact is slowed 40.7 → 13.8 km/h but not avoided.",
], "Compare with the mid evaluation using the table: the mid model only predicted risk; used for braking it still "
   "had 42 collisions in MetaDrive, while the final Predictive Hybrid has 2. Point at the chart, then explain "
   "the icy-road failure: not enough grip for any rule to stop in time. Test data is clean simulation data.", size=13)
body = [x for x in s[6].placeholders if x.placeholder_format.idx != 0 and x.name == "Text 0"][-1]
body.top, body.height = Inches(5.1), Inches(1.2)
rows = [("", "Mid evaluation", "Final (now)"),
        ("Model", "Static RF, 7 inputs", "Predictive RF, 12 inputs"),
        ("Test accuracy", "97.70%", "99.53%"),
        ("Brakes the car?", "No (risk only)", "Yes, OFF / PARTIAL / FULL"),
        ("Visible output", "Accuracy table", "3-D simulation + video"),
        ("Collisions, custom sim (500)*", "320", "87"),
        ("Collisions, MetaDrive (63)*", "42", "2"),
        ("False braking, safe cases", "not tested", "0")]
tb = s[6].shapes.add_table(len(rows), 3, Inches(0.6), Inches(2.15), Inches(6.7), Inches(2.5)).table
tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Inches(2.5), Inches(1.9), Inches(2.3)
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tb.cell(r, c); cell.text = v
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        f = cell.text_frame.paragraphs[0].runs[0].font if v else None
        if f:
            f.size = Pt(11); f.bold = (r == 0 or c == 2)
note = s[6].shapes.add_textbox(Inches(0.6), Inches(4.78), Inches(6.7), Inches(0.3))
note.text_frame.text = "*Mid-evaluation model placed in the braking loop. No AEB: 385 and 55 collisions."
note.text_frame.paragraphs[0].runs[0].font.size = Pt(9); note.text_frame.paragraphs[0].runs[0].font.italic = True
remove(s[6], "Text 6"); remove(s[6], "Text 7")
picture(s[6], FIG / "fig_metadrive_63.png", 7.85, 1.6, 4.6)
picture(s[6], FIG / "fig_custom_500.png", 7.95, 4.05, 4.3)

# 8 Conclusion -----------------------------------------------------------------------------------
frame(s[7], 6, "Wrap up", [
    "**Objective met (in simulation):** detects the lead car, computes gap and TTC, brakes automatically; 2 collisions in 63 MetaDrive scenarios.",
    "**Main limitation:** synthetic training data and exact simulator measurements; straight roads, one lead car.",
], "Say clearly that the objective is met in simulation, then state the limitation and the next steps for each sub-team.")
card_text(s[7], "Next steps", [
    "1. Detect the car ahead from a simulated camera/LiDAR with bounding boxes.",
    "2. Test harder cases: curves, cut-ins, pedestrians.",
    "3. Train the model on MetaDrive data with sensor noise.",
])

# 9 References -----------------------------------------------------------------------------------
refs = [
    "[1] E. Coelingh et al., \"Collision warning with full auto brake and pedestrian detection,\" IEEE ITSC, 2010.",
    "[2] K. Lee and H. Peng, \"Evaluation of automotive FCW and collision avoidance algorithms,\" Veh. Syst. Dyn., 2005.",
    "[3] J. B. Cicchino, \"Effectiveness of FCW and AEB systems,\" Accid. Anal. Prev., 2017.",
    "[4] J. C. Hayward, \"Near-miss determination through use of a scale of danger,\" HRR, 1972.",
    "[5] Q. Li et al., \"MetaDrive,\" IEEE TPAMI, 2023.",
    "[6] A. Dosovitskiy et al., \"CARLA: An open urban driving simulator,\" CoRL, 2017.",
    "[7] L. Breiman, \"Random forests,\" Machine Learning, 2001.",
    "[8] F. Pedregosa et al., \"Scikit-learn,\" JMLR, 2011.",
    "[9] Euro NCAP, \"Test protocol - AEB car-to-car systems.\"",
    "[10] R. Krajewski et al., \"The highD dataset,\" IEEE ITSC, 2018.",
]
frame(s[8], 7, "Sources (IEEE style)", refs, "Show the references for a few seconds. Full versions are in the report.", size=12)
card_text(s[8], "Code and data", [
    "All code, datasets, trained models and results are in our project folder (metadrive_aeb_integration/, final_evaluation/).",
    "Datasets are synthetic, generated by our own scripts.",
])

for sl in list(prs.slides)[1:]:
    for sh in list(sl.shapes):
        if sh.name.startswith("Tracker") or sh.name == "Time chip":
            sh._element.getparent().remove(sh._element)
for nm in ("Agenda card", "Text 3", "Text 4"):
    remove(prs.slides[0], nm)
out = HERE / "BTP_AEB_Final_Presentation.pptx"
prs.save(out)
print("Saved", out)
