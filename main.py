"""
MetaDrive + BTP AEB integration. Run with the MetaDrive Python environment:

  cd /Users/udaykiran/Desktop/BTP/metadrive_aeb_integration
  ~/metadrive_env/bin/python main.py calibrate
  ~/metadrive_env/bin/python main.py demo --scenario S02_sudden --all
  ~/metadrive_env/bin/python main.py demo --scenario S02_sudden --mode hybrid
  ~/metadrive_env/bin/python main.py compare --suite named
"""
import argparse
import sys

import config as C


def cmd_calibrate(_):
    from control.calibrate import run_calibration
    run_calibration()


def cmd_check(_):
    from ml.btp_agents import model_feature_names, make_agent
    raw, encoded = model_feature_names()
    print("Model:", C.PREDICTIVE_MODEL)
    print("Raw input columns :", raw)
    print("Encoded features  :", encoded)
    agent = make_agent("hybrid")
    print("Agent FEATURE_NAMES match pipeline:", [n.split("__", 1)[1] for n in encoded] == agent.FEATURE_NAMES)


def cmd_compare(a):
    import scenarios as S
    from evaluation.report import run_and_report
    scs = S.SUITES[a.suite]() if not a.scenario else [S.get(x) for x in a.scenario]
    run_and_report(scs, a.modes, out_name=a.out or a.suite)


def cmd_demo(a):
    import scenarios as S
    from visualization.demo import run_demo
    if a.all:
        modes = list(C.MODES)
    elif a.compare and a.mode != "no_aeb":
        modes = ["no_aeb", a.mode]
    else:
        modes = [a.mode]
    run_demo(S.get(a.scenario), modes, shots_dir=a.screenshots, video=a.record)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate", help="measure MetaDrive braking and tyre friction").set_defaults(fn=cmd_calibrate)
    sub.add_parser("check", help="verify the saved model and feature order").set_defaults(fn=cmd_check)
    c = sub.add_parser("compare", help="run modes headless and export results")
    c.add_argument("--suite", default="named", choices=["named", "grid", "all"])
    c.add_argument("--scenario", nargs="*")
    c.add_argument("--modes", nargs="*", default=C.MODES, choices=C.MODES)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    d = sub.add_parser("demo", help="open the 3D window")
    d.add_argument("--scenario", default="S02_sudden")
    d.add_argument("--mode", default="hybrid", choices=C.MODES)
    d.add_argument("--compare", action="store_true", help="play No AEB first, then the chosen mode")
    d.add_argument("--all", action="store_true", help="play all six approaches, then a scoreboard")
    d.add_argument("--screenshots", help="folder to save key frames (optional)")
    d.add_argument("--record", help="save the window as an MP4 video, e.g. results/demo_all.mp4")
    d.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
