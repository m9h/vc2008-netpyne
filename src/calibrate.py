#!/usr/bin/env python3
"""Calibrate synScale against the paper's own stated behaviour.

The paper fixes a quantitative target for the *undriven* network (METHODS):
  "With this level of noise input, in the absence of click train drive, pyramidal cells in the
   GENESIS model fire at an average rate of 29 Hz."

That is the criterion used here: scan synScale, run the network with background noise only
(no click-train drive), and report the mean pyramidal firing rate. Pick the scale whose rate is
closest to 29 Hz. Result is recorded in DEVIATIONS.md and becomes the default in cfg.py.

    /tmp/vc_venv/bin/python src/calibrate.py 4 8 12 16 20
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_HZ = 29.0

SNIPPET = r"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath("{here}/cfg.py")))
os.environ["VC_DRIVE_OFF"] = "1"
from netpyne import sim
from cfg import cfg
cfg.duration = 500.0
cfg.driveNumber = 0          # background noise only -- no click train
import netParams as npm
sim.createSimulateAnalyze(netParams=npm.netParams, simConfig=cfg)
rates = sim.allSimData.get("popRates", {})
print("RESULT " + json.dumps({"synScale": cfg.synScale, "pyr_hz": rates.get("PYR", 0.0),
                              "bask_hz": rates.get("BASK", 0.0), "chand_hz": rates.get("CHAND", 0.0)}))
"""


def main(scales):
    print(f"calibrating synScale against the paper's undriven target: PYR ~ {TARGET_HZ} Hz\n")
    rows = []
    for s in scales:
        env = dict(os.environ, VC_SYNSCALE=str(s), VC_OUT="/tmp/vc_cal")
        code = SNIPPET.format(here=HERE)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=os.path.dirname(HERE), env=env, timeout=1800)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            import json
            d = json.loads(line[len("RESULT "):])
            rows.append(d)
            print(f"  synScale={d['synScale']:6.2f}  PYR={d['pyr_hz']:7.2f} Hz  "
                  f"BASK={d['bask_hz']:7.2f}  CHAND={d['chand_hz']:7.2f}")
        else:
            print(f"  synScale={s:6.2f}  FAILED: {r.stderr.strip().splitlines()[-1:] }")
    if rows:
        best = min(rows, key=lambda d: abs(d["pyr_hz"] - TARGET_HZ))
        print(f"\nclosest to {TARGET_HZ} Hz -> synScale={best['synScale']} "
              f"(PYR {best['pyr_hz']:.2f} Hz)")


if __name__ == "__main__":
    vals = [float(a) for a in sys.argv[1:]] or [4, 8, 12, 16, 20]
    main(vals)
