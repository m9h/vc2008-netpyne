#!/usr/bin/env python3
"""Phase-response curve: does the VC2008 circuit reproduce the mouse BF-PV optogenetic result?

Experiment being modeled (Sci Data 2020, s41597-020-00621-z, dataset 2): 40 Hz click-train ASSR
with 1 ms optogenetic pulses driving basal-forebrain PV neurons at phase delays of
0 / 6.25 / 12.5 / 18.75 ms relative to sound onset (= 0/90/180/270 deg of the 25 ms cycle).
Reported effect: in-phase / advanced stimulation ENHANCES the ASSR; out-of-phase / delayed
stimulation REDUCES it.

This is a *held-out causal prediction* for the model: nothing here is fitted to that dataset.
The model is simply run with an extra phase-shifted inhibitory drive onto its interneuron
populations (BF-PV is GABAergic and targets cortical fast-spiking cells) and the 40 Hz ASSR power
is measured as a function of phase.

    python src/prc.py                       # 8 phases + no-opto baseline, mouse variant
    python src/prc.py --phases 0 6.25 12.5 18.75 --trials 3
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(phase, trial, outdir, species="mouse", sign="inhibitory", opto=True, drive=40, w=1.0,
        condition="control"):
    env = dict(os.environ, VC_SPECIES=species, VC_CONDITION=condition, VC_DRIVE=str(drive),
               VC_TRIAL=str(trial), VC_SUBJECT="0", VC_OUT=outdir,
               VC_OPTO="1" if opto else "0", VC_OPTO_PHASE=str(phase),
               VC_OPTO_SIGN=sign, VC_OPTO_W=str(w))
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "init.py")],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=3600)
    import re
    m = re.search(r'\{[^{]*"schema".*?\n\}', r.stdout, re.S)
    if not m:
        return None
    return json.loads(m.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=float, nargs="*",
                    default=[0.0, 3.125, 6.25, 9.375, 12.5, 15.625, 18.75, 21.875])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sign", default="inhibitory", choices=["inhibitory", "excitatory"])
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--condition", default="control", choices=["control","schizophrenia"])
    ap.add_argument("--outdir", default=os.path.join(REPO, "output", "prc"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    from joblib import Parallel, delayed
    jobs = [("baseline", None, t) for t in range(a.trials)] + \
           [("opto", p, t) for p in a.phases for t in range(a.trials)]
    print(f"phase-response curve: {len(a.phases)} phases x {a.trials} trials + baseline "
          f"(opto={a.sign}, w={a.weight}, cond={a.condition})", flush=True)
    res = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(run)(0.0 if p is None else p, t, a.outdir, sign=a.sign,
                     opto=(kind == "opto"), w=a.weight, condition=a.condition)
        for kind, p, t in jobs)

    base = [r["power_40Hz"] for (k, _, _), r in zip(jobs, res) if k == "baseline" and r]
    b = float(np.mean(base)) if base else float("nan")
    print(f"\nbaseline (no opto) 40 Hz power: {b:.2f}  (n={len(base)})")
    print(f"\n  {'phase(ms)':>10}{'deg':>6}{'P40':>10}{'vs baseline':>13}")
    curve = []
    for p in a.phases:
        vals = [r["power_40Hz"] for (k, pp, _), r in zip(jobs, res)
                if k == "opto" and pp == p and r]
        if not vals:
            continue
        m = float(np.mean(vals))
        curve.append({"phase_ms": p, "deg": 360.0 * p / 25.0, "p40": m,
                      "ratio": m / b if b else None, "n": len(vals)})
        print(f"  {p:>10.3f}{360.0*p/25.0:>6.0f}{m:>10.2f}{m/b if b else 0:>12.2f}x")

    json.dump({"baseline_p40": b, "sign": a.sign, "weight": a.weight, "curve": curve},
              open(os.path.join(a.outdir, "prc.json"), "w"), indent=2)

    if curve:
        best = max(curve, key=lambda c: c["p40"]); worst = min(curve, key=lambda c: c["p40"])
        print(f"\n  max enhancement @ {best['phase_ms']:g} ms ({best['deg']:.0f} deg): "
              f"{best['ratio']:.2f}x baseline")
        print(f"  max suppression @ {worst['phase_ms']:g} ms ({worst['deg']:.0f} deg): "
              f"{worst['ratio']:.2f}x baseline")
        print("\n  PAPER predicts: enhancement near in-phase (0 deg), suppression near "
              "out-of-phase (180 deg = 12.5 ms)")


if __name__ == "__main__":
    main()
