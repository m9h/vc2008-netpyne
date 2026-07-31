#!/usr/bin/env python3
"""Run the paper's full design and produce grand-averaged spectra (Fig. 3 equivalent).

METHODS: "Ten simulated patients are created for each cohort by selecting a random connectivity and
fixing that connectivity for 10 trials where background noise is distinct trial to trial. ... The
grand average in time over all subjects and trials is taken for each cohort and then the frequency
transform is performed."

    python src/batch.py                     # 10 subjects x 10 trials x 3 drives x 2 conditions
    python src/batch.py --subjects 3 --trials 4 --jobs 16

Writes output/grand/<condition>_<drive>Hz_grand.npz (grand-averaged MEG + spectrum) and prints the
2 x 3 power table. Each individual run is a separate process (NEURON state is global), fanned out
with joblib -- the same in-allocation ensemble pattern used for NSG.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = ("control", "schizophrenia")
DRIVES = (20, 30, 40)


def run_one(condition, drive, subject, trial, outdir):
    env = dict(os.environ, VC_CONDITION=condition, VC_DRIVE=str(drive),
               VC_SUBJECT=str(subject), VC_TRIAL=str(trial), VC_OUT=outdir)
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "init.py")],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=3600)
    label = f"vc2008_{condition}_{drive}Hz_s{subject}_t{trial}"
    path = os.path.join(outdir, f"{label}.npz")
    return path if os.path.isfile(path) else None


def grand_average(paths, dt_ms):
    """Average MEG in time across subjects/trials, then take the spectrum (METHODS order)."""
    megs = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        megs.append(z["meg"])
    n = min(len(m) for m in megs)
    M = np.stack([m[:n] for m in megs])
    grand = M.mean(axis=0)
    seg = grand[-4096:] if len(grand) >= 4096 else grand
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 / len(seg)
    freqs = np.fft.rfftfreq(len(seg), d=dt_ms / 1000.0)
    return grand, freqs, spec, len(megs)


def band(freqs, spec, f0, half=2.0):
    m = (freqs > f0 - half) & (freqs < f0 + half)
    return float(spec[m].max()) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--outdir", default=os.path.join(REPO, "output", "batch"))
    ap.add_argument("--granddir", default=os.path.join(REPO, "output", "grand"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.granddir, exist_ok=True)

    jobs = [(c, d, s, t) for c in CONDITIONS for d in DRIVES
            for s in range(a.subjects) for t in range(a.trials)]
    print(f"running {len(jobs)} simulations on {a.jobs} workers "
          f"({a.subjects} subjects x {a.trials} trials x {len(DRIVES)} drives x "
          f"{len(CONDITIONS)} conditions)", flush=True)

    from joblib import Parallel, delayed
    paths = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(run_one)(c, d, s, t, a.outdir) for c, d, s, t in jobs)

    ok = [p for p in paths if p]
    print(f"\ncompleted {len(ok)}/{len(jobs)} runs")
    if len(ok) < len(jobs):
        print(f"  WARNING: {len(jobs) - len(ok)} runs produced no output (not silently dropped)")

    print("\ngrand-averaged spectral power (MEG proxy):")
    print(f"  {'condition':<15}{'drive':>7}{'P20':>10}{'P30':>10}{'P40':>10}{'n':>6}")
    table = {}
    for c in CONDITIONS:
        for d in DRIVES:
            sel = [p for p in ok if f"_{c}_{d}Hz_" in os.path.basename(p)]
            if not sel:
                continue
            grand, freqs, spec, n = grand_average(sel, 0.1)
            np.savez_compressed(os.path.join(a.granddir, f"{c}_{d}Hz_grand.npz"),
                                meg=grand, freqs=freqs, power=spec, n_runs=n,
                                condition=c, drive=d)
            p20, p30, p40 = (band(freqs, spec, f) for f in (20, 30, 40))
            table[f"{c}_{d}"] = dict(p20=p20, p30=p30, p40=p40, n=n)
            print(f"  {c:<15}{d:>5}Hz{p20:>10.2f}{p30:>10.2f}{p40:>10.2f}{n:>6}")

    json.dump(table, open(os.path.join(a.granddir, "summary.json"), "w"), indent=2)
    c40 = table.get("control_40", {}).get("p40")
    s40 = table.get("schizophrenia_40", {}).get("p40")
    if c40 and s40:
        print(f"\nkey result: 40 Hz power control {c40:.2f} -> schizophrenia {s40:.2f} "
              f"({c40/s40:.1f}x reduction)")


if __name__ == "__main__":
    main()
