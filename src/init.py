#!/usr/bin/env python3
"""Run one Vierling-Claassen et al. (2008) simulation and emit the modeled MEG signal + spectrum.

    python src/init.py                       # control, 40 Hz
    VC_CONDITION=schizophrenia VC_DRIVE=20 python src/init.py

Writes <saveFolder>/<simLabel>.npz  with: t, meg (averaged EPSC onto pyramidal cells, the paper's
MEG proxy), freqs, power (FFT of the final 4096 points, as in METHODS), spike times/ids, and the
full parameter provenance. This .npz is the oracle record consumed by tests/ and by hnn-jax.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mechanisms():
    """Load the compiled hhvc mechanism. NEURON only auto-loads ./<arch>/ from the CWD, so point it
    at mod/<arch>/libnrnmech.so explicitly (build with: cd mod && nrnivmodl .)."""
    from neuron import h
    import glob
    for so in glob.glob(os.path.join(REPO, "mod", "*", "libnrnmech.so")):
        h.nrn_load_dll(so)
        return so
    raise RuntimeError("hhvc not compiled — run:  cd mod && nrnivmodl .")


_MECH_SO = load_mechanisms()

from netpyne import sim  # noqa: E402
from cfg import cfg  # noqa: E402
import netParams as _np_mod  # noqa: E402

netParams = _np_mod.netParams


def setup_meg_recording(sim):
    """Record every excitatory (AMPA) synaptic current onto pyramidal cells — the paper's MEG proxy.

    "we model the MEG signal by averaging EPSCs received by pyramidal cells in the network".
    Iterates the section-level synMechs so it captures BOTH network connections and the
    background/drive stim synapses (which do not appear in cell.conns).
    """
    from neuron import h
    recs = []
    for cell in sim.net.cells:
        if cell.tags.get("pop") != "PYR":
            continue
        for _sec_name, sec in getattr(cell, "secs", {}).items():
            for sm in sec.get("synMechs", []) or []:
                if sm.get("label") == "AMPA" and sm.get("hObj") is not None:
                    v = h.Vector()
                    try:
                        v.record(sm["hObj"]._ref_i, sim.cfg.recordStep)
                        recs.append(v)
                    except Exception:
                        pass
    return recs


def main():
    sim.initialize(simConfig=cfg, netParams=netParams)
    sim.net.createPops()
    sim.net.createCells()
    sim.net.connectCells()
    sim.net.addStims()
    sim.setupRecording()

    meg_recs = setup_meg_recording(sim)

    sim.runSim()
    sim.gatherData()

    t = np.array(sim.allSimData["t"])
    if meg_recs:
        arr = np.array([np.array(v.to_python()) for v in meg_recs])
        n = min(len(t), arr.shape[1])
        meg = -arr[:, :n].mean(axis=0) * 100.0   # rescale x100 as in METHODS; sign -> EPSC positive
        t = t[:n]
    else:
        meg = np.zeros_like(t)

    # ---- spectrum: final 4096 points of the trial (METHODS) ----
    seg = meg[-4096:] if len(meg) >= 4096 else meg
    seg = seg - seg.mean()
    win = np.hanning(len(seg))
    fs = 1000.0 / cfg.dt
    spec = np.abs(np.fft.rfft(seg * win)) ** 2 / len(seg)
    freqs = np.fft.rfftfreq(len(seg), d=cfg.dt / 1000.0)

    spk_t = np.array(sim.allSimData.get("spkt", []))
    spk_id = np.array(sim.allSimData.get("spkid", []))

    os.makedirs(cfg.saveFolder, exist_ok=True)
    out = os.path.join(cfg.saveFolder, f"{cfg.simLabel}.npz")
    prov = {k: getattr(cfg, k) for k in
            ("condition", "driveRate", "subject", "trial", "duration", "dt", "nPyr", "nBask",
             "nChand", "gnabar", "gkbar", "an_scale", "tau2Exc", "tau2Bask", "tau2Chand",
             "gmaxExc", "gmaxInh", "bkgRate")}
    np.savez_compressed(out, t=t, meg=meg, freqs=freqs, power=spec,
                        spkt=spk_t, spkid=spk_id, provenance=json.dumps(prov))

    band = lambda f0: float(spec[(freqs > f0 - 2) & (freqs < f0 + 2)].max()) if len(spec) else 0.0
    print(json.dumps({
        "schema": "vc2008-netpyne/run/v1", "label": cfg.simLabel,
        "condition": cfg.condition, "drive_hz": cfg.driveRate,
        "n_spikes": int(len(spk_t)),
        "power_20Hz": band(20.0), "power_30Hz": band(30.0), "power_40Hz": band(40.0),
        "out": out, **{f"prov_{k}": v for k, v in prov.items() if k in ("tau2Chand", "an_scale")},
    }, indent=2))


if __name__ == "__main__":
    main()
