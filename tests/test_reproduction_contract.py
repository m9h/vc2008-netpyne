"""Contract tests: the model's structure, its calibration, and the paper's key findings.

Structural/calibration tests run fast. The findings tests consume the .npz outputs of the 2x3
condition x drive grid, so generate them first:

    for c in control schizophrenia; do for d in 20 30 40; do
      VC_CONDITION=$c VC_DRIVE=$d VC_OUT=output python src/init.py; done; done
    pytest tests -q
"""
import glob
import json
import os
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("VC_OUT", REPO / "output"))


def _load(cond, drive):
    hits = sorted(glob.glob(str(OUT / f"vc2008_{cond}_{drive}Hz_*.npz")))
    if not hits:
        pytest.skip(f"no run for {cond} @ {drive}Hz — generate the 2x3 grid first")
    return np.load(hits[0], allow_pickle=True)


def _band(z, f0, half=2.0):
    f, p = z["freqs"], z["power"]
    m = (f > f0 - half) & (f < f0 + half)
    return float(p[m].max()) if m.any() else 0.0


# ---------------------------------------------------------------- structure & calibration
def test_mechanism_is_compiled():
    assert glob.glob(str(REPO / "mod" / "*" / "libnrnmech.so")), \
        "hhvc not compiled — run: cd mod && nrnivmodl ."


def test_network_composition():
    """160 pyramidal + 40 basket + 40 chandelier (METHODS)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from cfg import cfg
    assert (cfg.nPyr, cfg.nBask, cfg.nChand) == (160, 40, 40)


def test_condition_only_changes_chandelier_decay():
    """The single manipulation: tau2_ch 8 ms (control) -> 25 ms (schizophrenia)."""
    import importlib
    import sys
    sys.path.insert(0, str(REPO / "src"))
    os.environ["VC_CONDITION"] = "control"
    import cfg as cfg_mod
    ctrl = importlib.reload(cfg_mod).cfg
    os.environ["VC_CONDITION"] = "schizophrenia"
    sz = importlib.reload(cfg_mod).cfg
    assert ctrl.tau2Chand == 8.0 and sz.tau2Chand == 25.0
    assert ctrl.tau2Bask == sz.tau2Bask     # basket decay is unchanged between conditions
    os.environ["VC_CONDITION"] = "control"


# ---------------------------------------------------------------- the paper's findings
def test_control_entrains_at_40hz():
    z = _load("control", 40)
    assert _band(z, 40) > 10 * _band(z, 20), "control should entrain strongly at 40 Hz"


def test_schizophrenia_reduces_40hz_power():
    """Central claim: prolonged chandelier IPSC decay reduces the 40 Hz response."""
    c, s = _load("control", 40), _load("schizophrenia", 40)
    assert _band(s, 40) < 0.5 * _band(c, 40), \
        f"expected reduced 40 Hz power in SZ (control {_band(c,40):.1f} vs SZ {_band(s,40):.1f})"


def test_schizophrenia_increases_20hz_response():
    c, s = _load("control", 20), _load("schizophrenia", 20)
    assert _band(s, 20) > _band(c, 20), "SZ should show a larger 20 Hz response to 20 Hz drive"


def test_both_conditions_entrain_at_30hz():
    for cond in ("control", "schizophrenia"):
        z = _load(cond, 30)
        assert _band(z, 30) > 5 * max(_band(z, 20), _band(z, 40)), \
            f"{cond} should entrain at 30 Hz"


def test_provenance_recorded():
    z = _load("control", 40)
    prov = json.loads(str(z["provenance"]))
    for k in ("condition", "driveRate", "tau2Chand", "an_scale", "gnabar", "gkbar"):
        assert k in prov, f"missing provenance field {k}"
