"""NetPyNE network specification for Vierling-Claassen et al. (2008) J Neurophysiol 99:2656-2671.

Faithful re-implementation of the GENESIS auditory-cortex model: 160 pyramidal + 40 basket +
40 chandelier two-compartment (soma/AIS + dendrite) Hodgkin-Huxley cells, dual-exponential
synapses, probabilistic connectivity, Poisson background and a periodic (20/30/40 Hz) drive.

Every value below is traceable to the paper's Tables 1-2 / METHODS; anything inferred is listed
in DEVIATIONS.md. Parameter symbols in comments match the paper's notation.
"""
from netpyne import specs

try:
    from .cfg import cfg
except ImportError:
    from cfg import cfg

netParams = specs.NetParams()

# ---------------------------------------------------------------- cell geometry (Table 1)
# l_s = d_s = 30e-4 cm = 30 um (soma/AIS);  l_d = 100e-4 cm = 100 um, d_d = 2e-4 cm = 2 um (dend)
# C_M = 1 uF/cm2, R_M = 10 kOhm-cm2 -> g_pas = 1/(10e3) S/cm2 = 1e-4 S/cm2; R_A = 0.05 kOhm-cm2 = 50 Ohm-cm
SOMA_L, SOMA_D = 30.0, 30.0
DEND_L, DEND_D = 100.0, 2.0
E_LEAK = -59.4          # E_m  (Table 1)
E_NA, E_K = 45.0, -82.0  # (Table 1)
G_PAS = 1.0 / 10e3       # S/cm2 from R_M = 10 kOhm-cm2
C_M = 1.0                # uF/cm2
R_A = 50.0               # Ohm-cm from R_A = 0.05 kOhm-cm2

netParams.defaultThreshold = cfg.spikeThresh


def _two_compartment(gnabar, gkbar):
    """soma/AIS + dendrite, HH (hhvc) in both, per the paper's single-neuron model."""
    return {
        "secs": {
            "soma": {
                "geom": {"L": SOMA_L, "diam": SOMA_D, "Ra": R_A, "cm": C_M},
                "mechs": {"hhvc": {"gnabar": gnabar, "gkbar": gkbar, "em": E_LEAK,
                                   "an_scale": cfg.an_scale},
                          "pas": {"g": G_PAS, "e": E_LEAK}},
                "ions": {"na": {"e": E_NA}, "k": {"e": E_K}},
            },
            "dend": {
                "geom": {"L": DEND_L, "diam": DEND_D, "Ra": R_A, "cm": C_M},
                "mechs": {"hhvc": {"gnabar": gnabar * cfg.dendActiveFrac,
                                   "gkbar": gkbar * cfg.dendActiveFrac, "em": E_LEAK,
                                   "an_scale": cfg.an_scale},
                          "pas": {"g": G_PAS, "e": E_LEAK}},
                "ions": {"na": {"e": E_NA}, "k": {"e": E_K}},
                "topol": {"parentSec": "soma", "parentX": 1.0, "childX": 0.0},
            },
        }
    }


for _ct in ("PYR", "BASK", "CHAND"):
    netParams.cellParams[_ct] = _two_compartment(cfg.gnabar, cfg.gkbar)

# ---------------------------------------------------------------- populations (METHODS)
netParams.popParams["PYR"] = {"cellType": "PYR", "numCells": cfg.nPyr}       # 160 pyramidal
netParams.popParams["BASK"] = {"cellType": "BASK", "numCells": cfg.nBask}    # 40 basket
netParams.popParams["CHAND"] = {"cellType": "CHAND", "numCells": cfg.nChand}  # 40 chandelier

# ---------------------------------------------------------------- synapses (Table 2)
# G_s(t) = A g_max,s (exp(-t/tau1) - exp(-t/tau2)) / (tau1 - tau2), normalized to peak g_max,s.
# E_exc = E_c = 45 mV ; E_b = E_ch = -82 mV
# g_max,exc = g_max,d = 80 pS ; g_max,b = g_max,ch = 40 pS
# tau2,exc = 3 ms (AMPA) ; tau2,b = 8 +/- 5 ms ; tau2,ch = 8 +/- 5 (control) | 25 +/- 15 (schiz)
netParams.synMechParams["AMPA"] = {"mod": "Exp2Syn", "tau1": cfg.tau1Exc,
                                   "tau2": cfg.tau2Exc, "e": 45.0}
netParams.synMechParams["GABA_B"] = {"mod": "Exp2Syn", "tau1": cfg.tau1Inh,
                                     "tau2": cfg.tau2Bask, "e": -82.0}   # basket (mean)
netParams.synMechParams["GABA_CH"] = {"mod": "Exp2Syn", "tau1": cfg.tau1Inh,
                                      "tau2": cfg.tau2Chand, "e": -82.0}  # chandelier (mean)

# Per-cell decay jitter: each inhibitory cell gets its own tau2 drawn uniformly, and every synapse
# it makes uses that value (METHODS). Realized as one synMech + one connectivity rule per
# presynaptic inhibitory cell. Seeded by "subject" so a simulated patient has a fixed network.
import numpy as _np  # noqa: E402

_rng = _np.random.RandomState(7000 + cfg.subject)
TAU2_BASK = (_rng.uniform(cfg.tau2Bask - cfg.tau2BaskJitter, cfg.tau2Bask + cfg.tau2BaskJitter,
                          cfg.nBask) if cfg.useJitter
             else _np.full(cfg.nBask, cfg.tau2Bask))
TAU2_CHAND = (_rng.uniform(cfg.tau2Chand - cfg.tau2ChandJitter, cfg.tau2Chand + cfg.tau2ChandJitter,
                           cfg.nChand) if cfg.useJitter
              else _np.full(cfg.nChand, cfg.tau2Chand))


G_EXC = cfg.gmaxExc     # uS recurrent excitation
G_DRIVE = cfg.gmaxDrive # uS periodic drive (separately scalable)
G_INH = cfg.gmaxInh     # uS inhibition (separately scalable)

# ---------------------------------------------------------------- connectivity (METHODS p.2659)
# "All synaptic connections are made from the soma/AIS compartment of the presynaptic cell onto
#  the dendritic compartment of the postsynaptic cell unless otherwise indicated."
_C = dict(synsPerConn=1, delay=cfg.synDelay)

# Each pyramidal cell projects to 2% of the [pyramidal] cells in the model
netParams.connParams["PYR->PYR"] = {
    "preConds": {"pop": "PYR"}, "postConds": {"pop": "PYR"},
    "probability": 0.02, "weight": G_EXC, "synMech": "AMPA", "sec": "dend", "loc": 0.5, **_C}

# Pyramidal cells project to 10% of all basket cells and 10% of all chandelier cells
netParams.connParams["PYR->BASK"] = {
    "preConds": {"pop": "PYR"}, "postConds": {"pop": "BASK"},
    "probability": 0.10, "weight": G_EXC, "synMech": "AMPA", "sec": "dend", "loc": 0.5, **_C}
netParams.connParams["PYR->CHAND"] = {
    "preConds": {"pop": "PYR"}, "postConds": {"pop": "CHAND"},
    "probability": 0.10, "weight": G_EXC, "synMech": "AMPA", "sec": "dend", "loc": 0.5, **_C}

# Inhibitory projections. With jitter enabled these are emitted per presynaptic cell so that each
# interneuron's own tau2 is carried by every synapse it makes; otherwise one rule per population.
#   basket -> 80% of other basket, 80% of chandelier
#   basket -> 10% of PYR, "equally to somata and dendrites of a given cell"
#   chandelier -> 10% of PYR, "only to the soma/AIS compartment"
_BASK_TARGETS = [("BASK", 0.80, "dend", G_INH), ("CHAND", 0.80, "dend", G_INH),
                 ("PYR", 0.10, "soma", G_INH / 2.0), ("PYR", 0.10, "dend", G_INH / 2.0)]

for _post, _p, _sec, _w in _BASK_TARGETS:
    netParams.connParams[f"BASK->{_post}_{_sec}"] = {
        "preConds": {"pop": "BASK"}, "postConds": {"pop": _post},
        "probability": _p, "weight": _w, "synMech": "GABA_B", "sec": _sec, "loc": 0.5, **_C}
netParams.connParams["CHAND->PYR"] = {
    "preConds": {"pop": "CHAND"}, "postConds": {"pop": "PYR"},
    "probability": 0.10, "weight": G_INH, "synMech": "GABA_CH", "sec": "soma", "loc": 0.5, **_C}
# NOTE: per-cell decay jitter is applied to the *instantiated* synapses after connectCells()
# (see apply_decay_jitter in src/init.py) using TAU2_BASK / TAU2_CHAND above. NetPyNE creates one
# synapse per NetCon (cfg.oneSynPerNetcon), so each connection's decay can be set from the identity
# of its presynaptic interneuron -- which is exactly the paper's per-cell distribution.

# ---------------------------------------------------------------- background noise (METHODS)
# "all pyramidal cells receiving a Poisson train of EPSCs at an average rate of 4 Hz"
netParams.stimSourceParams["bkg"] = {"type": "NetStim", "rate": cfg.bkgRate,
                                     "noise": 1.0, "start": 0}
netParams.stimTargetParams["bkg->PYR"] = {
    "source": "bkg", "conds": {"pop": "PYR"}, "weight": G_EXC, "synMech": "AMPA",
    "sec": "dend", "loc": 0.5, "delay": cfg.synDelay}

# ---------------------------------------------------------------- periodic drive (METHODS)
# "Input drive was applied via projections from a simulated rhythm generator that sent synapses to
#  all pyramidal cell dendrites and 65% of interneuron dendrites." First pulse is 2x stronger.
netParams.stimSourceParams["drive"] = {
    "type": "NetStim", "rate": cfg.driveRate, "noise": 0.0,
    "start": cfg.driveStart, "number": cfg.driveNumber}

netParams.stimTargetParams["drive->PYR"] = {
    "source": "drive", "conds": {"pop": "PYR"}, "weight": G_DRIVE, "synMech": "AMPA",
    "sec": "dend", "loc": 0.5, "delay": cfg.synDelay}
for _pop in ("BASK", "CHAND"):
    netParams.stimTargetParams[f"drive->{_pop}"] = {
        "source": "drive", "conds": {"pop": _pop, "cellList": list(range(
            int(0.65 * (cfg.nBask if _pop == "BASK" else cfg.nChand))))},
        "weight": G_DRIVE, "synMech": "AMPA", "sec": "dend", "loc": 0.5, "delay": cfg.synDelay}

# ---------------------------------------------------------------- optogenetic BF-PV drive
# 1 ms LED pulses at the click frequency, phase-shifted by cfg.optoPhaseMs relative to the sound
# drive, delivered onto the cortical interneuron populations (BF-PV targets cortical FS cells).
if cfg.optoEnabled:
    netParams.synMechParams["GABA_OPTO"] = {"mod": "Exp2Syn", "tau1": cfg.tau1Inh,
                                            "tau2": cfg.optoTau2, "e": -82.0}
    _opto_mech = "GABA_OPTO" if cfg.optoSign == "inhibitory" else "AMPA"
    _opto_w = (G_INH if cfg.optoSign == "inhibitory" else G_EXC) * cfg.optoWeightScale
    netParams.stimSourceParams["opto"] = {
        "type": "NetStim", "rate": cfg.driveRate, "noise": 0.0,
        "start": cfg.driveStart + cfg.optoPhaseMs, "number": cfg.driveNumber}
    for _pop in ("BASK", "CHAND"):
        netParams.stimTargetParams[f"opto->{_pop}"] = {
            "source": "opto", "conds": {"pop": _pop}, "weight": _opto_w,
            "synMech": _opto_mech, "sec": "dend", "loc": 0.5, "delay": cfg.synDelay}

# stronger first pulse (2x g_max), delivered as a single extra co-timed input
netParams.stimSourceParams["drive_first"] = {
    "type": "NetStim", "rate": cfg.driveRate, "noise": 0.0,
    "start": cfg.driveStart, "number": 1}
netParams.stimTargetParams["drive_first->PYR"] = {
    "source": "drive_first", "conds": {"pop": "PYR"}, "weight": G_DRIVE, "synMech": "AMPA",
    "sec": "dend", "loc": 0.5, "delay": cfg.synDelay}
