"""Simulation configuration for the Vierling-Claassen et al. (2008) NetPyNE model.

Two experimental factors, exactly as in the paper:
  cfg.condition   'control' | 'schizophrenia'   -> chandelier IPSC decay tau2_ch
  cfg.driveRate   20 | 30 | 40 (Hz)             -> click-train frequency
"""
import os

from netpyne import specs

cfg = specs.SimConfig()

# ------------------------------------------------- experiment factors
cfg.condition = os.environ.get("VC_CONDITION", "control")     # control | schizophrenia
cfg.driveRate = float(os.environ.get("VC_DRIVE", 40))         # 20, 30 or 40 Hz
cfg.subject = int(os.environ.get("VC_SUBJECT", 0))            # connectivity seed ("simulated patient")
cfg.trial = int(os.environ.get("VC_TRIAL", 0))                # background-noise seed

# ------------------------------------------------- simulation (METHODS: 600 ms, dt = 0.1 ms)
cfg.duration = 600.0
cfg.dt = 0.1
cfg.hParams = {"celsius": 6.3, "v_init": -59.4}   # HH rates are at the classic 6.3 C
cfg.verbose = False
cfg.printPopAvgRates = True
cfg.spikeThresh = -20.0

# deterministic, separable seeds: connectivity fixed per "subject", noise varies per trial
cfg.seeds = {"conn": 1000 + cfg.subject, "stim": 2000 + 100 * cfg.subject + cfg.trial,
             "loc": 3000 + cfg.subject}

# ------------------------------------------------- network size (METHODS)
cfg.nPyr, cfg.nBask, cfg.nChand = 160, 40, 40

# ------------------------------------------------- channels (see DEVIATIONS.md)
cfg.gnabar = float(os.environ.get("VC_GNA", 0.12))   # S/cm2 (Table 1 value is unit-ambiguous)
cfg.gkbar = float(os.environ.get("VC_GK", 0.0048))   # S/cm2 (calibrated: v_rest = -59.35 ~ E_m)
cfg.an_scale = float(os.environ.get("VC_AN_SCALE", 0.01))  # canonical HH; paper prints 0.1 (unusable, see DEVIATIONS.md)
cfg.dendActiveFrac = float(os.environ.get("VC_DEND_ACTIVE", 0.1))  # dendritic HH density fraction

# ------------------------------------------------- synapses (Table 2)
# synScale: the printed g_max (80 pS) with the printed geometry gives only a ~2.4 mV EPSP, which
# contradicts the paper's own statement that "the EPSCs are strong enough to trigger firing in a
# resting cell" (and its 29 Hz spontaneous rate). A single scale factor on all synaptic weights is
# calibrated to that stated behaviour; see DEVIATIONS.md and src/calibrate.py.
cfg.synScale = float(os.environ.get("VC_SYNSCALE", 12.0))
cfg.gmaxExc = 80e-6 * cfg.synScale    # uS  (80 pS x scale)  g_max,exc = g_max,d
cfg.gmaxInh = 40e-6 * cfg.synScale    # uS  (40 pS x scale)  g_max,b = g_max,ch
cfg.tau1Exc = 0.5        # ms  rise (see DEVIATIONS.md: table prints rise = decay = 3 ms)
cfg.tau2Exc = 3.0        # ms  decay, AMPA
cfg.tau1Inh = 0.5        # ms  rise
cfg.tau2Bask = 8.0       # ms  tau2,b  mean (8 +/- 5 ms, uniform)
cfg.tau2Chand = 8.0 if cfg.condition == "control" else 25.0   # tau2,ch: 8 (control) | 25 (schiz)
# Per-cell IPSC decay jitter, drawn uniformly (METHODS): tau2,b = 8 +/- 5 ms in both conditions;
# tau2,ch = 8 +/- 5 ms (control) | 25 +/- 15 ms (schizophrenia). Each inhibitory cell carries its
# own decay time, so the synapses it makes inherit it.
cfg.useJitter = os.environ.get("VC_JITTER", "1") == "1"
cfg.tau2BaskJitter = 5.0
cfg.tau2ChandJitter = 5.0 if cfg.condition == "control" else 15.0
cfg.synDelay = 1.0       # ms

# ------------------------------------------------- inputs (METHODS)
cfg.bkgRate = 4.0                       # Hz Poisson EPSCs onto every pyramidal cell
cfg.driveStart = 100.0                  # ms
cfg.driveNumber = int(cfg.driveRate * (cfg.duration - cfg.driveStart) / 1000.0)

# ------------------------------------------------- recording
# The paper's "MEG" proxy = average EPSC received by pyramidal cells -> record AMPA i on PYR dend.
cfg.recordTraces = {"V_soma": {"sec": "soma", "loc": 0.5, "var": "v"}}
cfg.recordStim = False
cfg.recordStep = cfg.dt
cfg.recordCells = [("PYR", 0), ("BASK", 0), ("CHAND", 0)]
cfg.recordLFP = None

cfg.simLabel = f"vc2008_{cfg.condition}_{int(cfg.driveRate)}Hz_s{cfg.subject}_t{cfg.trial}"
cfg.saveFolder = os.environ.get("VC_OUT", "output")
cfg.savePickle = False
cfg.saveJson = False
cfg.analysis = {}
