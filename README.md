# Vierling-Claassen et al. (2008) auditory-entrainment model — NetPyNE re-implementation

A faithful NEURON/[NetPyNE](https://netpyne.org) re-implementation of the GENESIS model from:

> Vierling-Claassen D, Siekmeier P, Stufflebeam S, Kopell N (2008).
> **Modeling GABA alterations in schizophrenia: a link between impaired inhibition and altered
> gamma and beta range auditory entrainment.** *J Neurophysiol* 99(5):2656–2671.
> doi:[10.1152/jn.00870.2007](https://doi.org/10.1152/jn.00870.2007)

The model shows how a single biophysical change — **prolonged GABA_A decay at chandelier-cell
synapses**, the putative consequence of reduced GAT-1/GAD67 in schizophrenia — shifts auditory
steady-state entrainment from 40 Hz toward 20 Hz, matching MEG click-train data.

## Model

240 two-compartment (soma/AIS + dendrite) Hodgkin–Huxley cells: **160 pyramidal, 40 basket,
40 chandelier**, with the paper's probabilistic connectivity (PC→2%, PC→10% of interneurons,
basket→80%, chandelier→soma/AIS only), dual-exponential synapses, 4 Hz Poisson background, and a
periodic 20/30/40 Hz "click-train" drive. The modeled MEG signal is the average EPSC onto pyramidal
cells (per the paper), analyzed by FFT over the final 4096 samples.

The only difference between conditions is the chandelier IPSC decay:
**τ2,ch = 8 ms (control) → 25 ms (schizophrenia)**.

## Quick start

```bash
python -m venv venv && ./venv/bin/pip install netpyne neuron "numpy<2" scipy
cd mod && ../venv/bin/nrnivmodl . && cd ..          # compile the hhvc mechanism

VC_CONDITION=control       VC_DRIVE=40 python src/init.py
VC_CONDITION=schizophrenia VC_DRIVE=40 python src/init.py
```
Each run writes `output/<label>.npz` with `t`, `meg`, `freqs`, `power`, spike times and full
parameter provenance.

## Reproduction status

Grand average of the paper's full design — **10 subjects × 10 trials × 3 drives × 2 conditions
(600 simulations, 600/600 completed)** (`python src/batch.py`). Two spectra are reported per cohort,
and the distinction turns out to matter (see below):

- **EVOKED** — average the MEG *in time*, then transform (the order METHODS specifies). Retains only
  stimulus-phase-locked activity.
- **TOTAL** — average the per-trial power spectra. Retains phase-locked *and* non-phase-locked activity.

| condition | drive | EVOKED P20 | P30 | P40 | TOTAL P20 | P30 | P40 |
|---|---|---|---|---|---|---|---|
| control | 20 Hz | 59.6 | 0.0 | 6.1 | 59.9 | 0.5 | 6.6 |
| control | 30 Hz | 0.0 | **113.6** | 0.0 | 0.1 | 114.2 | 0.4 |
| control | **40 Hz** | 0.0 | 0.0 | **83.9** | 0.03 | 0.1 | 86.1 |
| schizophrenia | 20 Hz | **93.6** | 0.0 | 9.7 | 93.9 | 0.4 | 9.9 |
| schizophrenia | 30 Hz | 0.0 | **97.1** | 0.0 | 0.1 | 98.1 | 0.2 |
| schizophrenia | **40 Hz** | 0.0 | 0.0 | **8.7** | **0.40** | 0.4 | 8.9 |

### Reproduced ✓

- **Control entrains strongly at 40 Hz** to 40 Hz drive (83.9, no competing components).
- **Schizophrenia reduces 40 Hz power ~9.7×** (83.9 → 8.7) — the central claim, and the MEG finding
  it explains ("less power at 40 Hz in SZ compared with NC", *P* < 0.001).
- **Schizophrenia increases the 20 Hz response** to 20 Hz drive (59.6 → 93.6).
- **Both cohorts entrain at 30 Hz** with comparable power (113.6 vs 97.1).
- **The mixed-mode 20 Hz subharmonic under 40 Hz drive emerges in the schizophrenia network**:
  TOTAL 20 Hz power rises **13×** over control (0.03 → 0.40), alongside the reduced 40 Hz peak.

### The subharmonic is *not phase-locked* — a methodological caveat for anyone reproducing this

The 20 Hz component is produced by pyramidal cells **skipping** drive pulses when the extended
chandelier IPSC has not decayed within the 25 ms inter-pulse interval. **Which** pulse is skipped
(even or odd) is not fixed across trials, so the subharmonic has no consistent phase relative to
stimulus onset. Consequently it **cancels completely in the time-domain grand average** (EVOKED
P20 = 0.00) and is visible only in trial-averaged **TOTAL** power (0.40, 13× control).

Since METHODS specifies the time-domain grand average, this is a real discrepancy in how the
published figure can be obtained; `src/batch.py` therefore reports both. Remaining gap: in the
paper's Fig. 3 the SZ 20 Hz and 40 Hz peaks are of *comparable* magnitude, whereas here the
subharmonic is present and strongly enhanced but still smaller than the residual 40 Hz peak
(0.40 vs 8.9).

**Two mechanistic hypotheses for this gap have been tested and both are refuted:**

1. *Global synaptic gain* (`VC_DRIVESCALE` × `VC_INHSCALE`): raising inhibition suppresses control
   entrainment (P40 79.6 → 0.9) before any subharmonic appears.
2. *Chandelier-specific gain* (`VC_CHANDSCALE`) — the better hypothesis, since the paper's mechanism
   is chandelier-specific (axo-axonic, AIS-targeting) and those synapses sit electrotonically closest
   to spike initiation. It fails too, and in the wrong direction:

   | `chandScale` | control P40 | SZ P40 | SZ P20/P40 |
   |---|---|---|---|
   | 1× | **79.6** | 10.8 | 0.050 |
   | 2× | 27.6 | 19.0 | 0.035 |
   | 4× | 5.0 | 31.2 | **0.021** |

   Control entrainment collapses, SZ 40 Hz *rises*, the subharmonic ratio *falls*, and by 4× the
   control/SZ contrast has inverted. Chandelier gain is not the missing ingredient.

Since two targeted hypotheses have been ruled out, further blind parameter scanning is not
productive. The remaining candidates are structural rather than gain-based — most plausibly a
**separate AIS compartment** (chandelier synapses currently land on a soma that also carries the
somatic HH conductances, so their veto of spike initiation may be under-represented) or the drive's
synaptic kinetics. Both are model changes, not parameter changes, and are left open.

## Murine variant: optogenetic PV phase-response

`VC_SPECIES=mouse` + a phase-shifted optogenetic drive onto the interneuron populations models the
mouse BF-PV ASSR experiment ([*Sci Data* 2020](https://www.nature.com/articles/s41597-020-00621-z)).
Run `python src/prc.py`. As a **held-out causal prediction** (nothing fitted to that dataset) the
circuit reproduces **robust phase-dependent modulation** of the 40 Hz ASSR, but **not** the reported
phase alignment. It also predicts a **ceiling effect**: the saturated control network can only be
degraded by PV stimulation, whereas in the impaired (schizophrenia) circuit optimally-timed
stimulation **restores 40 Hz power to 1.40× baseline**. Full results and caveats:
**[doc/MURINE_PRC.md](doc/MURINE_PRC.md)**.

## Differentiable model (`jaxmodel/`)

The paper's **own simplified theta-neuron network** (20 E + 10 I) in JAX/Diffrax — continuous, so
differentiable end-to-end with no surrogate gradients. It **can reproduce** the mouse BF-PV
phase-response (in-phase enhances 1.11×, delayed suppresses to 0.74×) in a plausible parameter
regime — a consistency demonstration, **not** a prediction (parameter and model selection both
entered; see the doc). The HH implementation of the same circuit gives the *opposite* alignment,
which is itself informative: theta is Type I excitable, HH is Type II, and the network PRC inherits
that. It also locates the optimum (3.0 ms) and trough (15.0 ms) **between** the four phases the
experiment sampled. A 2-D (phase × τ_inh) landscape takes 4.3 s via `vmap`.
Details, gradient-derived experimental-design results, and limitations:
**[doc/JAX_GRADIENTS.md](doc/JAX_GRADIENTS.md)**.

## Honest reproduction notes

The published description is not sufficient, as printed, to produce a spiking network. Three
documented calibrations were required — most notably that **the printed α_n prefactor (0.1) makes
action potentials impossible** (the canonical HH value is 0.01), and that Table 1's channel
conductances must be read as densities. Every deviation, its evidence, and how to revert it is in
**[DEVIATIONS.md](DEVIATIONS.md)**.

## Layout

```
mod/hhvc.mod        HH Na/K with the paper's exact rate functions (reduced-potential convention)
src/netParams.py    network: cells, populations, connectivity, stimuli (Tables 1-2 + METHODS)
src/cfg.py          simulation config; every deviation is a named, overridable parameter
src/init.py         run one simulation -> MEG proxy + spectrum -> .npz
src/calibrate.py    calibrate synScale against the paper's stated 29 Hz undriven firing rate
src/batch.py        10 subjects x 10 trials x 3 drives x 2 conditions (NetPyNE batch / NSG)
tests/              contract tests: structure, resting potential, and the paper's key findings
```

## License / citation

MIT (this implementation). Please cite the original paper for the model, and see `CITATION.cff`
for this re-implementation.
