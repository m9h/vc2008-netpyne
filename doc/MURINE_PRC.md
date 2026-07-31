# Murine variant: phase-response of the ASSR to optogenetic PV drive

A **held-out causal prediction** from the VC2008 circuit, tested against the mouse BF-PV
optogenetics dataset ([*Sci Data* 2020, s41597-020-00621-z](https://www.nature.com/articles/s41597-020-00621-z)).
**Nothing here is fitted to that dataset** — the model is simply run with an extra phase-shifted
drive onto its interneuron populations and asked what happens to the 40 Hz ASSR.

## The experiment being modeled

- Dataset 2 of the mouse study: 40 Hz click-train ASSR **plus** 1 ms optogenetic pulses driving
  basal-forebrain PV neurons at phase delays **0 / 6.25 / 12.5 / 18.75 ms** relative to sound onset
  (= 0° / 90° / 180° / 270° of the 25 ms cycle).
- Reported effect: **in-phase / advanced stimulation enhances** the (frontal) ASSR;
  **delayed / out-of-phase reduces** it.
- BF-PV neurons are **GABAergic** and preferentially target **cortical fast-spiking interneurons**,
  so they are modeled here as an inhibitory input onto `BASK` + `CHAND` (a disinhibition motif).
  `VC_OPTO_SIGN=excitatory` tests the alternative.

## Results (8 phases × 3 trials, control subject, `src/prc.py`)

| phase (ms) | deg | control, inhibitory | control, excitatory | **schizophrenia, inhibitory** |
|---|---|---|---|---|
| baseline (no opto) | — | 89.75 | 89.75 | **13.05** |
| 0.000 | 0 | 0.80× | 0.66× | 1.18× |
| 3.125 | 45 | 0.70× | 0.75× | 1.20× |
| 6.250 | 90 | **0.69×** | 0.71× | 0.96× |
| 9.375 | 135 | 0.88× | 0.76× | **0.94×** |
| 12.500 | 180 | 0.93× | **0.90×** | 1.06× |
| 15.625 | 225 | 0.92× | 0.72× | 1.24× |
| 18.750 | 270 | **1.00×** | **0.47×** | **1.40×** |
| 21.875 | 315 | 0.88× | **0.44×** | 1.24× |

## What the model gets right

**Robust phase-dependent modulation of the ASSR.** A ~31% swing in 40 Hz power purely as a function
of the *timing* of an identical PV input (control, inhibitory). The phenomenon the mouse experiment
reports — that PV stimulation's effect depends on its phase relative to the click train — is
reproduced by this circuit without any fitting. That is not trivial: it requires the inhibitory
conductance to interact with the drive *within* the cycle.

## What it gets wrong (stated plainly)

**The phase alignment does not match.** The model suppresses near 90° and peaks near 270°; the paper
reports enhancement in-phase (0°) and suppression out-of-phase (180°) — roughly a quarter-cycle
offset. This holds for **both** sign hypotheses, so it is not explained by the GABAergic-vs-
glutamatergic modeling choice:

| hypothesis | max | min |
|---|---|---|
| inhibitory (biologically accurate) | 270° | 90° |
| excitatory (alternative) | 180° | 315° |

## The ceiling effect — and the prediction it produces

In the **control** network the opto drive can *only* suppress (best case 1.00×, never above
baseline). The reason is in the paper's own words: with τ_inh = 8 ms "inhibition decays sufficiently
between pulses to permit **all** E cells to respond to every drive input." The control network at
40 Hz is **saturated**, so there is no headroom — any perturbation can only degrade entrainment.

Re-running with the **schizophrenia** circuit as the baseline (40 Hz power degraded to 13.05) gives
genuine enhancement: **1.40× at 270°**, i.e. optimally-timed PV drive **partially restores** the lost
gamma entrainment, and the effect remains strongly phase-dependent (1.40× vs 0.94× — a 49% spread
between best and worst timing).

**Model prediction:** phase-locked interneuron stimulation should enhance the ASSR only where the
baseline response is *not* already saturated; in an impaired (long-τ_inh) circuit there is headroom
and a specific optimal phase exists. In a healthy, fully-entrained circuit the same stimulation can
only disrupt. This is directly testable — and it is the therapeutic form of the question the
JoVE/HNN framing is aimed at.

## Caveats

- **Single cortical column.** The mouse study's frontal-vs-parietal dissociation ("compensatory
  emergence of parietal responses") is *structurally* outside this model; ≥2 coupled columns would
  be required. No attempt is made to fit it.
- **No thermal rescaling.** Mouse recordings are at ~37 °C; the HH rate functions here are the
  classic 6.3 °C forms with no q10 correction. Only the GABA_A decay constants are mouse-adjusted
  (`VC_SPECIES=mouse`). This is the most likely contributor to the phase offset, since the PRC's
  shape depends on the ratio of synaptic/membrane time constants to the 25 ms cycle.
- 3 trials/phase, one connectivity seed; a single 40 Hz drive frequency.
- The mouse dataset has n = 6 male B6 PV-Cre mice.

## Reproduce

```bash
python src/prc.py --trials 3                                   # control, inhibitory
python src/prc.py --trials 3 --sign excitatory                 # alternative hypothesis
python src/prc.py --trials 3 --condition schizophrenia         # the restoration result
```
