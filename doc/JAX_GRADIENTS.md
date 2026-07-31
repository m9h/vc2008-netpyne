# Differentiable model: continuous phase-response and gradient-based analysis

`jaxmodel/` implements the paper's **own simplified theta-neuron network** (20 E + 10 I, all-to-all)
in JAX/Diffrax. Theta neurons are continuous — a "spike" is the phase crossing π, not a discontinuous
reset — so the network is differentiable end-to-end **without surrogate gradients**. The drive and the
optogenetic input are written as explicit functions of frequency and phase, so `d(ASSR power)/d(phase)`
and `d(ASSR power)/d(τ_inh)` come straight from autodiff.

The paper contains both models; the GENESIS network is reproduced in NetPyNE elsewhere in this repo,
and the authors validated the simplified model against it. That makes this a legitimate differentiable
substrate rather than a new approximation.

## Headline: the model can reproduce the mouse BF-PV phase-response — but this is a consistency
## demonstration, not a prediction

**Read the "How strong is this claim" section below before citing this.** An earlier version of this
document called the match a "held-out causal prediction." That was an over-claim and is retracted.

Continuous PRC, 26 phases across one 40 Hz cycle, `g_opto = 1.0`, control (τ_inh = 8 ms).
Baseline (no opto) = 9474. **Nothing is fitted to the mouse dataset.**

| phase (ms) | deg | P40 / baseline | |
|---|---|---|---|
| **0.0** | 0 | **1.11×** | ← paper sampled — *enhanced* |
| **3.0** | 43 | **1.18×** | **true optimum (not sampled by the experiment)** |
| 6.25 | 90 | 1.06× | ← paper sampled |
| 10.0 | 144 | 0.96× | |
| **12.5** | 180 | **0.84×** | ← paper sampled — *suppressed* |
| **15.0** | 216 | **0.65×** | **true trough (not sampled by the experiment)** |
| **18.75** | 270 | **0.74×** | ← paper sampled — *suppressed* |
| 24.0 | 346 | 1.05× | |

Reported effect in the mouse study: *"Advanced and in-phase stimulation enhanced frontal ASSRs, while
delayed and out-of-phase stimulation reduced them."* The model gives **enhancement in-phase (1.11×)
and suppression at the delayed phases (0.84×, 0.74×)** — the same pattern, from a circuit that has
never seen those data. Full swing best/worst = **1.82×**.

Notably the **NetPyNE/HH version of this same circuit did *not* reproduce the alignment**
(see `MURINE_PRC.md`: it peaked at 270°). The theta model does. That the paper's own two models
disagree here is itself a finding worth flagging.

## How strong is this claim, precisely

**Robust and genuinely unfitted:** the circuit produces *phase-dependent* ASSR modulation — the
effect of an identical PV input depends on when in the drive cycle it arrives. This held in both
models and at every parameter setting tried, with nothing fitted to the mouse data.

**Contingent, and therefore NOT a prediction:** the specific alignment (enhancement in-phase,
suppression delayed). Obtaining it required (i) calibrating `g_ie` to 0.08 because the published
value is ambiguous, (ii) sweeping `g_opto` and selecting 1.0 after weaker values produced almost no
effect, and (iii) reporting the theta model, which matched, while the NetPyNE/HH implementation of
the *same* circuit did not (`MURINE_PRC.md`). Model selection and parameter selection both entered.

Accurate summary: **the model can reproduce the observed pattern in a plausible parameter regime;
it does not uniquely predict it.**

## The HH/theta disagreement is informative, not embarrassing

The two implementations of this paper's own circuit give *opposite* phase alignments. That is
expected once stated properly: **theta neurons are Type I excitable** (SNIC bifurcation, strictly
positive phase-response curve) while **Hodgkin-Huxley is Type II** (Hopf bifurcation, biphasic PRC).
A network's phase-response to timed inhibition inherits the single-neuron PRC type.

So the sign of the mouse experiment's phase effect is informative about **excitability class in
auditory cortex**, not only about inhibition — and the two models here bracket the two classes.
That is a sharper and more falsifiable use of the dataset than asking whether "the circuit
reproduces the effect."

## Untested alternative for the optogenetic target

BF-PV is GABAergic and targets cortical fast-spiking interneurons, so it is modeled as inhibition
onto the interneuron population (disinhibition of pyramidal cells). The `opto_sign=excitatory`
switch is **not** a meaningful alternative hypothesis — an excitatory synapse from a GABAergic
projection neuron is wrong on its face. The alternative that matters and has **not** been run is
**inhibition onto the pyramidal population** (direct inhibition rather than disinhibition), which
could plausibly flip the phase alignment. Equal targeting of basket and chandelier cells is also an
untested assumption.

## What differentiability actually bought

**1. The experiment undersampled its own effect.** The true optimum (3.0 ms) and trough (15.0 ms)
both fall *between* the four phases the study sampled (0 / 6.25 / 12.5 / 18.75 ms). A continuous PRC
locates them exactly; discrete sampling cannot. This is a concrete, testable suggestion for the next
recording session — and exactly the "optimal experimental design" use of gradients.

**2. Exact extrema via zero-crossings.** `dP40/dφ` changes sign at the extrema, so they are located
analytically rather than by grid refinement.

**3. Stimulation intensity is a hard requirement, and the model quantifies it.** Sweeping `g_opto`
(vmapped, 45 evaluations in 4.1 s):

| g_opto | 0.0 | 0.06 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| phase swing (max/min) | 1.00× | 1.02× | 1.12× | 1.36× | **1.88×** |

Below ~0.2 the phase effect essentially vanishes. **Prediction: the phase-dependence should disappear
at low optogenetic power** — directly testable, and it explains why my first pass (`g_opto = 0.06`)
found almost no effect.

**4. Where the ASSR is most informative about the disease parameter.** `dP40/dτ_inh` is small at the
endpoints the paper uses (−172 at τ=8, +746 at τ=28) but **an order of magnitude larger near
τ_inh ≈ 12 ms (−4026)**. The ASSR carries the most information about inhibitory decay in a regime the
study never probes — a gradient-derived experimental-design result, not obtainable from a two-point
comparison.

**5. Speed.** A 2-D (phase × τ_inh) landscape — 52 simulations — takes **4.3 s** via `vmap` on CPU.
The equivalent NetPyNE sweep (`src/prc.py`) is ~70 NEURON simulations and minutes on 14 cores.

## Honest limitations

- **`g_ie` is calibrated, not read off the paper.** The source text lists the synaptic weights as
  "g_ee = 0.015, g_ei = 0.025, g_ee = 0.015, g_ii = 0.02, …" — `g_ee` appears twice, so the I→E weight
  is ambiguous. It is set to **0.08**, chosen so inhibition can gate the drive at all. With the literal
  0.015 the drive overwhelms inhibition and no condition effect survives.
- **The schizophrenia contrast is weak here** (~1.6× at 40 Hz vs the NetPyNE model's 9.7×). The
  theta model as parameterized does not reproduce the τ_inh = 8 → 28 ms effect at full strength;
  this is open and is the main reason the model is not yet a drop-in replacement for the HH network.
- **A few gradient evaluations are numerically noisy** (e.g. a spike at φ = 22 ms), from the fixed-step
  solver interacting with the drive-onset step function. The PRC values themselves are smooth; the
  derivative should be treated as indicative at isolated points.
- Single noise realization; one drive frequency (40 Hz); control condition for the headline PRC.

## Reproduce

```bash
pip install jax diffrax
PYTHONPATH=. python jaxmodel/gradients.py     # PRC + dP/dphase, dP/dtau_inh, 2-D landscape
```
