# Deviations from the published description

This re-implementation follows Vierling-Claassen et al. (2008) *J Neurophysiol* 99:2656–2671 as
literally as possible. Where the published description is ambiguous, internally inconsistent, or
insufficient to reproduce the reported behaviour, the choice made here is documented below with the
evidence for it. **Every deviation is a single named parameter in `src/cfg.py`, so the printed value
can always be restored** (e.g. `VC_AN_SCALE=0.1`).

The original is a GENESIS model; this is NEURON/NetPyNE. Numerical results are therefore not expected
to be bit-identical — the target is reproduction of the *reported phenomena*.

---

## 1. The printed α_n prefactor (0.1) cannot produce action potentials — canonical HH (0.01) used

**Printed (METHODS):** `α_n(V) = 0.1 (10 − V) / (exp((10 − V)/10) − 1)`
**Canonical Hodgkin–Huxley (1952):** the prefactor is **0.01**.

**Evidence this is a typo:** with 0.1, `n_inf(rest) = 0.823` instead of the canonical `0.318`, giving
a resting K⁺ conductance ~180× the leak. The membrane is clamped near `E_K`. Measured in an isolated
model cell (`mod/hhvc.mod`, 0.05 nA for 200 ms):

| α_n prefactor | V_rest | spikes |
|---|---|---|
| 0.1 (as printed) | −80.97 mV | **0** |
| 0.01 (canonical) | −59.35 mV | 12 |

No choice of the other parameters rescues the printed value; the cell cannot spike. **Default:
`an_scale = 0.01`.** Set `VC_AN_SCALE=0.1` to reproduce the printed (non-spiking) behaviour.

## 2. Maximal channel conductances — Table 1 units are ambiguous; calibrated to the stated V_rest

**Printed (Table 1):** `ḡ_Na = 80 pS`, `ḡ_K = 40 pS`.

With the printed geometry (soma 30 × 30 µm ⇒ 2.83 × 10⁻⁵ cm²) the leak conductance alone is
≈ 2.8 nS, i.e. **35× larger than a "80 pS" sodium conductance** — such a cell cannot spike. The
values are therefore read as densities (S/cm²), the usual convention for `ḡ` in HH formulations.

`ḡ_K` is then set analytically so that the cell rests at the paper's own leak potential
`E_m = −59.4 mV`: requiring zero net current at −59.4 mV gives `ḡ_K ≈ 0.0402 · ḡ_Na`. Confirmed
numerically:

| ḡ_K (S/cm²) | V_rest |
|---|---|
| **0.0048 (default)** | **−59.35 mV**  ← matches the paper's E_m |
| 0.006 | −62.24 mV |
| 0.012 | −66.17 mV |

**Defaults: `gnabar = 0.12 S/cm²` (canonical HH), `gkbar = 0.0048 S/cm²`.**

## 3. Synaptic conductance scale — calibrated to the paper's stated firing behaviour

**Printed (Table 2):** `g_max,exc = 80 pS`, `g_max,inh = 40 pS`.

With the printed geometry a single 80 pS event produces a ≈ 2.4 mV EPSP, which contradicts the
paper's own statements that "the EPSCs are strong enough to trigger firing in a resting cell" and
that the undriven network fires at **29 Hz**. A single multiplicative factor `synScale` is applied
to all synaptic weights and calibrated against that stated 29 Hz target (`src/calibrate.py`).
**Default: `synScale = 12`.** Set `VC_SYNSCALE=1` for the literal printed conductances.

## 4. Synaptic rise times — printed rise = decay = 3 ms is degenerate

Table 2 lists both `τ1,exc` (rise) and `τ2,exc` (decay) as **3 ms**. The dual-exponential form given
in METHODS, `(e^{−t/τ1} − e^{−t/τ2})/(τ1 − τ2)`, is undefined when `τ1 = τ2` (0/0), and NEURON's
`Exp2Syn` requires `τ1 < τ2`. Rise times are set to **0.5 ms** (`tau1Exc`, `tau1Inh`), preserving the
paper's decay constants, which are the parameters the paper's mechanism actually depends on
(`τ2,exc = 3`, `τ2,b = 8`, `τ2,ch = 8 control / 25 schizophrenia`).

## 5. Not yet implemented (roadmap, not silent omissions)

- **Per-cell decay jitter.** The paper draws IPSC decays from uniform distributions
  (`τ2,b = 8 ± 5 ms`; `τ2,ch = 8 ± 5` control / `25 ± 15` schizophrenia). Current code uses the
  distribution *means*. Jitter is needed to fully reproduce the mixed-mode 20/40 Hz response.
- **Simulated subjects/trials.** The paper averages 10 "subjects" (fixed random connectivity) × 10
  trials (distinct noise). `cfg.subject` / `cfg.trial` seed this; the grand-average pipeline is in
  `src/batch.py`.
- **Dendritic active conductances.** The paper does not state whether dendrites carry HH channels;
  `dendActiveFrac = 0.1` (10% of somatic density) is assumed.
- **The simplified theta-neuron model** (20 E + 10 I, MATLAB in the original) is not ported.

## Reproduction status

With the calibration above, a single trial per condition reproduces the paper's central qualitative
results (see README "Reproduced findings"), including the **11× reduction in 40 Hz power** in the
schizophrenia configuration — the model's headline claim.
