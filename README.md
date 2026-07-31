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

## Reproduced findings

Single trial per cell of the 2 × 3 design (spectral power of the modeled MEG signal):

| condition | drive | P(20 Hz) | P(30 Hz) | P(40 Hz) |
|---|---|---|---|---|
| control | 20 Hz | 59.7 | 0.4 | 10.4 |
| control | 30 Hz | 0.2 | 113.5 | 0.8 |
| control | **40 Hz** | 0.01 | 0.2 | **93.6** |
| schizophrenia | 20 Hz | **101.9** | 0.4 | 12.6 |
| schizophrenia | 30 Hz | 0.01 | 83.4 | 0.1 |
| schizophrenia | **40 Hz** | 0.3 | 0.7 | **8.5** |

Matching the paper (and the MEG data it models):
- **Control entrains strongly at 40 Hz** to 40 Hz drive.
- **Schizophrenia shows an ~11× reduction in 40 Hz power** — the model's central claim
  ("less power at 40 Hz in SZ compared with NC").
- **Schizophrenia shows increased 20 Hz power** to 20 Hz drive (59.7 → 101.9).
- **Both configurations entrain at 30 Hz**, roughly equivalently.
- A 40 Hz component is present in the control response to 20 Hz drive.

Still to add for the full mixed-mode 20/40 Hz result: per-cell IPSC-decay jitter and the
10-subject × 10-trial grand average (see `DEVIATIONS.md` §5).

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
