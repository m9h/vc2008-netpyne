"""Differentiable JAX implementation of the Vierling-Claassen et al. (2008) *simplified* model.

The paper contains two models. The GENESIS network is reproduced in NetPyNE elsewhere in this repo;
this module implements the paper's own **simplified theta-neuron network** (20 excitatory +
10 inhibitory, all-to-all), which the authors used for mechanistic analysis and validated against
the GENESIS model.

Why this one for JAX: theta neurons are *continuous* — the "spike" is the phase passing pi, not a
discontinuous reset — so the whole network is differentiable end-to-end with **no surrogate
gradients**. Combined with a drive written as an explicit function of frequency and phase, this
makes d(ASSR power)/d(phase) and d(ASSR power)/d(tau_inh) directly available by autodiff.

Equations (METHODS, p. 2660), time in ms:

    dtheta_k/dt = 1 - cos(theta_k) + (b_k + S_k + N_k(t)) (1 + cos(theta_k))
    S_k         = sum_j alpha_j g_jk s_j            alpha = +1 excitatory, -1 inhibitory
    ds_j/dt     = -s_j/tau_j + exp(-eta (1 + cos theta_j)) (1 - s_j)/tau_R

with eta = 5, tau_R = 0.1, tau_exc = 2, tau_inh = 8 (control) / 28 (schizophrenia),
b_k = -0.01, and g_ee = 0.015, g_ei = 0.025, g_ie = 0.015, g_ii = 0.02, g_de = 0.3, g_di = 0.08.

The drive is "a single excitatory pacemaker cell firing at the modeled click train frequency"; it is
represented here by a phase-locked oscillator theta_d(t) = 2*pi*f*(t - t0), whose release variable
uses the same exp(-eta(1+cos)) form. An optional optogenetic drive (the murine BF-PV experiment)
is an identical oscillator shifted by `phase_ms` and targeting the inhibitory population.
"""
from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp

N_E, N_I = 20, 10
N = N_E + N_I
ETA, TAU_R = 5.0, 0.1
B_K = -0.01
TAU_EXC = 2.0
G = dict(ee=0.015, ei=0.025, ie=0.015, ii=0.02, de=0.3, di=0.08)

IS_EXC = jnp.concatenate([jnp.ones(N_E), jnp.zeros(N_I)])       # 1 = excitatory
ALPHA = jnp.where(IS_EXC > 0, 1.0, -1.0)                        # sign of each presynaptic cell


def _weight_matrix(g=G):
    """W[j, k] = magnitude of the synapse from presynaptic j onto postsynaptic k (all-to-all)."""
    W = jnp.zeros((N, N))
    W = W.at[:N_E, :N_E].set(g["ee"])
    W = W.at[:N_E, N_E:].set(g["ei"])
    W = W.at[N_E:, :N_E].set(g["ie"])
    W = W.at[N_E:, N_E:].set(g["ii"])
    return W


W_DEFAULT = _weight_matrix()
G_DRIVE = jnp.concatenate([jnp.full(N_E, G["de"]), jnp.full(N_I, G["di"])])


def _release(theta):
    """Smooth transmitter-release pulse, peaking as the phase crosses pi (the theta-model 'spike')."""
    return jnp.exp(-ETA * (1.0 + jnp.cos(theta)))


def _noise_kernel(t, spike_times, amp=0.5, tau1=0.1, tau2=2.0):
    """Sum of dual-exponential EPSCs at FIXED pre-drawn times (reparameterized -> differentiable).

    spike_times: (N, K) array, padded with jnp.inf for unused slots.
    """
    dt = t - spike_times
    k = jnp.where(dt > 0, jnp.exp(-dt / tau2) - jnp.exp(-dt / tau1), 0.0)
    return amp * jnp.sum(k, axis=-1) / (tau2 - tau1)


def make_noise(key, rate_hz=33.0, duration=500.0, max_events=48):
    """Pre-draw Poisson EPSC times per cell (fixed across a gradient computation)."""
    k1, _ = jax.random.split(key)
    mean_isi = 1000.0 / rate_hz
    isis = jax.random.exponential(k1, (N, max_events)) * mean_isi
    times = jnp.cumsum(isis, axis=1)
    return jnp.where(times < duration, times, jnp.inf)


def _field(t, y, args):
    """State = [theta (N), s (N), s_drive, s_opto].

    The pacemaker and optogenetic inputs are *synapses*, so their gating variables obey the same
    ds/dt law as network synapses (decay tau_exc / tau_opto) rather than being raw release pulses.
    """
    theta, s = y[:N], y[N:2 * N]
    s_d, s_o = y[2 * N], y[2 * N + 1]
    p = args
    S = (ALPHA * s) @ p["W"]                      # recurrent input
    S = S + p["g_drive"] * s_d                    # click-train drive (E-weighted)
    S = S + p["opto_sign"] * p["g_opto"] * p["opto_target"] * s_o
    Nk = _noise_kernel(t, p["noise_times"])       # fixed-realization background
    drive_term = B_K + S + Nk
    dtheta = (1.0 - jnp.cos(theta)) + drive_term * (1.0 + jnp.cos(theta))
    ds = -s / p["tau_syn"] + _release(theta) * (1.0 - s) / TAU_R
    # pacemaker phases (explicit functions of t -> differentiable wrt f and phase_ms)
    on = (t > p["t0"]).astype(jnp.float32)
    th_d = 2.0 * jnp.pi * p["f"] * (t - p["t0"]) / 1000.0
    th_o = 2.0 * jnp.pi * p["f"] * (t - p["t0"] - p["phase_ms"]) / 1000.0
    ds_d = -s_d / TAU_EXC + on * _release(th_d) * (1.0 - s_d) / TAU_R
    ds_o = -s_o / p["tau_opto"] + on * _release(th_o) * (1.0 - s_o) / TAU_R
    return jnp.concatenate([dtheta, ds, jnp.array([ds_d]), jnp.array([ds_o])])


def simulate(tau_inh=8.0, f=40.0, phase_ms=0.0, g_opto=0.0, opto_sign=-1.0,
             noise_times=None, duration=500.0, t0=50.0, dt=0.05, save_dt=0.5, W=None,
             tau_opto=5.0):
    """Run the network. Returns (ts, meg) where meg = mean excitatory input onto E cells."""
    if noise_times is None:
        noise_times = jnp.full((N, 1), jnp.inf)
    tau_syn = jnp.concatenate([jnp.full(N_E, TAU_EXC), jnp.full(N_I, tau_inh)])
    opto_target = jnp.concatenate([jnp.zeros(N_E), jnp.ones(N_I)])   # BF-PV -> interneurons
    args = dict(W=W_DEFAULT if W is None else W, tau_syn=tau_syn, f=f, t0=t0,
                g_drive=G_DRIVE, phase_ms=phase_ms, g_opto=g_opto, opto_sign=opto_sign,
                opto_target=opto_target, tau_opto=tau_opto, noise_times=noise_times)
    y0 = jnp.concatenate([jnp.full(N, -0.2), jnp.zeros(N), jnp.zeros(2)])
    ts = jnp.arange(0.0, duration, save_dt)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(_field), dfx.Tsit5(), t0=0.0, t1=duration, dt0=dt, y0=y0, args=args,
        saveat=dfx.SaveAt(ts=ts), max_steps=1_000_000,
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.RecursiveCheckpointAdjoint())
    s = sol.ys[:, N:2 * N]
    s_d = sol.ys[:, 2 * N]
    # "average EPSCs received by excitatory cells" -> excitatory synaptic input onto E cells
    meg = G["ee"] * jnp.sum(s[:, :N_E], axis=1) + G["de"] * s_d
    return ts, meg


def assr_power(freq_hz, tau_inh=8.0, f=40.0, phase_ms=0.0, g_opto=0.0, opto_sign=-1.0,
               noise_times=None, duration=500.0, t0=50.0, save_dt=0.5, tau_opto=5.0,
               W=None):
    """Differentiable spectral power of the modeled MEG at `freq_hz` (Goertzel-style projection)."""
    ts, meg = simulate(tau_inh=tau_inh, f=f, phase_ms=phase_ms, g_opto=g_opto,
                       opto_sign=opto_sign, noise_times=noise_times,
                       duration=duration, t0=t0, save_dt=save_dt, tau_opto=tau_opto, W=W)
    m = ts > t0 + 50.0                       # discard onset transient
    x = jnp.where(m, meg - jnp.sum(jnp.where(m, meg, 0.0)) / jnp.sum(m), 0.0)
    w = 2.0 * jnp.pi * freq_hz * ts / 1000.0
    re = jnp.sum(x * jnp.cos(w))
    im = jnp.sum(x * jnp.sin(w))
    return (re ** 2 + im ** 2) / jnp.sum(m) ** 2 * 1e6
