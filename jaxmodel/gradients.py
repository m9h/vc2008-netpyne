#!/usr/bin/env python3
"""What the differentiable model buys: continuous PRC, sensitivities, and a 2-D landscape.

Three things that are awkward or expensive with the NetPyNE/NEURON model and cheap here:

  1. CONTINUOUS phase-response curve  d(P40)/d(phase)  -- exact, by autodiff, at any phase,
     instead of interpolating between the 4 discrete phases the mouse experiment sampled.
  2. PARAMETER SENSITIVITY  d(P40)/d(tau_inh)  -- how strongly the ASSR reports the disease
     parameter, evaluated pointwise rather than by finite-difference scans.
  3. A vmapped 2-D map over (phase, tau_inh): the whole restoration landscape in one pass.

    python jaxmodel/gradients.py
"""
import json
import os
import time

import jax
import jax.numpy as jnp

from jaxmodel import theta_model as T

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "jax")


def main():
    os.makedirs(OUT, exist_ok=True)
    nt = T.make_noise(jax.random.PRNGKey(0))
    g = dict(T.G); g["ie"] = 0.08                     # calibrated (see doc/JAX_GRADIENTS.md)
    W = T._weight_matrix(g)

    def p40(phase, tau_inh, g_opto=0.06):
        return T.assr_power(40.0, tau_inh=tau_inh, f=40.0, phase_ms=phase,
                            g_opto=g_opto, opto_sign=-1.0, noise_times=nt, W=W)

    # ---------------- 1. continuous PRC + its exact derivative ----------------
    val_and_grad = jax.jit(jax.value_and_grad(p40, argnums=0))
    phases = jnp.linspace(0.0, 25.0, 21)              # a full 40 Hz cycle
    t0 = time.time()
    pv, pg = [], []
    for ph in phases:
        v, gr = val_and_grad(ph, 8.0)
        pv.append(float(v)); pg.append(float(gr))
    print(f"continuous PRC ({len(phases)} points, exact gradients) in {time.time()-t0:.1f}s")
    print(f"\n  {'phase(ms)':>10}{'deg':>6}{'P40':>12}{'dP40/dphase':>14}")
    for ph, v, gr in zip(phases, pv, pg):
        print(f"  {float(ph):>10.2f}{360*float(ph)/25:>6.0f}{v:>12.1f}{gr:>14.2f}")
    imax = int(jnp.argmax(jnp.array(pv))); imin = int(jnp.argmin(jnp.array(pv)))
    print(f"\n  optimum phase  = {float(phases[imax]):.2f} ms ({360*float(phases[imax])/25:.0f} deg)")
    print(f"  worst phase    = {float(phases[imin]):.2f} ms ({360*float(phases[imin])/25:.0f} deg)")
    # zero-crossings of the derivative locate the extrema exactly
    zc = [float((phases[i] + phases[i+1]) / 2) for i in range(len(pg)-1)
          if pg[i] * pg[i+1] < 0]
    print(f"  dP/dphase zero-crossings (extrema): {[round(z,2) for z in zc]} ms")

    # ---------------- 2. sensitivity to the disease parameter ----------------
    d_tau = jax.jit(jax.grad(p40, argnums=1))
    print(f"\n  {'tau_inh':>8}{'P40':>12}{'dP40/dtau_inh':>16}")
    taus = [6.0, 8.0, 12.0, 18.0, 24.0, 28.0]
    sens = []
    for ti in taus:
        v = float(p40(0.0, ti)); s = float(d_tau(0.0, ti)); sens.append((ti, v, s))
        print(f"  {ti:>8.1f}{v:>12.1f}{s:>16.2f}")

    # ---------------- 3. vmapped 2-D landscape (phase x tau_inh) ----------------
    ph_grid = jnp.linspace(0.0, 25.0, 13)
    tau_grid = jnp.array([8.0, 14.0, 20.0, 28.0])
    f2 = jax.jit(jax.vmap(jax.vmap(p40, in_axes=(0, None)), in_axes=(None, 0)))
    t0 = time.time()
    Z = f2(ph_grid, tau_grid)
    print(f"\n2-D landscape {Z.shape} (phase x tau_inh) in {time.time()-t0:.1f}s via vmap")
    print(f"\n  {'tau_inh':>8}" + "".join(f"{float(p):>8.1f}" for p in ph_grid))
    for i, ti in enumerate(tau_grid):
        row = "".join(f"{float(z):>8.0f}" for z in Z[i])
        print(f"  {float(ti):>8.1f}{row}")
    # best phase per tau_inh -> does the optimal stimulation timing move with the disease parameter?
    print(f"\n  {'tau_inh':>8}{'best phase(ms)':>16}{'gain vs worst':>15}")
    for i, ti in enumerate(tau_grid):
        b = int(jnp.argmax(Z[i])); w = int(jnp.argmin(Z[i]))
        print(f"  {float(ti):>8.1f}{float(ph_grid[b]):>16.2f}{float(Z[i][b]/Z[i][w]):>14.2f}x")

    json.dump({"phases_ms": [float(p) for p in phases], "prc": pv, "dprc_dphase": pg,
               "tau_sensitivity": sens,
               "landscape": {"phase_ms": [float(p) for p in ph_grid],
                             "tau_inh": [float(t) for t in tau_grid],
                             "p40": [[float(z) for z in row] for row in Z]}},
              open(os.path.join(OUT, "gradients.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'gradients.json')}")


if __name__ == "__main__":
    main()
