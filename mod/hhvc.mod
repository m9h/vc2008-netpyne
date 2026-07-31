TITLE Hodgkin-Huxley Na/K channels with Vierling-Claassen et al. (2008) rate functions

COMMENT
Sodium and potassium channels for the two-compartment cells of

  Vierling-Claassen D, Siekmeier P, Stufflebeam S, Kopell N (2008)
  "Modeling GABA alterations in schizophrenia: a link between impaired
   inhibition and altered gamma and beta range auditory entrainment."
  J Neurophysiol 99(5):2656-2671.

The paper's METHODS give voltage-gated kinetics in the classic Hodgkin-Huxley
"reduced potential" convention (the paper notes "the inside potential is 0 mV",
i.e. rates are functions of the depolarization above the leak potential Em):

  vr = v - em                     (mV, depolarization positive)

  alpha_m = 0.1 (25 - vr) / (exp((25 - vr)/10) - 1)
  beta_m  = 4 exp(-vr/18)
  alpha_h = 0.07 exp(-vr/20)
  beta_h  = 1 / (exp((30 - vr)/10) + 1)
  alpha_n = an_scale (10 - vr) / (exp((10 - vr)/10) - 1)
  beta_n  = 0.125 exp(-vr/80)

  G_Na = gnabar * m^3 * h        G_K = gkbar * n^4

NOTE (see DEVIATIONS.md): the paper prints the alpha_n prefactor as 0.1, whereas
the canonical HH value is 0.01. It is exposed here as `an_scale` so both can be
run; the default reproduces the printed value.

Singularities at vr = 25 (m) and vr = 10 (n) are handled with the standard
L'Hopital expansion used by NEURON's own hh.mod (vtrap).
ENDCOMMENT

NEURON {
    SUFFIX hhvc
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnabar, gkbar, gna, gk, em, an_scale
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gnabar   = 0.12 (S/cm2)   : calibrated; paper's Table 1 value is unit-ambiguous
    gkbar    = 0.04 (S/cm2)   :   (see DEVIATIONS.md)
    em       = -59.4 (mV)     : leak/reference potential for the reduced-potential rates
    an_scale = 0.1            : as printed in the paper (0.01 = canonical HH)
}

ASSIGNED {
    v (mV)
    ena (mV)
    ek (mV)
    gna (S/cm2)
    gk (S/cm2)
    ina (mA/cm2)
    ik (mA/cm2)
}

STATE { m h n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gna = gnabar * m*m*m * h
    gk  = gkbar * n*n*n*n
    ina = gna * (v - ena)
    ik  = gk * (v - ek)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

ASSIGNED { minf hinf ninf mtau (ms) htau (ms) ntau (ms) }

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
    n' = (ninf - n)/ntau
}

FUNCTION vtrap(x, y) {   : x/(exp(x/y)-1), well behaved as x -> 0
    if (fabs(x/y) < 1e-6) {
        vtrap = y * (1 - x/y/2)
    } else {
        vtrap = x / (exp(x/y) - 1)
    }
}

PROCEDURE rates(vm (mV)) {
    LOCAL vr, alpham, betam, alphah, betah, alphan, betan
    vr = vm - em

    alpham = 0.1 * vtrap(25 - vr, 10)
    betam  = 4 * exp(-vr/18)
    alphah = 0.07 * exp(-vr/20)
    betah  = 1 / (exp((30 - vr)/10) + 1)
    alphan = an_scale * vtrap(10 - vr, 10)
    betan  = 0.125 * exp(-vr/80)

    mtau = 1/(alpham + betam)   minf = alpham * mtau
    htau = 1/(alphah + betah)   hinf = alphah * htau
    ntau = 1/(alphan + betan)   ninf = alphan * ntau
}
