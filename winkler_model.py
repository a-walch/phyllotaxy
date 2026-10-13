"""
Phyllotactic pressure model - forced Winkler only
==================================================

Tightened version of the model: keeps only the "forced Winkler" block
(tunica membrane under tension, resting on an elastic corpus foundation,
forced by point-like primordia -> Green solution in terms of the modified
Bessel function K0). The exponential geometric inhibition field
(Douady-Couder) and the buckling block are deliberately absent from this
version (Winkler only).

Parameters taken from the Methods section (Table 1 + "physical
dimensions" block):
    rho   = 1.12        plastochron ratio (SEM average, Smyth et al. 1990)
    gamma = 0.5         dimensionless decay parameter (units of R0)
    a0    = 0.35 * R0   contact radius of the youngest primordium (P0)
    a6    = 0.95 * R0   contact radius of the oldest primordium (P6)

Note on gamma (revision 3): gamma = lambda/R0 = 0.5 (lambda = 15.5 um,
k = T/lambda^2 = 1.4e11 Pa/m). Two independent constraints converge:
 (i) mechanical: lambda^2 = T d / E with T = P R_c / 2 = 33 N/m, E = 1.37 MPa
     (turgid linear regime, Beauzamy et al. 2015) and d = 10 um (effective
     thickness of the corpus layer compressed under the tunica, ASSUMED, not
     measured) gives lambda ~ 15 um, i.e. gamma ~ 0.5; d = 20 um would give
     gamma ~ 0.7;
 (ii) phyllotactic: for gamma in [0.1, 0.5] the minima P-1..P-3 stay within
     2.6 deg of the golden spiral; beyond that the deviation on P-3 grows
     quickly (4.2 deg at 0.6, 6 deg at 0.7). gamma = 0.5 is therefore at the
     upper end.
The kernel K0(d/lambda) is not a pure exponential (prefactor x^-1/2): the
Douady-Couder exponential equivalent to K0 over d_eff in [0.35, 2] R0 has a
range gamma_exp ~ 0.41 when lambda/R0 = 0.5 (0.22 when lambda/R0 = 0.25).
Limitation: lambda ~ one to two cells (a0/lambda = 0.70), so the continuum
model is at the edge of its validity; E and d are effective parameters.

Physical block:
    P_turgor = 0.82 MPa      (Beauzamy, Louveaux, Hamant & Boudaoud 2015)
    R_c      = 80 um         (Banwarth-Kuhn et al. 2023)
    R0       = 31 um         (Galvan-Ampudia et al. 2020; the manuscript
                              rounds it to "about 30 um")
    T        = P_turgor * R_c / 2         (Laplace law)
    lambda   = gamma * R0                 (physical screening length)
    k        = T / lambda**2              (derived foundation stiffness)
    F0       = P_turgor * pi * a_mean**2  (point force, CONSTANT,
                                            a_mean ~ 0.5 R0)

Note on F0: an intermediate version of this script tested an age-dependent
F0(a_k) (increasing a_k -> increasing F0, through the same projected-area
theorem F0 = P*pi*a^2). On checking, that weighting shifts the minima by
about 3-4 deg (worse agreement with the naive continuation of the spiral),
and it duplicates an effect already present in the model: an older
primordium has a larger R_k (it has moved away from the CZ), so its
contribution already decays through K0 by geometry alone, with no need to
make F0 grow on top of that. We therefore go back to a CONSTANT F0
(a_mean ~ 0.5 R0), as in the "physical dimensions" block of the manuscript.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.special import kv  # kv(0, x) = K0(x), modified Bessel of the 2nd kind, order 0


# =====================================================================
# 1. PARAMETERS
# =====================================================================

class Params:
    # --- Table 1: dimensionless model (units of R0) ---
    rho = 1.12
    gamma = 0.5     # = lambda/R0 ; equivalent to gamma_exp ~ 0.41 (see the note at
                    # the top of the file) ; mechanical: d = 10 um -> gamma ~ 0.5
    a0_frac = 0.35     # contact radius of P0 (the youngest), in units of R0
    a6_frac = 0.95     # contact radius of P6 (the oldest), in units of R0
    n_gen = 7          # number of primordia generated: P0 .. P6
    divergence_deg = 137.5

    # --- Physical block ---
    P_turgor = 0.82e6      # Pa   (Beauzamy, Louveaux, Hamant & Boudaoud 2015)
    R_c = 80e-6            # m    (Banwarth-Kuhn et al. 2023)
    R0 = 31e-6             # m    (Galvan-Ampudia et al. 2020)
    E_corpus = 1.37e6      # Pa   (Beauzamy 2015, turgid linear regime)
    d_eff_foundation = 10e-6  # m  (effective thickness, ASSUMED)

    @property
    def T_tension(self):
        """Membrane tension (Laplace law): T = P_turgor * R_c / 2  [N/m]."""
        return self.P_turgor * self.R_c / 2

    @property
    def lam(self):
        """Physical screening length: lambda = gamma * R0  [m]."""
        return self.gamma * self.R0

    @property
    def k_foundation(self):
        """Derived foundation stiffness: k = T / lambda^2  [Pa/m]."""
        return self.T_tension / self.lam ** 2

    @property
    def a0_phys(self):
        return self.a0_frac * self.R0

    @property
    def a6_phys(self):
        return self.a6_frac * self.R0

    @property
    def a_mean(self):
        """Mean contact radius used for F0: ~0.5 R0 ("physical dimensions"
        block of the manuscript), CONSTANT for every primordium -- see the
        note at the top of the file on the choice not to weight by age."""
        return 0.5 * self.R0

    @property
    def F0(self):
        """Point force per primordium, CONSTANT: F0 = P_turgor*pi*a_mean^2  [N]."""
        return self.P_turgor * np.pi * self.a_mean ** 2


P = Params()


# =====================================================================
# 2. GENERATION OF THE SPIRAL P0..P6
# =====================================================================

def generate_spiral(P, theta0=0.0):
    """Generate P0..P_{n_gen-1}: list of dicts holding both the dimensionless
    version (units of R0, angle in radians) and the physical one (m, radians)."""
    primordia = []
    divergence = np.deg2rad(P.divergence_deg)
    for k in range(P.n_gen):
        R_dimless = P.rho ** k                                   # units of R0
        theta_k = (theta0 + k * divergence) % (2 * np.pi)
        a_dimless = P.a0_frac + (P.a6_frac - P.a0_frac) * k / (P.n_gen - 1)
        primordia.append(dict(
            k=k,
            R_dimless=R_dimless, theta=theta_k, a_dimless=a_dimless,
            R_phys=R_dimless * P.R0, a_phys=a_dimless * P.R0,
        ))
    return primordia


# =====================================================================
# 3. WINKLER PRESSURE FIELD (K0 Green function)
# =====================================================================

def d_eff(theta, R_source, theta_source, a_source, r_eval=1.0):
    """Effective distance (Plummer-style regularisation) between the
    evaluation point (r_eval, theta) and a point source (R_source,
    theta_source, contact radius a_source). All lengths in the same
    units (dimensionless or physical, consistently)."""
    d = np.sqrt(r_eval ** 2 + R_source ** 2 - 2 * r_eval * R_source * np.cos(theta - theta_source))
    return np.sqrt(d ** 2 + a_source ** 2)


def bessel_field(theta, primordia, gamma, r_eval=1.0):
    """Return (curves, total): curves = list of the individual K0
    contributions (dimensionless, unweighted -- F0 is constant, hence
    omitted here; see sigma_phys below for the physical version in Pa);
    total = their sum.
    """
    curves = []
    for pr in primordia:
        d = d_eff(theta, pr["R_dimless"], pr["theta"], pr["a_dimless"], r_eval=r_eval)
        curves.append(kv(0, d / gamma))
    total = np.sum(curves, axis=0)
    return curves, total


def find_minima(theta_scan, y, n_expected=3):
    """The n_expected deepest local minima, sorted by increasing angle."""
    is_min = (np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], True])
    idx = np.where(is_min)[0]
    idx = idx[np.argsort(y[idx])][:n_expected]
    return np.sort(idx)


# =====================================================================
# 4. MAIN COMPUTATION
# =====================================================================

if __name__ == "__main__":

    primordia = generate_spiral(P)

    print("=" * 72)
    print("Derived physical parameters")
    print("=" * 72)
    print(f"T (membrane tension)         = {P.T_tension:.3g} N/m")
    print(f"lambda = gamma*R0            = {P.lam * 1e6:.2f} um  "
          f"(gamma={P.gamma} : equivalent to gamma_exp ~ 0.41)")
    print(f"k (foundation stiffness)     = {P.k_foundation:.3g} Pa/m")
    k_mech = P.E_corpus / P.d_eff_foundation
    lam_mech = np.sqrt(P.T_tension / k_mech)
    print(f"Mechanical check : k = E/d   = {k_mech:.3g} Pa/m -> lambda = {lam_mech*1e6:.1f} um "
          f"(gamma = {lam_mech/P.R0:.2f}) ; d = 20 um -> gamma = "
          f"{np.sqrt(P.T_tension/(P.E_corpus/20e-6))/P.R0:.2f}")
    print(f"a0 physical (P0)             = {P.a0_phys * 1e6:.2f} um")
    print(f"a6 physical (P6)             = {P.a6_phys * 1e6:.2f} um")
    print(f"a_mean (used for F0)         = {P.a_mean * 1e6:.2f} um  (~0.5 R0)")
    print(f"F0 = P_turgor*pi*a_mean^2    = {P.F0 * 1e3:.3g} mN  (constant, same for every primordium)")
    print(f"a0 / lambda                  = {P.a0_phys / P.lam:.2f}  "
          f"({'a0 < lambda' if P.a0_phys < P.lam else 'a0 > lambda : lambda ~ one cell, continuum limit (cf. docstring)'})")
    print()

    print("=" * 72)
    print(f"Generated positions (rho={P.rho}, R0={P.R0 * 1e6:.0f} um)")
    print("=" * 72)
    for pr in primordia:
        print(f"P{pr['k']}: R={pr['R_phys'] * 1e6:6.2f} um   "
              f"theta={np.rad2deg(pr['theta']):6.1f}deg   "
              f"a={pr['a_phys'] * 1e6:5.2f} um")
    print()

    # --- Angular scan at the CZ periphery (r = R0, i.e. r_eval=1) ---
    theta_scan = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    theta_deg = np.rad2deg(theta_scan)

    # (A) individual effective distances (dimensionless, units of R0)
    deff_curves = [d_eff(theta_scan, pr["R_dimless"], pr["theta"], pr["a_dimless"])
                   for pr in primordia]

    # (B) individual K0 contributions (dimensionless)
    # (C) = their sum (sigma_dimless)
    bessel_curves, sigma_dimless = bessel_field(theta_scan, primordia, P.gamma)

    # (D) physical field in kPa: F0 CONSTANT (a_mean ~ 0.5 R0), the same for
    # every primordium -- cf. the note at the top of the file: no weighting
    # by age, already captured geometrically by the increasing R_k.
    sigma_phys = np.zeros_like(theta_scan)
    for pr in primordia:
        d = d_eff(theta_scan, pr["R_phys"], pr["theta"], pr["a_phys"], r_eval=P.R0)
        sigma_phys += P.F0 / (2 * np.pi * P.lam ** 2) * kv(0, d / P.lam)
    sigma_phys_kPa = sigma_phys / 1e3

    # --- Minima (candidates P-1, P-2, P-3) ---
    # Since F0 is constant, the minima of sigma_phys coincide exactly with
    # those of sigma_dimless (multiplying by a constant does not shift the
    # minima) -- so we compute them only once, on sigma_dimless.
    min_idx = find_minima(theta_scan, sigma_dimless, n_expected=3)
    minima_deg = np.sort(theta_deg[min_idx])
    naive_deg = np.sort([(-kk * P.divergence_deg) % 360 for kk in (1, 2, 3)])

    # Check: the minima of sigma_phys_kPa must coincide with those of
    # sigma_dimless (constant F0 => no shift of the minima).
    idx_phys = find_minima(theta_scan, sigma_phys_kPa, n_expected=3)
    minima_phys_deg = np.sort(theta_deg[idx_phys])
    assert np.allclose(minima_deg, minima_phys_deg, atol=0.5), (
        "The minima of C (dimensionless) and D (physical) no longer coincide "
        "-- check that no age-dependent weighting has been reintroduced into "
        "the computation of sigma_phys.")

    print("=" * 72)
    print("Prediction P-1, P-2, P-3 (Winkler model, K0)")
    print("=" * 72)
    print(f"sigma(theta=0) = {sigma_phys_kPa[0]:.0f} kPa ; max = {sigma_phys_kPa.max():.0f} kPa ; min = {sigma_phys_kPa.min():.0f} kPa")
    print("Pressure field minima                      :", np.round(minima_deg, 1), "deg")
    print("Naive continuation of the spiral (137.5deg) :", np.round(naive_deg, 1), "deg")
    print("Deviations (minima - naive)                 :",
          np.round(minima_deg - naive_deg, 1), "deg")
    print()

    print("=" * 72)
    print("Gamma sweep (deviations from the golden spiral, P-1/P-2/P-3)")
    print("=" * 72)
    for g_scan in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        _, s_scan = bessel_field(theta_scan, primordia, g_scan)
        m_scan = np.sort(theta_deg[find_minima(theta_scan, s_scan, n_expected=3)])
        print(f"gamma={g_scan:.1f} -> deviations = {np.round(m_scan - naive_deg, 1)} deg")
    print()

    print("=" * 72)
    print("Sensitivity of the minima to gamma")
    print("=" * 72)
    for gamma_test in [P.gamma * 0.7, P.gamma, P.gamma * 1.3]:
        _, sigma_test = bessel_field(theta_scan, primordia, gamma_test)
        idx_test = find_minima(theta_scan, sigma_test, n_expected=3)
        m_deg = np.sort(theta_deg[idx_test])
        print(f"gamma={gamma_test:.3f} -> minima = {np.round(m_deg, 1)} deg")
    print()
