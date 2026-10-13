"""
Phyllotactic pressure model - Arabidopsis SAM
===============================================

Implements the three blocks of the model:

  1. Geometric inhibition field (Douady-Couder / Linum, Walch 2022)
  2. Physical pressure field (Winkler membrane forced by point-like
     primordia, Green solution in terms of the Bessel function K0)
  3. Buckling threshold (homogeneous Winkler, Green/Steele/Newell)

All default numerical values are the calibrated ones; every parameter
is commented with its source.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.special import kv  # kv(0, x) = K0(x), modified Bessel function of the 2nd kind


# =====================================================================
# 1. CALIBRATED PARAMETERS
# =====================================================================

class Params:
    """All the model parameters, with their origin."""

    # --- Block 1: geometric inhibition field (Walch 2022, Linum) ---
    gamma = 0.2                 # dimensionless range (units of CZ radius), never readjusted

    # --- Block 2: physical calibration (forced Winkler) ---
    P_turgor = 0.82e6           # Pa,  Beauzamy, Louveaux, Hamant & Boudaoud 2015
    R_c = 75e-6                 # m,   radius of curvature of the WT meristem, idem
    R0 = 31e-6                  # m,   CZ radius where auxin is maximal, Galvan-Ampudia et al. 2020
    a = 15e-6                   # m,   effective contact radius (auxin peak width)

    # --- Block 4: kinematics / plastochron ---
    T_plasto_h = 9.6            # h,   inflorescence plastochron (mean), Alimchandani et al. 2026
    T_plasto_h_sd = 3.0
    rho_SEM = 1.12              # plastochron ratio measured on SEM (Walch, own data)
    rho_confocal = 1.29         # ratio measured on C(R) profiles, Galvan-Ampudia (mean 1.28-1.30)

    # --- Block 5: wall relaxation time (for the Deborah number) ---
    tau_relax_min_h = 1 / 60    # 1 minute, lower bound (Yamamoto et al. 1970 ; Hayot et al. 2012)
    tau_relax_max_h = 1.0       # 1 hour, upper bound

    # --- Block 3: buckling (homogeneous Winkler) ---
    n_mode = 5                  # number of waves observed (e.g. a calyx with 5 sepals)
    tunica_thickness = 10e-6    # m, effective L1+L2 thickness (assumption, not measured)

    # --- Independent comparison (Milani et al. 2011 ; Kierzkowski et al. 2012) ---
    E_measured_center_MPa = 5.0     # +/- 2 MPa, centre of the apex
    E_measured_range_MPa = (0.01, 10.0)  # range reported over the whole map

    @property
    def T_tension(self):
        """Membrane tension from the Laplace law: T = P_turgor * R_c / 2  [N/m]."""
        return self.P_turgor * self.R_c / 2

    @property
    def lam(self):
        """Physical range lambda = gamma * R0  [m]."""
        return self.gamma * self.R0

    @property
    def k_foundation(self):
        """Derived foundation stiffness k = T / lambda^2  [Pa/m]."""
        return self.T_tension / self.lam**2

    @property
    def F0(self):
        """Point force per primordium F0 = P_turgor * pi * a^2  [N]."""
        return self.P_turgor * np.pi * self.a**2

    @property
    def deborah(self):
        """Deborah number tau_relax / T_plastochron (min, max)."""
        return (self.tau_relax_min_h / self.T_plasto_h,
                self.tau_relax_max_h / self.T_plasto_h)


P = Params()


# =====================================================================
# 2. GEOMETRIC INHIBITION FIELD (Douady-Couder / Linum)
# =====================================================================

def d_ij(theta_i, R_j, theta_j):
    """Geometric distance between the test point (unit CZ radius, angle theta_i)
    and organ j located at (R_j, theta_j), in units of CZ radius."""
    return np.sqrt(1 + R_j**2 - 2 * R_j * np.cos(theta_i - theta_j))


def inhibition_field(theta, primordia, gamma=P.gamma):
    """U_i(theta) = sum_j exp(-d_ij / gamma)

    primordia : list of tuples (R_j, theta_j), theta_j in radians.
    theta : array of angles (radians) at which to evaluate the field.
    """
    U = np.zeros_like(theta, dtype=float)
    for R_j, theta_j in primordia:
        U += np.exp(-d_ij(theta, R_j, theta_j) / gamma)
    return U


# =====================================================================
# 3. PHYSICAL PRESSURE FIELD (forced Winkler, K0 Green function)
# =====================================================================

def pressure_point_source(d, F0=P.F0, lam=P.lam):
    """sigma(d) = F0 / (2 pi lambda^2) * K0(d / lambda)   [Pa]

    d : Euclidean distance (m) to the primordium; it may be regularised
    beforehand with d_eff = sqrt(d^2 + a^2) to avoid the singularity of K0 at 0.
    """
    return F0 / (2 * np.pi * lam**2) * kv(0, d / lam)


def pressure_map(X, Y, primordia_xy, F0=P.F0, lam=P.lam, a=P.a):
    """2D pressure map sigma(x,y), linear superposition of the sources.

    X, Y : meshgrid (m)
    primordia_xy : list of tuples (x_i, y_i) in m
    """
    sigma = np.zeros_like(X, dtype=float)
    for x_i, y_i in primordia_xy:
        d = np.sqrt((X - x_i)**2 + (Y - y_i)**2)
        d_eff = np.sqrt(d**2 + a**2)     # finite-size regularisation
        sigma += pressure_point_source(d_eff, F0=F0, lam=lam)
    return sigma


def pressure_profile(theta, R, primordia_polar, F0=P.F0, lam=P.lam, a=P.a):
    """Angular profile sigma(theta) on a circle of radius R (m).

    primordia_polar : list of tuples (R_i, theta_i) in (m, rad)
    """
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    sigma = np.zeros_like(theta, dtype=float)
    for R_i, theta_i in primordia_polar:
        x_i, y_i = R_i * np.cos(theta_i), R_i * np.sin(theta_i)
        d = np.sqrt((x - x_i)**2 + (y - y_i)**2)
        d_eff = np.sqrt(d**2 + a**2)
        sigma += pressure_point_source(d_eff, F0=F0, lam=lam)
    return sigma


# =====================================================================
# 4. BUCKLING (homogeneous Winkler, beam/shell compressed on a foundation)
# =====================================================================

def buckling_wavelength(D, k):
    """lambda_c = 2 pi (D/k)^(1/4)   [m]"""
    return 2 * np.pi * (D / k)**0.25


def buckling_mode_number(R, D, k):
    """n = R * (k/D)^(1/4) : number of waves selected on a ring of radius R."""
    return R * (k / D)**0.25


def buckling_critical_stress(D, k):
    """N_c = 2 sqrt(D k)   [N/m] : critical buckling compression."""
    return 2 * np.sqrt(D * k)


def invert_D_from_mode(R, n, k):
    """Invert D from the observed number of waves n and the radius R: D = k (R/n)^4."""
    return k * (R / n)**4


def flexural_rigidity_to_modulus(D, thickness):
    """E = 12 D / t^3, flexural rigidity -> equivalent elastic modulus (thin shell)."""
    return 12 * D / thickness**3


# =====================================================================
# 5. USAGE EXAMPLE
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Derived physical parameters")
    print("=" * 70)
    print(f"T (membrane tension)         = {P.T_tension:.3g} N/m")
    print(f"lambda (physical range)      = {P.lam*1e6:.3g} µm")
    print(f"k (foundation stiffness)     = {P.k_foundation:.3g} Pa/m")
    print(f"F0 (force per primordium)    = {P.F0*1e3:.3g} mN")
    De_min, De_max = P.deborah
    print(f"Deborah number               = {De_min:.4f} - {De_max:.3f}  (<< 1 : quasi-static OK)")
    print()

    # --- Buckling: D inverted from n=5, comparison with Milani et al. ---
    D_inferred = invert_D_from_mode(R=P.R0, n=P.n_mode, k=P.k_foundation)
    E_inferred = flexural_rigidity_to_modulus(D_inferred, P.tunica_thickness) / 1e6  # MPa
    print("=" * 70)
    print("Buckling (homogeneous Winkler) : D / E consistency")
    print("=" * 70)
    print(f"D inverted (n=5, R0, k)       = {D_inferred:.3g} N.m")
    print(f"E implied (t={P.tunica_thickness*1e6:.0f} µm)           = {E_inferred:.1f} MPa")
    print(f"E measured (Milani et al. 2011) = {P.E_measured_center_MPa} +/- 2 MPa (centre)")
    print(f"  -> discrepancy factor : {E_inferred/P.E_measured_center_MPa:.1f}x")
    print()

    # --- Primordia P0-P4 (Galvan-Ampudia), polar positions ---
    # Measured radii (actual peak after the trough), angles ~137 degrees (Galvan-Ampudia 2020)
    R_primordia_um = [31, 40, 50, 65, 80]          # µm
    theta_primordia_deg = [k * 137.0 for k in range(5)]  # chosen origin convention

    primordia_polar = [(R * 1e-6, np.deg2rad(th))
                        for R, th in zip(R_primordia_um, theta_primordia_deg)]

    theta_scan = np.linspace(0, 2 * np.pi, 720)
    sigma_theta = pressure_profile(theta_scan, P.R0, primordia_polar)

    # Simple maxima/minima detection
    peaks = (np.diff(np.sign(np.diff(sigma_theta))) < 0).nonzero()[0] + 1
    troughs = (np.diff(np.sign(np.diff(sigma_theta))) > 0).nonzero()[0] + 1

    print("=" * 70)
    print("Pressure profile on the CZ periphery (r = R0)")
    print("=" * 70)
    print("Maxima (deg) :", np.round(np.rad2deg(theta_scan[peaks]), 1))
    print("Minima (deg) :", np.round(np.rad2deg(theta_scan[troughs]), 1))
    print()
