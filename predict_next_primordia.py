"""
Prediction of P-1, P-2, P-3 from the forced Winkler model
=========================================================



    theta_k = k * 137.5 deg             (mod 360 deg), k = 0..6
    R_k     = R0 * rho^k                            (rho in [1.1, 1.2])
    a_k     = a0 + (a6 - a0) * k / 6                (organ radius grows with age)

P0 is the youngest, its centre sitting on the CZ (R0). P6 is the oldest.
We then compute sigma(theta) on the circle r = R0 and look for the
minima: those are the candidates P-1, P-2, P-3 (next initiation sites).
"""

import numpy as np
from scipy.special import kv
from phyllotaxis_pressure_model import Params, pressure_point_source

P = Params()

DIVERGENCE = np.deg2rad(137.5)


def generate_spiral(R0, rho, a0, a6, n_gen=7, theta0=0.0):
    """Generate P0..P_{n_gen-1}: list of dicts {k, R, theta, a}."""
    primordia = []
    for k in range(n_gen):
        R_k = R0 * rho**k
        theta_k = (theta0 + k * DIVERGENCE) % (2 * np.pi)
        a_k = a0 + (a6 - a0) * k / (n_gen - 1)
        primordia.append(dict(k=k, R=R_k, theta=theta_k, a=a_k))
    return primordia


def pressure_profile_variable_a(theta_scan, R_eval, primordia, F0=P.F0, lam=P.lam):
    """sigma(theta) on the circle r=R_eval, with an organ radius a_k specific to each primordium."""
    x = R_eval * np.cos(theta_scan)
    y = R_eval * np.sin(theta_scan)
    sigma = np.zeros_like(theta_scan)
    for pr in primordia:
        x_i = pr["R"] * np.cos(pr["theta"])
        y_i = pr["R"] * np.sin(pr["theta"])
        d = np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2)
        d_eff = np.sqrt(d ** 2 + pr["a"] ** 2)
        sigma += pressure_point_source(d_eff, F0=F0, lam=lam)
    return sigma


def find_minima(theta_scan, sigma_theta, n_expected=3):
    """Return the n_expected deepest local minima, sorted by angle."""
    is_min = (np.r_[True, sigma_theta[1:] < sigma_theta[:-1]] &
              np.r_[sigma_theta[:-1] < sigma_theta[1:], True])
    idx = np.where(is_min)[0]
    idx = idx[np.argsort(sigma_theta[idx])][:n_expected]
    return np.sort(idx)


if __name__ == "__main__":

    # --- Scenario parameters (to be adjusted) ---
    rho = 1.15                     # middle of the requested range [1.1, 1.2]
    a0_um, a6_um = 8.0, 28.0       # organ radius: young P0 -> old P6, linear growth
    theta0 = 0.0                   # P0 placed at 0 deg (reference axis)

    primordia = generate_spiral(
        R0=P.R0, rho=rho,
        a0=a0_um * 1e-6, a6=a6_um * 1e-6,
        n_gen=7, theta0=theta0,
    )

    print("=" * 70)
    print(f"Generated positions (rho={rho}, R0={P.R0*1e6:.0f} µm)")
    print("=" * 70)
    for pr in primordia:
        print(f"P{pr['k']}: R={pr['R']*1e6:6.1f} µm   theta={np.rad2deg(pr['theta']):6.1f}°"
              f"   a={pr['a']*1e6:5.1f} µm")
    print()

    # --- Pressure profile on the CZ boundary ---
    theta_scan = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    sigma_theta = pressure_profile_variable_a(theta_scan, P.R0, primordia)

    min_idx = find_minima(theta_scan, sigma_theta, n_expected=3)
    minima_deg = np.sort(np.rad2deg(theta_scan[min_idx]))

    # --- Comparison with the naive continuation of the spiral (137.5° x negative k) ---
    naive_deg = np.sort([(-k * 137.5) % 360 for k in (1, 2, 3)])

    print("=" * 70)
    print("Prediction P-1, P-2, P-3")
    print("=" * 70)
    print("Pressure field minima (model)            :", np.round(minima_deg, 1), "°")
    print("Naive continuation of the spiral (137.5°) :", np.round(naive_deg, 1), "°")
    print("Deviations (minima - naive)               :",
          np.round(minima_deg - naive_deg, 1), "°")
    print()

    # --- Sensitivity to gamma (hence to lambda = gamma*R0): quick +/-30% test ---
    print("=" * 70)
    print("Sensitivity of the minima to gamma")
    print("=" * 70)
    for gamma_test in [P.gamma * 0.7, P.gamma, P.gamma * 1.3]:
        lam_test = gamma_test * P.R0
        sigma_test = pressure_profile_variable_a(theta_scan, P.R0, primordia, lam=lam_test)
        min_idx_test = find_minima(theta_scan, sigma_test, n_expected=3)
        m_deg = np.sort(np.rad2deg(theta_scan[min_idx_test]))
        print(f"gamma={gamma_test:.3f} (lambda={lam_test*1e6:4.1f} µm) -> minima = {np.round(m_deg,1)}°")
    print()
