"""Error metrics comparing the DNS with the homogenized model."""

import numpy as np

from .solver import _eval_at_points, reconstruct_fine_scale


def field_errors_grid(uh_dns, T_macro, tx, ty, epsilon, Lx, Ly, n_grid=40, margin=0.15):
    """RMS differences between the DNS field and the homogenized fields.

    Both are measured on an n_grid x n_grid grid of points covering
    [margin*Lx, (1-margin)*Lx] x [margin*Ly, (1-margin)*Ly]. Points that fall inside a hole
    (where the DNS mesh has no elements) are skipped automatically.

    Returns
    -------
    (err_macro, err_twoscale) :
        err_macro    = RMS(u_DNS - T)                          (macroscopic field only)
        err_twoscale = RMS(u_DNS - (T + eps * chi . grad T))   (first-order two-scale reconstruction)
    """
    xs = np.linspace(margin * Lx, (1 - margin) * Lx, n_grid)
    ys = np.linspace(margin * Ly, (1 - margin) * Ly, n_grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

    u = _eval_at_points(uh_dns, pts).flatten()
    T = _eval_at_points(T_macro, pts).flatten()
    rec = reconstruct_fine_scale(T_macro, tx, ty, pts, epsilon=epsilon)
    # skip points where any field is undefined: inside a hole (no DNS element), or where the
    # corrector lookup lands exactly on a cell face and misses every element
    ok = np.isfinite(u) & np.isfinite(T) & np.isfinite(rec)
    err_macro = float(np.sqrt(np.mean((u[ok] - T[ok]) ** 2)))
    err_twoscale = float(np.sqrt(np.mean((u[ok] - rec[ok]) ** 2)))
    return err_macro, err_twoscale


def tensor_error(K_app, Kstar):
    """Relative Frobenius-norm difference ||K_app - K*||_F / ||K*||_F."""
    return float(np.linalg.norm(K_app - Kstar) / np.linalg.norm(Kstar))


def fit_convergence_rate(epsilons, errors):
    """Least-squares slope of log(error) vs log(epsilon): error ~ C * eps^p.

    Returns (p, C).
    """
    p, logC = np.polyfit(np.log(epsilons), np.log(errors), 1)
    return float(p), float(np.exp(logC))
