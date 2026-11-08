"""Main result: the homogenization error of the two-scale reconstruction is O(eps^2).

Ground truth is the DNS on the fully perforated plate. For each cell size eps = L/n we
measure, on a 40 x 40 grid of interior points,

    RMS(u_DNS - T)                       the macroscopic field alone        (expected O(eps))
    RMS(u_DNS - (T + eps chi . grad T))  the first-order reconstruction     (expected O(eps^2))

The second error mixes the homogenization (model) error with the FEM error of the DNS, the
cell problem and the macro solve. To make sure we are looking at the former, every solve uses
quadratic (P2) elements on curved (second-order) meshes, and the cell and DNS meshes are
refined in three levels: if the error stops changing between levels, the FEM error is
negligible and what remains is the homogenization error.

The hole is a ROTATED ELLIPSE. For a centred circle with these boundary conditions the
reconstruction is an exact solution of the DNS problem (zero model error), so a circle would
only measure FEM error.

Usage:
    python scripts/04_homogenization_error.py            # n = 2..12, three mesh levels
    python scripts/04_homogenization_error.py --quick    # n = 2, 3, 4, 6
Outputs: figures/homogenization_error.png, results/homogenization_error.csv
(--plot-only redraws the figure from the CSV, ~1 s)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homogenization.geometry import make_unit_cell, make_periodic_tiling
from homogenization.solver import (
    solve_corrector, compute_effective_tensor, solve_macro, solve_heat,
)
from homogenization.validation import field_errors_grid, fit_convergence_rate

ROOT = Path(__file__).resolve().parents[1]
FIGURES, RESULTS = ROOT / "figures", ROOT / "results"

K_SOLID = 1.0
A, B, THETA = 0.3, 0.12, 40.0     # rotated ellipse (semi-axes in unit-cell units, degrees)
L = 0.8                           # domain is L x L
N_LIST = [2, 3, 4, 6, 8, 10, 12]
N_FIT_MIN = 4                     # slopes fitted over n >= 4 (eps <= 0.2); for n = 2, 3 the
                                  # sampling region lies within one cell of the Dirichlet edges
MACRO_N = 80                      # fixed P2 macro mesh: its FEM error is negligible at any eps
RTOL = 1e-10                      # linear-solver tolerance for the DNS and macro solves
# (cell mesh_ratio, DNS mesh_ratio): element size = 1.5/ratio on the unit cell, eps/ratio in the DNS
LEVELS = [(40, 20), (80, 40), (120, 60)]


def run_level(cell_ratio, dns_ratio, n_list):
    mesh, _, facets = make_unit_cell(hole_radius=A, hole_radius_b=B,
                                     hole_rotation_deg=THETA, mesh_ratio=cell_ratio)
    tx = solve_corrector(mesh, facets, "x", k_value=K_SOLID)
    ty = solve_corrector(mesh, facets, "y", k_value=K_SOLID)
    Kstar = compute_effective_tensor(mesh, tx, ty, k_value=K_SOLID)
    print(f"   K* = [[{Kstar[0,0]:.6f}, {Kstar[0,1]:.6f}], [{Kstar[1,0]:.6f}, {Kstar[1,1]:.6f}]]",
          flush=True)
    T = solve_macro(Kstar, L, L, nx=MACRO_N, ny=MACRO_N, T_left=0.0, T_right=1.0, rtol=RTOL)
    err_T, err_2s = [], []
    for n in n_list:
        eps = L / n
        dns_mesh, _, dns_facets = make_periodic_tiling(
            epsilon=eps, n_cells_x=n, n_cells_y=n, hole_radius_ratio=A,
            hole_radius_ratio_b=B, hole_rotation_deg=THETA, mesh_ratio=dns_ratio)
        uh, _ = solve_heat(dns_mesh, dns_facets, k_value=K_SOLID, rtol=RTOL)
        e_T, e_2s = field_errors_grid(uh, T, tx, ty, eps, L, L)
        print(f"   n={n:3d} eps={eps:.4f}  RMS(u-T)={e_T:.3e}  RMS(u-T-eps*chi.gradT)={e_2s:.3e}",
              flush=True)
        err_T.append(e_T); err_2s.append(e_2s)
    return np.array(err_T), np.array(err_2s)


def main(n_list):
    FIGURES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    eps = np.array([L / n for n in n_list])
    err_T, err_2s = {}, {}
    for lev, (cr, dr) in enumerate(LEVELS, start=1):
        print(f"\n### mesh level {lev}: cell mesh_ratio {cr}, DNS mesh_ratio {dr} ###", flush=True)
        err_T[lev], err_2s[lev] = run_level(cr, dr, n_list)

    top = len(LEVELS)
    print("\n=== FEM check: relative change of RMS(u - T - eps chi.gradT) between the last two levels ===")
    for i, n in enumerate(n_list):
        d = abs(err_2s[top][i] - err_2s[top - 1][i]) / err_2s[top][i]
        print(f"  n={n:3d}: {d:6.2%}")
    sel = np.array(n_list) >= N_FIT_MIN
    p2s, _ = fit_convergence_rate(eps[sel], err_2s[top][sel])
    pT, _ = fit_convergence_rate(eps[sel], err_T[top][sel])
    print(f"\nFinest level, fit over n >= {N_FIT_MIN}: two-scale error ~ eps^{p2s:.2f}, "
          f"macro-only error ~ eps^{pT:.2f}")

    header = "n,eps," + ",".join(f"rms_u_minus_T_L{l},rms_u_minus_twoscale_L{l}" for l in err_T)
    rows = np.column_stack([n_list, eps] + [c for l in err_T for c in (err_T[l], err_2s[l])])
    np.savetxt(RESULTS / "homogenization_error.csv", rows, delimiter=",", header=header,
               comments="", fmt="%.10g")

    plot(RESULTS / "homogenization_error.csv")


def plot(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    n_list, eps = data["n"].astype(int), data["eps"]
    levels = sorted({int(k.split("_L")[-1]) for k in data.dtype.names if "_L" in k})
    top = levels[-1]
    err_T = data[f"rms_u_minus_T_L{top}"]
    err_2s = {l: data[f"rms_u_minus_twoscale_L{l}"] for l in levels}
    sel = n_list >= N_FIT_MIN
    pT, _ = fit_convergence_rate(eps[sel], err_T[sel])
    p2s, _ = fit_convergence_rate(eps[sel], err_2s[top][sel])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    i0 = int(np.argmax(sel))
    ax.axvspan(eps[i0] * 1.12, eps[0] * 1.15, color="0.92", zorder=0)
    ax.text(np.sqrt(eps[i0] * eps[0]) * 1.07, 0.6, "near the edges,\nnot fitted", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.4")
    ax.loglog(eps, err_T, "^-", color="0.35",
              label=rf"macroscopic field only: $u_{{\rm DNS}}-T$  (slope {pT:.2f})")
    ax.loglog(eps, err_2s[top], "o-", color="tab:red", ms=5,
              label=rf"two-scale: $u_{{\rm DNS}}-(T+\varepsilon\chi\cdot\nabla T)$  (slope {p2s:.2f})")
    for l, c in zip(levels[:-1], ("tab:blue", "tab:orange")):
        cr, dr = LEVELS[l - 1]
        ax.loglog(eps, err_2s[l], "o", color=c, mfc="none", ms=11 - 2 * l, alpha=0.6,
                  label=f"two-scale, coarser meshes (cell {cr}, DNS {dr})")
    for ref, y0 in ((1, err_T[i0]), (2, err_2s[top][i0])):
        x = eps[i0:]
        ax.loglog(x, 0.55 * y0 * (x / x[0]) ** ref, ":", color="gray")
        ax.text(x[0] * 1.03, 0.55 * y0, rf"$\varepsilon^{ref}$", color="gray", va="center")
    ax.set(xlabel=r"cell size $\varepsilon$", ylabel="RMS error against the DNS",
           title=f"Rotated ellipse ($a$={A}, $b$={B}, {THETA:.0f}$^\\circ$), P2 on curved meshes")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "homogenization_error.png", dpi=150)
    print(f"Figure written to {FIGURES / 'homogenization_error.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="n = 2, 3, 4, 6 only")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved CSV")
    args = ap.parse_args()
    if args.plot_only:
        FIGURES.mkdir(exist_ok=True)
        plot(RESULTS / "homogenization_error.csv")
    else:
        main([2, 3, 4, 6] if args.quick else N_LIST)
