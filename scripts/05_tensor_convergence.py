"""Does the conductivity tensor measured on a finite perforated sample converge to K*?

For each eps = L/n, the DNS is solved twice on the n x n perforated plate with the affine
boundary condition u = x_j on the whole outer boundary (j = 1, 2; holes insulated). The
volume-averaged flux gives the sample's full apparent tensor K_app (see
solver.apparent_tensor), which is compared entry by entry with K* from the cell problem.

Theory: the affine boundary data forces a boundary layer of width ~eps along the outer
edge, so K_app - K* = O(eps), and K_app >= K* (as quadratic forms). Both a circle and the
rotated ellipse of 04_homogenization_error.py are run, each at two DNS mesh resolutions to
check that the FEM error is far below the difference being measured.

Usage:
    python scripts/05_tensor_convergence.py            # n = 2..12
    python scripts/05_tensor_convergence.py --quick    # n = 2, 4, 6
Outputs: figures/tensor_convergence.png, results/tensor_convergence.csv
(--plot-only redraws the figure from the CSV)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homogenization.geometry import make_unit_cell, make_periodic_tiling
from homogenization.solver import solve_corrector, compute_effective_tensor, apparent_tensor
from homogenization.validation import tensor_error, fit_convergence_rate

ROOT = Path(__file__).resolve().parents[1]
FIGURES, RESULTS = ROOT / "figures", ROOT / "results"

K_SOLID = 1.0
L = 0.8
N_LIST = [2, 3, 4, 6, 8, 10, 12]
CELL_RATIO = 160                  # K* accurate to ~1e-7 at this resolution (P2, curved)
DNS_RATIOS = [30, 60]             # coarse / fine DNS meshes (element size eps/ratio)
RTOL = 1e-10
SHAPES = {
    "circle": dict(a=0.3, b=None, theta=0.0, color="tab:blue", label="circle, $r$ = 0.3"),
    "ellipse": dict(a=0.3, b=0.12, theta=40.0, color="tab:red",
                    label="ellipse, $a$ = 0.3, $b$ = 0.12, 40$^\\circ$"),
}


def kstar(cfg):
    mesh, _, facets = make_unit_cell(hole_radius=cfg["a"], hole_radius_b=cfg["b"],
                                     hole_rotation_deg=cfg["theta"], mesh_ratio=CELL_RATIO)
    tx = solve_corrector(mesh, facets, "x", k_value=K_SOLID)
    ty = solve_corrector(mesh, facets, "y", k_value=K_SOLID)
    return compute_effective_tensor(mesh, tx, ty, k_value=K_SOLID)


def main(n_list):
    FIGURES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    eps = np.array([L / n for n in n_list])
    Ks, rows = {}, []
    for name, cfg in SHAPES.items():
        Ks[name] = kstar(cfg)
        print(f"\n### {name}: K* = {np.array2string(Ks[name], precision=7)} ###", flush=True)
        for dr in DNS_RATIOS:
            for n, e in zip(n_list, eps):
                mesh, _, facets = make_periodic_tiling(
                    epsilon=e, n_cells_x=n, n_cells_y=n, hole_radius_ratio=cfg["a"],
                    hole_radius_ratio_b=cfg["b"], hole_rotation_deg=cfg["theta"], mesh_ratio=dr)
                K = apparent_tensor(mesh, facets, k_value=K_SOLID, rtol=RTOL)
                d = K - Ks[name]
                print(f"   DNS ratio {dr:3d}  n={n:3d}  rel. error {tensor_error(K, Ks[name]):.3e}   "
                      f"dK11={d[0,0]:+.2e} dK22={d[1,1]:+.2e} dK12={d[0,1]:+.2e} "
                      f"min eig(dK)={np.linalg.eigvalsh(0.5*(d+d.T)).min():+.1e}", flush=True)
                rows.append([name == "ellipse", dr, n, e, *K.ravel(), *Ks[name].ravel()])

    np.savetxt(RESULTS / "tensor_convergence.csv", np.array(rows, dtype=float), delimiter=",",
               header="is_ellipse,dns_ratio,n,eps,Kapp11,Kapp12,Kapp21,Kapp22,Kstar11,Kstar12,Kstar21,Kstar22",
               comments="", fmt="%.12g")

    plot(RESULTS / "tensor_convergence.csv")


def plot(csv_path):
    D = np.genfromtxt(csv_path, delimiter=",", names=True)
    ratios = sorted({int(r) for r in D["dns_ratio"]})
    coarse, fine = ratios[0], ratios[-1]

    def pick(is_ellipse, ratio):
        m = (D["is_ellipse"] == is_ellipse) & (D["dns_ratio"] == ratio)
        Ka = np.stack([D["Kapp11"][m], D["Kapp12"][m], D["Kapp21"][m], D["Kapp22"][m]], 1).reshape(-1, 2, 2)
        Ks = np.array([[D["Kstar11"][m][0], D["Kstar12"][m][0]], [D["Kstar21"][m][0], D["Kstar22"][m][0]]])
        return D["eps"][m], Ka, Ks

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))
    for is_ell, (name, cfg) in enumerate(SHAPES.items()):
        eps, Ka, Ks = pick(is_ell, fine)
        _, Ka_c, _ = pick(is_ell, coarse)
        err = np.array([tensor_error(K, Ks) for K in Ka])
        err_c = np.array([tensor_error(K, Ks) for K in Ka_c])
        p, _ = fit_convergence_rate(eps, err)
        ax.loglog(eps, err_c, "o", color=cfg["color"], mfc="none", ms=10, alpha=0.5)
        ax.loglog(eps, err, "o-", color=cfg["color"], label=f"{cfg['label']}  (slope {p:.2f})")
    ax.loglog([], [], "o", color="gray", mfc="none", ms=10, alpha=0.5,
              label="same, DNS mesh twice as coarse")
    ax.loglog(eps, 0.5 * err[0] * eps / eps[0], ":", color="gray")
    ax.text(eps[-1] * 1.04, 0.5 * err[0] * eps[-1] / eps[0], r"$\varepsilon^1$", color="gray", va="bottom")
    ax.set(xlabel=r"cell size $\varepsilon$",
           ylabel=r"$\|\kappa_{\rm app}-\kappa^\star\|_F \,/\, \|\kappa^\star\|_F$",
           title=r"Apparent tensor of an $n\times n$ sample vs. $\kappa^\star$")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, which="both", alpha=0.3)

    eps, Ka, Ks = pick(1, fine)
    d = Ka - Ks
    for (i, j), mk, lab in (((0, 0), "o-", r"$\kappa_{11}$"), ((1, 1), "s-", r"$\kappa_{22}$")):
        p, _ = fit_convergence_rate(eps, d[:, i, j])
        ax2.loglog(eps, d[:, i, j], mk, label=f"{lab}  (slope {p:.2f})")
    # the off-diagonal deviation is ~1000x smaller and at the FEM resolution: report a bound only
    bound = np.abs(d[:, 0, 1]).max()
    ax2.text(0.97, 0.05, r"$|\kappa_{\rm app,12}-\kappa^\star_{12}| < $" + f"{bound:.0e}\n(below the FEM resolution)",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9, color="0.3")
    ax2.set(xlabel=r"cell size $\varepsilon$", ylabel=r"$\kappa_{{\rm app},ij}-\kappa^\star_{ij}$",
            title="Rotated ellipse, diagonal entries")
    ax2.legend(fontsize=9); ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "tensor_convergence.png", dpi=150)
    print(f"Figure written to {FIGURES / 'tensor_convergence.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="n = 2, 4, 6 only")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved CSV")
    args = ap.parse_args()
    if args.plot_only:
        FIGURES.mkdir(exist_ok=True)
        plot(RESULTS / "tensor_convergence.csv")
    else:
        main([2, 4, 6] if args.quick else N_LIST)
