"""The boundary layer: how far into the plate the boundary conditions are felt.

With the affine condition u = x_1 imposed on the whole outer boundary (holes insulated), the
macroscopic solution is T = x_1 and the two-scale expansion x_1 + eps*chi_1({x/eps}) is an exact
solution of the perforated problem everywhere except on the outer boundary, where it misses
the data by eps*chi_1. The difference

    e = u_DNS - (x_1 + eps*chi_1({x/eps}))

is therefore the boundary-layer correction alone. Plotting e/eps against d/eps, with d the
distance to the boundary, for several n shows that the profiles collapse onto one curve, so
that the layer is O(eps) thick, and that this curve is an exponential exp(-gamma*d/eps),
whose rate gamma is fitted.

For each eps = L/n the DNS (P2, curved mesh, element size eps/60) and the corrector are
sampled on a uniform grid, and the RMS of e/eps is taken in strips of width eps/16 parallel to
the boundary.

Usage:
    python scripts/07_boundary_layer.py              # n = 4, 8, 12 (~5 min)
    python scripts/07_boundary_layer.py --plot-only  # redraw from results/
Outputs: figures/boundary_layer_profile.png, results/boundary_layer_profiles.csv,
         results/boundary_layer_map.npz (error map of the middle n, for the left panel)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from matplotlib.colors import LogNorm

from homogenization.geometry import make_unit_cell, make_periodic_tiling
from homogenization.solver import solve_corrector

ROOT = Path(__file__).resolve().parents[1]
FIGURES, RESULTS = ROOT / "figures", ROOT / "results"

A, B, THETA = 0.3, 0.12, 40.0     # same ellipse as 04 and 05
L = 0.8
N_LIST = [4, 8, 12]
N_MAP = 8                         # error map drawn for this n
CELL_RATIO, DNS_RATIO = 120, 60   # finest levels of 04_homogenization_error.py
N_GRID = 801                      # sampling grid per side (spacing ~eps/67 at n = 12)
BIN = 1 / 16                      # strip width, in units of eps
D_MAX = 1.5                       # profiles up to 1.5 eps from the boundary
FIT = (0.0625, 0.75)              # exponential fitted over this range of d/eps


def evaluate(func, pts):
    """Values of a scalar Function at points (N, 3); NaN where no cell contains the point."""
    dom = func.function_space.mesh
    tree = geometry.bb_tree(dom, dom.topology.dim)
    col = geometry.compute_colliding_cells(dom, geometry.compute_collisions_points(tree, pts), pts)
    out = np.full(len(pts), np.nan)
    idx, cells = [], []
    for i in range(len(pts)):
        links = col.links(i)
        if len(links):
            idx.append(i); cells.append(links[0])
    idx = np.array(idx)
    out[idx] = func.eval(pts[idx], np.array(cells, dtype=np.int32))[:, 0]
    return out


def affine_dns(n):
    """DNS on the n x n tiling with u = x_1 on the whole outer boundary."""
    eps = L / n
    mesh, _, facets = make_periodic_tiling(epsilon=eps, n_cells_x=n, n_cells_y=n, hole_radius_ratio=A,
                                           hole_radius_ratio_b=B, hole_rotation_deg=THETA,
                                           mesh_ratio=DNS_RATIO)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    dofs = fem.locate_dofs_topological(V, 1, np.concatenate([facets.find(t) for t in (1, 2, 3, 4)]))
    g = fem.Function(V)
    g.interpolate(lambda X: X[0])
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = LinearProblem(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
                            fem.Constant(mesh, 0.0) * v * ufl.dx, bcs=[fem.dirichletbc(g, dofs)],
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                            petsc_options_prefix="layer_")
    return problem.solve()


def main():
    FIGURES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    mesh, _, facets = make_unit_cell(hole_radius=A, hole_radius_b=B, hole_rotation_deg=THETA,
                                     mesh_ratio=CELL_RATIO)
    chi = solve_corrector(mesh, facets, "x")
    edges = np.arange(0, D_MAX + 1e-12, BIN)
    profiles = {}
    for n in N_LIST:
        eps = L / n
        uh = affine_dns(n)
        xs = np.linspace(1e-7, L - 1e-7, N_GRID)
        X, Y = np.meshgrid(xs, xs)
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        cell_pts = np.stack([np.mod(P[:, 0] / eps, 1), np.mod(P[:, 1] / eps, 1), np.zeros(X.size)], axis=1)
        e = (evaluate(uh, P) - (P[:, 0] + eps * evaluate(chi, cell_pts))).reshape(X.shape)
        d = np.minimum.reduce([X, L - X, Y, L - Y]) / eps
        ok = np.isfinite(e)
        profiles[n] = [np.sqrt(np.mean(e[ok & (d >= lo) & (d < lo + BIN)] ** 2)) / eps for lo in edges[:-1]]
        bulk = ok & (d > D_MAX)
        print(f"n={n:3d}  RMS(e)/eps at the boundary {profiles[n][0]:.2e}, beyond {D_MAX} eps "
              f"{np.sqrt(np.mean(e[bulk] ** 2)) / eps:.2e}, mean there {np.mean(e[bulk]) / eps:.1e}",
              flush=True)
        if n == N_MAP:
            np.savez_compressed(RESULTS / "boundary_layer_map.npz", e=e.astype(np.float32), eps=eps)
    centres = edges[:-1] + BIN / 2
    np.savetxt(RESULTS / "boundary_layer_profiles.csv",
               np.column_stack([centres] + [profiles[n] for n in N_LIST]), delimiter=",",
               header="d_over_eps," + ",".join(f"rms_e_over_eps_n{n}" for n in N_LIST),
               comments="", fmt="%.8g")
    plot()


def plot():
    data = np.genfromtxt(RESULTS / "boundary_layer_profiles.csv", delimiter=",", names=True)
    m = np.load(RESULTS / "boundary_layer_map.npz")
    d = data["d_over_eps"]
    cols = [c for c in data.dtype.names if c.startswith("rms")]
    sel = (d >= FIT[0]) & (d <= FIT[1])
    ref = data[cols[-1]]
    slope, logc = np.polyfit(d[sel], np.log(ref[sel]), 1)
    gamma = -slope
    print(f"fitted decay rate gamma = {gamma:.2f}; the layer drops by 10 every {np.log(10) / gamma:.2f} eps")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    ax.set_facecolor("0.72")      # holes (no DNS value) in grey
    E = np.abs(m["e"]) / float(m["eps"])
    xs = np.linspace(1e-7, L - 1e-7, E.shape[0])
    im = ax.pcolormesh(xs, xs, np.ma.masked_invalid(E), norm=LogNorm(3e-5, 1e-2),
                       cmap="magma_r", shading="auto", rasterized=True)
    n_map = round(L / float(m["eps"]))
    ax.set(aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title=rf"(a) $|u_{{\rm DNS}}-(x_1+\varepsilon\chi_1)|/\varepsilon$, $n={n_map}$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axs[1]
    for c, mk, col in zip(cols, "s^o", ("tab:blue", "tab:orange", "tab:red")):
        n = int(c.split("_n")[-1])
        ax.semilogy(d, data[c], mk, ms=4.5, mfc="none", color=col, label=rf"$n={n}$, $\varepsilon={L / n:.3f}$")
    x = np.linspace(0, 1.0, 50)
    ax.semilogy(x, np.exp(logc - gamma * x), "-", color="k", lw=1,
                label=rf"fit $\propto e^{{-\gamma d/\varepsilon}}$, $\gamma={gamma:.2f}$")
    ax.set(xlabel=r"distance to the boundary $d/\varepsilon$",
           ylabel=r"RMS of $\left(u_{\rm DNS}-(x_1+\varepsilon\chi_1)\right)/\varepsilon$",
           title="(b) the layer profiles collapse in the variable $d/\\varepsilon$",
           ylim=(5e-7, 2e-2), xlim=(0, d.max() + 0.05))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "boundary_layer_profile.png", dpi=170)
    print(f"Figure written to {FIGURES / 'boundary_layer_profile.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/")
    args = ap.parse_args()
    plot() if args.plot_only else main()
