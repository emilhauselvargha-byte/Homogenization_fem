"""Plot the unit-cell solution for a circular and an elliptical inclusion.

For a unit macroscopic gradient along x, shows the local temperature t_x = x + chi_x
(filled contours) and the heat flux k (e_x + grad chi_x) (arrows). The flux is tangent to
the hole wall -- the zero-flux condition -- and is strongest where it must squeeze past
the hole.

Unit-cell problems use P2 elements on curved meshes; the fields are drawn on the mesh vertices.

Output: figures/cell_fields.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from homogenization.geometry import make_unit_cell
from homogenization.solver import solve_corrector, compute_effective_tensor

FIGURES = Path(__file__).resolve().parents[1] / "figures"

K_SOLID = 1.0
CASES = [
    ("circle  r = 0.3", dict(hole_radius=0.3, hole_radius_b=None, hole_rotation_deg=0.0)),
    ("ellipse  a = 0.4, b = 0.15, $55^\\circ$", dict(hole_radius=0.4, hole_radius_b=0.15, hole_rotation_deg=55.0)),
]


def solve_case(kwargs):
    mesh, _, facets = make_unit_cell(mesh_ratio=150, **kwargs)
    tx = solve_corrector(mesh, facets, "x", k_value=K_SOLID)
    ty = solve_corrector(mesh, facets, "y", k_value=K_SOLID)
    K = compute_effective_tensor(mesh, tx, ty, k_value=K_SOLID)
    return mesh, tx, K


def plot_case(ax, mesh, tx, K, title):
    # Plot on the mesh vertices: the P2 corrector is interpolated into P1 on the same mesh,
    # which matplotlib can draw as a plain triangulation.
    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    tx1 = fem.Function(V1); tx1.interpolate(tx)
    xy = V1.tabulate_dof_coordinates()[:, :2]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], V1.dofmap.list)
    t = xy[:, 0] + tx1.x.array                       # t_x = x + chi_x
    cs = ax.tricontourf(tri, t, levels=30, cmap="coolwarm")

    # flux on DG0, drawn at cell midpoints (thinned out for legibility)
    W = fem.functionspace(mesh, ("DG", 0, (2,)))
    e_x = fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0)))
    expr = fem.Expression(K_SOLID * (e_x + ufl.grad(tx)), W.element.interpolation_points)
    flux = fem.Function(W); flux.interpolate(expr)
    mid = xy[V1.dofmap.list].mean(axis=1)
    F = flux.x.array.reshape(-1, 2)
    keep = np.arange(0, len(mid), max(1, len(mid) // 400))
    ax.quiver(mid[keep, 0], mid[keep, 1], F[keep, 0], F[keep, 1], color="k", scale=25, width=0.003)

    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{title}\n$\\kappa^*_{{11}}$={K[0,0]:.4f}, $\\kappa^*_{{22}}$={K[1,1]:.4f}, "
                 f"$\\kappa^*_{{12}}$={K[0,1]:.4f}", fontsize=10)
    return cs


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (title, kw) in zip(axes, CASES):
        mesh, tx, K = solve_case(kw)
        print(title, "\n", K)
        cs = plot_case(ax, mesh, tx, K, title)
    fig.colorbar(cs, ax=axes, shrink=0.8, label=r"$t_x = x + \chi_x$")
    fig.savefig(FIGURES / "cell_fields.png", dpi=150, bbox_inches="tight")
    print(f"Figure written to {FIGURES}")
