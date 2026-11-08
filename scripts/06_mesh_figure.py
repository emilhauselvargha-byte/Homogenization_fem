"""Render the DNS mesh exactly as it is used in the convergence studies.

The domain is the 3 x 3 exact tiling of the rotated-ellipse unit cell (a = 0.3, b = 0.12,
40 degrees) on [0, L]^2 with L = 0.8, meshed at the finest DNS resolution of
04_homogenization_error.py (element size eps/60) with second-order, curved triangles.
Panel (a) shows the whole plate, panel (b) a zoom on one hole tip, where the edges of the
6-node triangles are drawn as the quadratic curves through their mid-edge nodes.

Only gmsh and matplotlib are needed (no FEniCSx solve); the geometry is built by the same
function as the DNS meshes.

Usage:
    python scripts/06_mesh_figure.py
Output: figures/perforated_rectangle_mesh.pdf
"""
from pathlib import Path

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from homogenization.geometry import _make_periodic_tiling_geometry

FIGURES = Path(__file__).resolve().parents[1] / "figures"

A, B, THETA = 0.3, 0.12, 40.0     # same ellipse as 04_homogenization_error.py
L, N = 0.8, 3                     # 3 x 3 cells
DNS_RATIO = 60                    # finest DNS level: element size eps/60
BLUE, RED = "#1f4e79", "#c0392b"


def build_mesh(eps):
    """Second-order mesh of the tiling; returns node coordinates and 6-node triangles."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("tiling")
    _make_periodic_tiling_geometry(eps, N, N, A, hole_radius_ratio_b=B, hole_rotation_deg=THETA)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", eps / DNS_RATIO)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", eps / DNS_RATIO)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    X = np.zeros((int(tags.max()) + 1, 2))
    X[tags.astype(int)] = coords.reshape(-1, 3)[:, :2]
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tri = enodes[list(etypes).index(9)].astype(int).reshape(-1, 6)   # gmsh type 9: 6-node triangle
    gmsh.finalize()
    return X, tri


def curved_edges(X, tri, n_pts=9):
    """Unique edges (vertex, vertex, mid-edge node) and their quadratic curves."""
    E = {}
    for c in tri:
        for p, q, m in ((c[0], c[1], c[3]), (c[1], c[2], c[4]), (c[2], c[0], c[5])):
            E[(min(p, q), max(p, q))] = m
    edges = np.array([(p, q, m) for (p, q), m in E.items()])
    t = np.linspace(0, 1, n_pts)[None, :, None]
    curves = ((1 - t) * (1 - 2 * t) * X[edges[:, 0]][:, None] + t * (2 * t - 1) * X[edges[:, 1]][:, None]
              + 4 * t * (1 - t) * X[edges[:, 2]][:, None])
    return edges, curves


def draw_holes(ax, eps):
    s = np.linspace(0, 2 * np.pi, 400)
    c, d = np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))
    for i in range(N):
        for j in range(N):
            px, py = A * eps * np.cos(s), B * eps * np.sin(s)
            ax.fill(c * px - d * py + (i + .5) * eps, d * px + c * py + (j + .5) * eps,
                    color="0.82", zorder=0)


def main():
    eps = L / N
    X, tri = build_mesh(eps)
    edges, curves = curved_edges(X, tri)
    print(f"{len(tri)} curved P2 triangles, {len(np.unique(tri))} nodes")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.9))
    ax = axs[0]
    draw_holes(ax, eps)
    ax.add_collection(LineCollection(curves, colors=BLUE, linewidths=0.12, rasterized=True))
    for k in range(1, N):
        ax.plot([k * eps] * 2, [0, L], ls=(0, (4, 4)), color="0.35", lw=0.6)
        ax.plot([0, L], [k * eps] * 2, ls=(0, (4, 4)), color="0.35", lw=0.6)
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="k", lw=0.8)
    centre = np.array([1.5 * eps, 1.5 * eps])
    tip = centre + A * eps * np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    w = 0.11 * eps
    x0, y0 = tip - w / 2
    ax.add_patch(Rectangle((x0, y0), w, w, fill=False, ec=RED, lw=1.2, zorder=5))
    ax.set(aspect="equal", xlim=(-0.01, L + 0.01), ylim=(-0.01, L + 0.01), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_title(rf"(a) $3\times3$ tiling, element size $\varepsilon/{DNS_RATIO}$", fontsize=11)

    ax = axs[1]
    draw_holes(ax, eps)
    near = np.all((curves[:, :, 0] > x0 - w) & (curves[:, :, 0] < x0 + 2 * w)
                  & (curves[:, :, 1] > y0 - w) & (curves[:, :, 1] < y0 + 2 * w), axis=1)
    ax.add_collection(LineCollection(curves[near], colors=BLUE, linewidths=0.9))
    ax.plot(*X[np.unique(edges[near][:, :2])].T, "o", ms=3.2, color=BLUE, zorder=3)
    ax.plot(*X[edges[near][:, 2]].T, "o", ms=3.2, mfc="white", mec=BLUE, mew=0.9, zorder=3)
    ax.set(aspect="equal", xlim=(x0, x0 + w), ylim=(y0, y0 + w), xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_edgecolor(RED); s.set_linewidth(1.2)
    ax.set_title("(b) zoom on a hole tip: curved P2 elements", fontsize=11)
    ax.plot([], [], "o", color=BLUE, ms=4, label="vertices")
    ax.plot([], [], "o", mfc="white", mec=BLUE, ms=4, label="mid-edge nodes")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "perforated_rectangle_mesh.pdf", dpi=300)
    print(f"Figure written to {FIGURES / 'perforated_rectangle_mesh.pdf'}")


if __name__ == "__main__":
    main()
