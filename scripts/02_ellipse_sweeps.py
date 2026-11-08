"""How inclusion shape shapes K*: aspect-ratio sweep and rotation sweep.

Only unit-cell problems are solved (P2 on curved meshes, no DNS), so this is cheap.

    Part 1: K*_11, K*_22 vs. aspect ratio b/a at CONSTANT hole area a*b (constant porosity),
            which isolates the effect of shape: K*_11 rises, K*_22 falls as the ellipse
            is stretched along x. (At fixed a instead, a thinner ellipse is also a smaller
            hole, and both entries rise because the porosity drops.)
    Part 2: K*_11, K*_22, K*_12 vs. rotation angle of a fixed elongated ellipse

Outputs: figures/ellipse_aspect_sweep.png, figures/ellipse_rotation_sweep.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homogenization.geometry import make_unit_cell
from homogenization.solver import solve_corrector, compute_effective_tensor

FIGURES = Path(__file__).resolve().parents[1] / "figures"

K_SOLID = 1.0
A = 0.3                                    # semi-axis a for the rotation sweep
AREA_PRODUCT = 0.06                        # a*b for the constant-porosity sweep
AREA_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]   # b/a for the constant-porosity sweep (a stays < 0.5)
FIXED_ASPECT = 0.3                         # for the rotation sweep
ROTATIONS_DEG = list(range(0, 91, 10))
MESH_RATIO = 80


def kstar(b, angle, a=A):
    mesh, _, facets = make_unit_cell(hole_radius=a, hole_radius_b=b,
                                     hole_rotation_deg=angle, mesh_ratio=MESH_RATIO)
    tx = solve_corrector(mesh, facets, "x", k_value=K_SOLID)
    ty = solve_corrector(mesh, facets, "y", k_value=K_SOLID)
    return compute_effective_tensor(mesh, tx, ty, k_value=K_SOLID)


def aspect_sweep():
    k11, k22 = [], []
    print(f"\n=== K* vs aspect ratio at constant hole area (a*b = {AREA_PRODUCT}) ===")
    for r in AREA_RATIOS:
        a = (AREA_PRODUCT / r) ** 0.5
        K = kstar(a * r, 0.0, a=a)
        k11.append(K[0, 0]); k22.append(K[1, 1])
        print(f"  b/a={r:.2f} (a={a:.3f}): K11={K[0,0]:.5f}  K22={K[1,1]:.5f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(AREA_RATIOS, k11, "o-", label=r"$\kappa^*_{11}$ (along major axis)")
    ax.plot(AREA_RATIOS, k22, "s-", label=r"$\kappa^*_{22}$ (along minor axis)")
    ax.set(xlabel="aspect ratio $b/a$ (1.0 = circle)", ylabel="effective conductivity",
           title=f"Anisotropy at constant porosity ($ab$ = {AREA_PRODUCT})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGURES / "ellipse_aspect_sweep.png", dpi=150)


def rotation_sweep():
    k11, k22, k12 = [], [], []
    print("\n=== K* vs rotation angle (b/a fixed) ===")
    for ang in ROTATIONS_DEG:
        K = kstar(A * FIXED_ASPECT, ang)
        k11.append(K[0, 0]); k22.append(K[1, 1]); k12.append(K[0, 1])
        print(f"  {ang:3d} deg: K11={K[0,0]:.5f}  K22={K[1,1]:.5f}  K12={K[0,1]:.5f}  "
              f"|K12-K21|={abs(K[0,1]-K[1,0]):.2e}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ROTATIONS_DEG, k11, "o-", label=r"$\kappa^*_{11}$")
    ax.plot(ROTATIONS_DEG, k22, "s-", label=r"$\kappa^*_{22}$")
    ax.plot(ROTATIONS_DEG, k12, "^-", label=r"$\kappa^*_{12}=\kappa^*_{21}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="rotation angle (degrees)", ylabel="tensor entries",
           title=f"$\\kappa^*$ vs. ellipse rotation ($b/a$ = {FIXED_ASPECT})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGURES / "ellipse_rotation_sweep.png", dpi=150)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    aspect_sweep()
    rotation_sweep()
    print(f"\nFigures written to {FIGURES}")
