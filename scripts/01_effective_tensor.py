"""Demo of the full homogenization pipeline for one inclusion shape.

    1. solve the two periodic corrector problems on the unit cell
    2. assemble the effective conductivity tensor K*
    3. solve the homogenized macro problem div(K* grad T) = 0 on a hole-free plate
    4. reconstruct the fine-scale temperature at a few sample points
    5. (--dns) compare against a direct numerical solve on the perforated plate

Usage:
    python scripts/01_effective_tensor.py            # steps 1-4
    python scripts/01_effective_tensor.py --dns      # + DNS benchmark
All solves use P2 elements on curved meshes. Outputs (ParaView .xdmf) are written to ./output/.
"""
import argparse
from pathlib import Path

import numpy as np

from homogenization.geometry import make_unit_cell, make_periodic_tiling
from homogenization.solver import (
    solve_heat, solve_corrector, compute_effective_tensor, solve_macro,
    reconstruct_fine_scale, export_macro_temperature, export_correctors,
)
from homogenization.validation import field_errors_grid

# ── Parameters (lengths of the hole in unit-cell units) ───────────────────
K_SOLID = 1.0
HOLE_A = 0.4           # semi-axis along local x, before rotation
HOLE_B = 0.15          # semi-axis along local y; None -> circle of radius HOLE_A (needs B <= A)
HOLE_ROTATION_DEG = 55.0
N_CELLS = 10           # cells per side of the macroscopic domain
LX = LY = 1.0          # macroscopic domain
EPSILON = LX / N_CELLS
MACRO_NX = MACRO_NY = 40   # coarse P2 macro mesh (no microstructure resolved)

OUT = Path("output")


def main(run_dns):
    OUT.mkdir(exist_ok=True)

    # 1-2. cell problem + effective tensor
    cell_mesh, _, cell_facets = make_unit_cell(
        hole_radius=HOLE_A, hole_radius_b=HOLE_B,
        hole_rotation_deg=HOLE_ROTATION_DEG, mesh_ratio=150,
    )
    tx = solve_corrector(cell_mesh, cell_facets, "x", k_value=K_SOLID)
    ty = solve_corrector(cell_mesh, cell_facets, "y", k_value=K_SOLID)
    Kstar = compute_effective_tensor(cell_mesh, tx, ty, k_value=K_SOLID)
    print("Effective conductivity tensor K*:\n", Kstar)
    print(f"Symmetry check |K*_01 - K*_10| = {abs(Kstar[0, 1] - Kstar[1, 0]):.3e}")
    export_correctors(cell_mesh, tx, ty, k_value=K_SOLID, prefix=str(OUT / "corrector"))

    # 3. macro solve
    T_macro = solve_macro(Kstar, LX, LY, MACRO_NX, MACRO_NY, T_left=0.0, T_right=1.0)
    export_macro_temperature(T_macro, filename=str(OUT / "temperature_hom.xdmf"))

    # 4. reconstruction at a few points (these may fall inside a hole of the real
    #    material -- the reconstruction is defined everywhere on the unit cell)
    pts = np.array([[0.25, 0.5, 0.0], [0.5, 0.5, 0.0], [0.75, 0.5, 0.0]])
    for p, t in zip(pts, reconstruct_fine_scale(T_macro, tx, ty, pts, epsilon=EPSILON)):
        print(f"u({p[0]:.2f}, {p[1]:.2f}) ~ {t:.6f}")

    # 5. optional DNS benchmark on the exactly-matching periodic tiling
    if run_dns:
        dns_mesh, _, dns_facets = make_periodic_tiling(
            epsilon=EPSILON, n_cells_x=N_CELLS, n_cells_y=N_CELLS,
            hole_radius_ratio=HOLE_A, hole_radius_ratio_b=HOLE_B,
            hole_rotation_deg=HOLE_ROTATION_DEG, mesh_ratio=20,
        )
        uh, _ = solve_heat(dns_mesh, dns_facets, k_value=K_SOLID,
                           output_file=str(OUT / "temperature_dns.xdmf"))
        e_T, e_2s = field_errors_grid(uh, T_macro, tx, ty, EPSILON, LX, LY)
        print(f"DNS vs. homogenization (RMS over interior points):  "
              f"u - T: {e_T:.3e}   u - (T + eps chi.grad T): {e_2s:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", action="store_true", help="also run the DNS benchmark")
    main(ap.parse_args().dns)
