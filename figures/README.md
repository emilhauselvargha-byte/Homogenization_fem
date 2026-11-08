# Figures

Regenerate the plots with the scripts in `../scripts` (see the main README). Generated here:

| File | Script |
|---|---|
| `ellipse_aspect_sweep.png`, `ellipse_rotation_sweep.png` | `02_ellipse_sweeps.py` |
| `cell_fields.png` | `03_cell_fields.py` |
| `homogenization_error.png` | `04_homogenization_error.py` |
| `tensor_convergence.png` | `05_tensor_convergence.py` |
| `perforated_rectangle_mesh.pdf` | `06_mesh_figure.py` (needs gmsh only) |
| `boundary_layer_profile.png` | `07_boundary_layer.py` |

## Boundary-condition diagrams

`bc_dns.pdf`, `bc_cell.pdf` and `boundary_layer.pdf` are TikZ drawings whose sources are in
`diagrams/` (shared colours in `diagrams/diagram_style.tex`). Rebuild them with `make diagrams`,
or by hand with `pdflatex` in `diagrams/` and copying the PDF here.

## Figures not produced by the scripts

The LaTeX report also includes four renderings made in gmsh and ParaView. Put them here under
these names so that the report compiles:

| File | Report figure |
|---|---|
| `perforated_cube_mesh.pdf` | perforated cube (gmsh) |
| `cube_temperature_slice.pdf` | slice of the 3D DNS temperature field |
| `temperature_circle.pdf` | circular perforations, temperature and flux |
| `temperature_ellipse.pdf` | elliptical perforations, temperature and flux |

The cube mesh and its temperature slice use first-order (straight-edged) elements; they only
illustrate a 3D extension. All 2D computations use second-order, curved elements, as shown in
`perforated_rectangle_mesh.pdf`.
