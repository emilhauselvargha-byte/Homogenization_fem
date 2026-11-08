"""First-order periodic homogenization of steady heat conduction in perforated media.

Modules
-------
geometry    gmsh -> dolfinx meshes: perforated plates, the periodic unit cell, exact periodic tilings
solver      FEM solvers (P2 on curved meshes by default): DNS, apparent tensor of a perforated
            sample, periodic corrector problems, effective tensor, macro solve, reconstruction
validation  DNS-vs-homogenization error metrics used by the convergence studies
"""
