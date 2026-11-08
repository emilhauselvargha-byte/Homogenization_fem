"""Sanity checks on the cell problem and the effective tensor (P2 on curved meshes by default).

Each test solves a few small FEM problems (a few seconds in total).
"""
import numpy as np
import pytest

from homogenization.geometry import make_unit_cell, make_periodic_tiling
from homogenization.solver import solve_corrector, compute_effective_tensor, apparent_tensor

pytestmark = pytest.mark.slow

K_CIRCLE_03 = 0.558486   # circle r = 0.3, converged to ~1e-6 (P2, curved, mesh_ratio 160)


def cell(a, b=None, angle=0.0, mesh_ratio=60, **kw):
    mesh, _, facets = make_unit_cell(hole_radius=a, hole_radius_b=b,
                                     hole_rotation_deg=angle, mesh_ratio=mesh_ratio, **kw)
    deg = kw.get("geometry_order", 2)
    tx = solve_corrector(mesh, facets, "x", degree=deg)
    ty = solve_corrector(mesh, facets, "y", degree=deg)
    return mesh, tx, ty


def kstar(*args, **kw):
    return compute_effective_tensor(*cell(*args, **kw))


def test_corrector_is_periodic_including_corners():
    # the (1,1) corner is the one that both periodicity constraints touch
    _, tx, _ = cell(0.3, 0.12, angle=40.0, mesh_ratio=40)
    X = tx.function_space.tabulate_dof_coordinates()
    for ax in (0, 1):
        o = 1 - ax
        lo = np.where(np.isclose(X[:, ax], 0.0))[0]
        hi = np.where(np.isclose(X[:, ax], 1.0))[0]
        lo, hi = lo[np.argsort(X[lo, o])], hi[np.argsort(X[hi, o])]
        np.testing.assert_allclose(X[lo, o], X[hi, o], atol=1e-12)
        np.testing.assert_allclose(tx.x.array[lo], tx.x.array[hi], atol=1e-10)


def test_circle_is_isotropic_and_diagonal():
    K = kstar(0.3)
    assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-5)
    assert abs(K[0, 1]) < 1e-5


def test_circle_matches_reference_value():
    assert kstar(0.3)[0, 0] == pytest.approx(K_CIRCLE_03, abs=2e-6)


def test_tensor_is_symmetric_for_rotated_ellipse():
    K = kstar(0.4, 0.15, angle=55.0)
    assert abs(K[0, 1] - K[1, 0]) < 1e-10


def test_elongating_at_constant_porosity_raises_k11_lowers_k22():
    # same hole area a*b = 0.06; a circle (r = sqrt(0.06)) vs an ellipse stretched along x
    K_round = kstar(0.06 ** 0.5)
    K_thin = kstar(0.4, 0.15)
    assert K_thin[0, 0] > K_round[0, 0]
    assert K_thin[1, 1] < K_round[1, 1]


def test_smaller_hole_conducts_better():
    assert kstar(0.2)[0, 0] > kstar(0.3)[0, 0]


def test_off_diagonal_vanishes_at_axis_aligned_angles_and_peaks_between():
    assert abs(kstar(0.3, 0.09, angle=0.0)[0, 1]) < 1e-6   # unstructured-mesh noise only
    assert kstar(0.3, 0.09, angle=45.0)[0, 1] > 1e-2


def test_p2_curved_mesh_is_far_more_accurate_than_p1():
    k_p1 = kstar(0.3, mesh_ratio=40, geometry_order=1)[0, 0]
    k_p2 = kstar(0.3, mesh_ratio=40, geometry_order=2)[0, 0]
    assert abs(k_p2 - K_CIRCLE_03) < 1e-2 * abs(k_p1 - K_CIRCLE_03)   # ~500x in practice


def test_apparent_tensor_bounds_kstar_from_above():
    # affine boundary conditions on a 2 x 2 sample: K_app - K* is positive semi-definite
    a, b, th = 0.3, 0.12, 40.0
    Ks = kstar(a, b, angle=th, mesh_ratio=80)
    mesh, _, facets = make_periodic_tiling(epsilon=0.4, n_cells_x=2, n_cells_y=2,
                                           hole_radius_ratio=a, hole_radius_ratio_b=b,
                                           hole_rotation_deg=th, mesh_ratio=20)
    Ka = apparent_tensor(mesh, facets, rtol=1e-10)
    assert abs(Ka[0, 1] - Ka[1, 0]) < 1e-8
    assert np.linalg.eigvalsh(Ka - Ks).min() > 0
