"""FEM solvers: DNS heat solve, periodic corrector problems, effective tensor, macro solve, reconstruction."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl
from dolfinx.io import XDMFFile

# dolfinx has no built-in periodic BC support; dolfinx_mpc (github.com/jorgensd/dolfinx_mpc)
# provides it via a MultiPointConstraint. Only needed for solve_corrector below.
import dolfinx_mpc

def _ksp_options(rtol=None):
    """CG + hypre; pass rtol to tighten the linear-solver tolerance (PETSc default is 1e-5)."""
    opts = {"ksp_type": "cg", "pc_type": "hypre"}
    if rtol is not None:
        opts["ksp_rtol"] = rtol
    return opts


def _write_xdmf(func, filename):
    """Write a Function to XDMF. XDMF needs the Function's Lagrange degree to match the mesh
    geometry degree (1 for the plain macro mesh, 2 for curved meshes), so a scalar Lagrange
    Function of another degree is interpolated first. Cell-wise (DG0) data is written as is."""
    mesh = func.function_space.mesh
    el = func.function_space.element.basix_element
    gdeg = mesh.geometry.cmap.degree
    if el.discontinuous is False and el.degree != gdeg:
        out = fem.Function(fem.functionspace(mesh, ("Lagrange", gdeg)))
        out.interpolate(func)
        out.name = func.name
        func = out
    with XDMFFile(mesh.comm, filename, "w") as f:
        f.write_mesh(mesh)
        f.write_function(func)


def solve_heat(mesh, facet_tags, k_value=1.0, output_file=None, cross_section_length=None, rtol=None, degree=2):
    """
    DNS: steady-state heat equation on the fully resolved (perforated) mesh.

    Dirichlet BCs: u = 0 on x_min (facet tag 1), u = 1 on x_max (facet tag 2).
    Every other boundary (top, bottom, hole walls) is insulated (natural BC).

    Parameters:
        mesh, facet_tags     : from geometry.make_periodic_tiling or make_perforated_rectangle
        k_value              : thermal conductivity of the solid (float)
        rtol                 : linear-solver relative tolerance (default: PETSc's 1e-5)
        degree               : Lagrange degree (default 2, for curved geometry_order=2 meshes)
        output_file          : optional XDMF path for ParaView, e.g. "temperature.xdmf"
        cross_section_length : if given (pass Ly), k_eff = (total x-flux) / Ly, the apparent
            conductivity along x under the imposed gradient 1/Lx. It equals K*_11 only when the
            homogenized solution is the linear profile x/Lx (e.g. a centred circular hole); use
            apparent_tensor for a general comparison with K*. If None, the x-flux is divided by
            the meshed (solid) area instead.

    Returns:
        uh    : dolfinx Function, the temperature field
        k_eff : float, see cross_section_length
    """

    # ── Function space ─────────────────────────────────────────────────────────
    V = fem.functionspace(mesh, ("Lagrange", degree))

    # ── Dirichlet BCs ──────────────────────────────────────────────────────────
    facets_xmin = facet_tags.find(1)
    facets_xmax = facet_tags.find(2)

    dofs_xmin = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets_xmin)
    dofs_xmax = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets_xmax)

    bc_xmin = fem.dirichletbc(fem.Constant(mesh, 0.0), dofs_xmin, V)
    bc_xmax = fem.dirichletbc(fem.Constant(mesh, 1.0), dofs_xmax, V)

    # ── Variational formulation ────────────────────────────────────────────────
    k = fem.Constant(mesh, k_value)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(mesh, 0.0) * v * ufl.dx   

    # ── Solve ──────────────────────────────────────────────────────────────────
    problem = LinearProblem(
        a, L, bcs=[bc_xmin, bc_xmax],
        petsc_options=_ksp_options(rtol),
        petsc_options_prefix="heat_",
    )
    uh = problem.solve()
    uh.name = "temperature"

    # ── Apparent conductivity along x ──────────────────────────────────────────
    flux_x_form = fem.form(k * uh.dx(0) * ufl.dx)
    flux_x = fem.assemble_scalar(flux_x_form)
    domain_volume = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx))
    norm = cross_section_length if cross_section_length is not None else domain_volume
    k_eff = flux_x / norm

    # ── Optional VTK export ────────────────────────────────────────────────────
    if output_file is not None:
        _write_xdmf(uh, output_file)

    return uh, k_eff


def apparent_tensor(mesh, facet_tags, k_value=1.0, rtol=None, degree=2):
    """
    Measure the full 2x2 conductivity tensor of a perforated sample by DNS.

    Two solves on the fully resolved perforated domain Omega, one per direction j, with the
    affine Dirichlet condition u = x_j imposed on the WHOLE outer boundary (facet tags 1-4;
    the hole walls stay insulated). The apparent tensor is the volume-averaged flux

        K_app[i, j] = (1/|Omega|) * integral_{Omega_solid} k * d(u_j)/d(x_i) dx,

    with |Omega| the full domain area (solid + holes), the same normalization as K*.
    Because the imposed boundary data has mean gradient e_j, K_app is directly comparable
    with K*: for a domain made of whole periodic cells it converges to K* as eps -> 0
    (boundary layer of width ~eps, so the difference is O(eps)), and K_app >= K* in the
    sense of quadratic forms (the affine-BC fluctuation space is a subspace of the periodic one).

    Returns
    -------
    K_app : 2x2 numpy array
    """
    V = fem.functionspace(mesh, ("Lagrange", degree))
    fdim = mesh.topology.dim - 1
    outer = np.concatenate([facet_tags.find(t) for t in (1, 2, 3, 4)])
    dofs = fem.locate_dofs_topological(V, fdim, outer)

    # bounding-box area of the sample (serial runs; the outer boundary is a rectangle)
    xg = mesh.geometry.x[:, :2]
    area = float(np.prod(xg.max(axis=0) - xg.min(axis=0)))

    k = fem.Constant(mesh, PETSc.ScalarType(k_value))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

    K_app = np.zeros((2, 2))
    for j in range(2):
        g = fem.Function(V)
        g.interpolate(lambda X, j=j: X[j])
        bc = fem.dirichletbc(g, dofs)
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=_ksp_options(rtol),
                                petsc_options_prefix=f"kubc{j}_")
        uh = problem.solve()
        for i in range(2):
            K_app[i, j] = fem.assemble_scalar(fem.form(k * uh.dx(i) * ufl.dx)) / area
    return K_app


# ═══════════════════════════════════════════════════════════════════════════
# CELL (CORRECTOR) PROBLEM
# ═══════════════════════════════════════════════════════════════════════════

def solve_corrector(mesh, facet_tags, direction, k_value=1.0, length=1.0, width=1.0, degree=2):
    """
    Solve the periodic unit-cell corrector problem for direction 'x' or 'y'.

    Weak form: find chi_j (Y-periodic) such that, for all periodic v,
        integral( k * grad(chi_j) . grad(v) ) dx = -integral( k * e_j . grad(v) ) dx
    i.e. chi_j is the periodic fluctuation induced by a unit macroscopic
    gradient e_j. The corrector is normalized to zero cell-average, which is
    the convention consistent with T being the average of the fine-scale field.

    Requires a mesh of [0,length]x[0,width] built with periodic=True (see
    geometry.make_unit_cell), so that opposite faces carry matching nodes. The
    hole wall is left as a natural (insulated) boundary. facet_tags is kept in
    the signature for compatibility; periodicity is imposed geometrically.

    Parameters:
        mesh, facet_tags : from geometry.make_unit_cell
        direction        : 'x' or 'y' -- which unit macroscopic gradient to impose
        k_value          : thermal conductivity of the solid phase
        length, width    : unit cell dimensions (must match what built the mesh)
        degree           : Lagrange degree (default 2; pair it with a geometry_order=2 mesh,
                           otherwise the straight-edged hole limits the accuracy)

    Returns:
        chi : dolfinx Function, the zero-mean periodic corrector (tx or ty)
    """
    V = fem.functionspace(mesh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = fem.Constant(mesh, PETSc.ScalarType(k_value))

    if direction == "x":
        e_j = fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0)))
    elif direction == "y":
        e_j = fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0)))
    else:
        raise ValueError("direction must be 'x' or 'y'")

    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = -k * ufl.inner(e_j, ufl.grad(v)) * ufl.dx

    # --- periodicity. Every point on the x_max / y_max faces is slaved to its image on the
    # x_min / y_min faces. The three corners (L,0), (0,W), (L,W) need a constraint of their
    # own: tying x_max to x_min and y_max to y_min separately would slave the (L,W) corner
    # twice (and to another slave), which dolfinx_mpc does not resolve -- the corrector then
    # jumps across the cell faces near that corner on some meshes. All corners map to (0,0).
    tol = 1e-8 * max(length, width)

    def on_x_max(x):   # x = L, corners excluded
        return np.isclose(x[0], length, atol=tol) & (x[1] > tol) & (x[1] < width - tol)

    def on_y_max(x):   # y = W, corners excluded
        return np.isclose(x[1], width, atol=tol) & (x[0] > tol) & (x[0] < length - tol)

    def slave_corners(x):
        at_x = np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], length, atol=tol)
        at_y = np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], width, atol=tol)
        origin = np.isclose(x[0], 0.0, atol=tol) & np.isclose(x[1], 0.0, atol=tol)
        return at_x & at_y & ~origin

    def x_shift(x):
        out = x.copy(); out[0] = x[0] - length; return out

    def y_shift(x):
        out = x.copy(); out[1] = x[1] - width; return out

    def to_origin(x):
        out = x.copy(); out[0] = 0.0; out[1] = 0.0; return out

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, on_x_max, x_shift, [])
    mpc.create_periodic_constraint_geometrical(V, on_y_max, y_shift, [])
    mpc.create_periodic_constraint_geometrical(V, slave_corners, to_origin, [])
    mpc.finalize()

    # --- pin one DOF to remove the constant null space (pure periodic problem
    # is singular up to an additive constant). We subtract the cell mean
    # afterward, so which point we pin doesn't matter.
    def origin(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    pin_dofs = fem.locate_dofs_geometrical(V, origin)
    bc_pin = fem.dirichletbc(PETSc.ScalarType(0.0), pin_dofs, V)

    # Direct solve (MUMPS). The cell problem is small, and CG + hypre on the
    # constrained system stalls well above its nominal tolerance, which shows up as
    # K*_12 != K*_21 at the 1e-4 level.
    problem = dolfinx_mpc.LinearProblem(
        a, L, mpc, bcs=[bc_pin],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
    )
    chi = problem.solve()
    chi.name = f"corrector_{direction}"

    # --- enforce zero cell-average
    solid_volume = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx))
    mean_chi = fem.assemble_scalar(fem.form(chi * ufl.dx)) / solid_volume
    chi.x.array[:] -= mean_chi

    return chi


def compute_effective_tensor(mesh, tx, ty, k_value=1.0, cell_area=1.0):
    """
    Assemble the effective conductivity tensor from the two correctors:

        K*_ij = (1/|Y|) * integral_Ysolid[ k * (delta_ij + d(chi_j)/d(x_i)) ] dx

    cell_area is the FULL unit cell area |Y| (solid + hole), NOT the solid
    volume -- this is the normalization that correctly reflects the porosity's
    effect on the macroscopic conductivity. For a unit square cell this is 1.0.

    Returns:
        Kstar : 2x2 numpy array
    """
    k = fem.Constant(mesh, PETSc.ScalarType(k_value))
    basis = {0: (1.0, 0.0), 1: (0.0, 1.0)}
    correctors = {0: tx, 1: ty}

    Kstar = np.zeros((2, 2))
    for j, chi_j in correctors.items():
        e_j = fem.Constant(mesh, PETSc.ScalarType(basis[j]))
        flux = k * (e_j + ufl.grad(chi_j))
        for i in range(2):
            Kstar[i, j] = fem.assemble_scalar(fem.form(flux[i] * ufl.dx)) / cell_area

    return Kstar


# ═══════════════════════════════════════════════════════════════════════════
# MACROSCOPIC PROBLEM
# ═══════════════════════════════════════════════════════════════════════════

def solve_macro(Kstar, Lx, Ly, nx, ny, T_left=0.0, T_right=1.0, rtol=None, degree=2):
    """
    Solve the homogenized macroscopic problem div(K* grad T) = 0 on a plain
    (hole-free) rectangle [0,Lx]x[0,Ly], with T=T_left on x=0, T=T_right on
    x=Lx, and insulated (natural) top/bottom -- mirrors the BC convention of
    solve_heat, but now with the anisotropic constant K* instead of scalar k.

    Parameters:
        Kstar    : 2x2 numpy array, effective conductivity tensor
        Lx, Ly   : physical size of the real (macroscopic) domain
        nx, ny   : coarse mesh resolution (no microstructure to resolve!)
        T_left, T_right : Dirichlet values on x=0 / x=Lx

    Returns:
        T_h : dolfinx Function, the macroscopic temperature field
    """
    from dolfinx import mesh as dmesh

    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [nx, ny],
        cell_type=dmesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    Kstar_const = fem.Constant(domain, PETSc.ScalarType(Kstar))

    a = ufl.inner(ufl.dot(Kstar_const, ufl.grad(T)), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], Lx)

    fdim = domain.topology.dim - 1
    dofs_left = fem.locate_dofs_geometrical(V, left)
    dofs_right = fem.locate_dofs_geometrical(V, right)
    bc_left = fem.dirichletbc(PETSc.ScalarType(T_left), dofs_left, V)
    bc_right = fem.dirichletbc(PETSc.ScalarType(T_right), dofs_right, V)

    problem = LinearProblem(
        a, L, bcs=[bc_left, bc_right],
        petsc_options=_ksp_options(rtol),
        petsc_options_prefix="macro_",
    )
    T_h = problem.solve()
    T_h.name = "T_macro"
    return T_h


# ═══════════════════════════════════════════════════════════════════════════
# FINE-SCALE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _eval_at_points(func, points):
    """Evaluate a dolfinx Function at an array of physical points, shape (N, 3)."""
    domain = func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    cells = []
    valid_points = []
    for i, point in enumerate(points):
        entities = colliding_cells.links(i)
        if len(entities) > 0:
            cells.append(entities[0])
            valid_points.append(point)
        else:
            cells.append(None)

    values = np.full((len(points), func.function_space.value_size), np.nan)
    for i, cell in enumerate(cells):
        if cell is not None:
            values[i] = func.eval(points[i:i + 1], np.array([cell]))
    return values


def reconstruct_fine_scale(T_macro, tx, ty, points, epsilon):
    """
    Reconstruct the fine-scale temperature t(x,y) at macroscopic points using
    the two-scale expansion:

        t(x,y) = T(x,y) + epsilon * [ tx({x/eps},{y/eps}) dT/dx
                                     + ty({x/eps},{y/eps}) dT/dy ]

    Parameters:
        T_macro : dolfinx Function on the coarse macro mesh (from solve_macro)
        tx, ty  : correctors on the unit cell (from solve_corrector)
        points  : (N,3) array of physical (x,y,0) points in the macro domain
        epsilon : the physical cell size

    Returns:
        t : (N,) array of reconstructed fine-scale temperatures
    """
    points = np.asarray(points, dtype=np.float64)

    # Macroscopic field and its gradient at the query points
    T_vals = _eval_at_points(T_macro, points).flatten()

    V_macro = T_macro.function_space
    domain = V_macro.mesh
    # grad T is exactly representable in DG(p-1) for T in Lagrange(p) on an affine mesh
    p = V_macro.element.basix_element.degree
    W = fem.functionspace(domain, ("DG", max(p - 1, 0), (domain.geometry.dim,)))
    grad_T = fem.Function(W)
    interp_pts = W.element.interpolation_points
    if callable(interp_pts):
        interp_pts = interp_pts()
    grad_expr = fem.Expression(ufl.grad(T_macro), interp_pts)
    grad_T.interpolate(grad_expr)
    grad_T_vals = _eval_at_points(grad_T, points)  # (N, 2)

    # Local unit-cell coordinates: fractional part of (x/eps, y/eps)
    xtilde = np.mod(points[:, 0] / epsilon, 1.0)
    ytilde = np.mod(points[:, 1] / epsilon, 1.0)
    cell_points = np.stack([xtilde, ytilde, np.zeros_like(xtilde)], axis=1)

    tx_vals = _eval_at_points(tx, cell_points).flatten()
    ty_vals = _eval_at_points(ty, cell_points).flatten()

    t = T_vals + epsilon * (tx_vals * grad_T_vals[:, 0] + ty_vals * grad_T_vals[:, 1])
    return t


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_macro_temperature(T_macro, filename="temperature_hom.xdmf"):
    """
    Write the homogenized macroscopic temperature field to an XDMF file
    (with companion .h5) for viewing in ParaView.

    Parameters:
        T_macro  : dolfinx Function, the macroscopic temperature (from solve_macro)
        filename : output path, e.g. "temperature_hom.xdmf"
    """
    _write_xdmf(T_macro, filename)
    print(f"Wrote homogenized temperature field to {filename}")


def export_correctors(mesh, tx, ty, k_value=1.0, prefix="corrector"):
    """
    Write the two unit-cell correctors, and their associated flux vector
    fields, to separate XDMF files for ParaView.

    Scalar files (color the corrector value itself):
        {prefix}_tx.xdmf, {prefix}_ty.xdmf

    Vector files (physically meaningful "heat flow" field -- use these for
    arrow/glyph plots, NOT a raw gradient of tx/ty, since the flux also
    includes the imposed unit macroscopic gradient e_j):
        flux_j = k * (e_j + grad(chi_j))
        {prefix}_flux_x.xdmf, {prefix}_flux_y.xdmf

    In ParaView: open a flux file, add a "Glyph" filter, set Glyph Type to
    Arrow, Orientation/Scale Array to the vector field, and (optionally)
    a "Calculator" filter with mag(flux) if you want to color by magnitude.

    Parameters:
        mesh     : the unit cell mesh (same one tx, ty live on)
        tx, ty   : correctors from solve_corrector
        k_value  : conductivity used in the corrector solve (must match)
        prefix   : filename prefix for the four output files
    """
    k = fem.Constant(mesh, PETSc.ScalarType(k_value))
    basis = {"x": (1.0, 0.0), "y": (0.0, 1.0)}
    correctors = {"x": tx, "y": ty}

    # --- scalar corrector fields
    for name, chi in correctors.items():
        fname = f"{prefix}_t{name}.xdmf"
        _write_xdmf(chi, fname)
        print(f"Wrote corrector t{name} to {fname}")

    # --- flux vector fields, projected onto a DG0 vector space for plotting
    W = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    for name, chi in correctors.items():
        e_j = fem.Constant(mesh, PETSc.ScalarType(basis[name]))
        flux_expr_ufl = k * (e_j + ufl.grad(chi))

        interp_pts = W.element.interpolation_points
        if callable(interp_pts):
            interp_pts = interp_pts()
        flux_expr = fem.Expression(flux_expr_ufl, interp_pts)

        flux = fem.Function(W)
        flux.name = f"flux_{name}"
        flux.interpolate(flux_expr)

        fname = f"{prefix}_flux_{name}.xdmf"
        _write_xdmf(flux, fname)
        print(f"Wrote flux_{name} vector field to {fname}")
