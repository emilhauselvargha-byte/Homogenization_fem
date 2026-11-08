"""Geometry and mesh construction (gmsh -> dolfinx) for perforated domains and the periodic unit cell."""
import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio



def _build_mesh(gdim, mesh_size=0, comm=MPI.COMM_WORLD, periodic=False, geometry_order=2):
    """
    Helper function to synchronize on a built geometry, mesh, and return it as a dolfinx mesh.

    geometry_order : 2 (default) for curved, second-order elements whose mid-edge nodes lie ON
        the curved hole boundary; 1 for straight-edged triangles. Curved elements are needed
        to benefit from P2 finite elements: with straight edges the polygonal hole carries an
        O(h^2) geometric error regardless of the FE degree.

    periodic : if True (2D only), sets gmsh periodicity constraints between
        x_min/x_max and y_min/y_max so the resulting mesh has matching node
        layouts on opposite faces -- required for periodic BC cell problems.
    """
    gmsh.model.occ.synchronize()
    
    # Tag the bulk volume
    entities = gmsh.model.getEntities(gdim)
    gmsh.model.addPhysicalGroup(gdim, [tag for _, tag in entities], tag=1)
    gmsh.model.setPhysicalName(gdim, 1, "matrix")

    if gdim == 3:
        _tag_boundaries_3d()
    else:
        _tag_boundaries_2d(periodic=periodic)

    if mesh_size:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4) if gdim == 3 else None
    gmsh.model.mesh.generate(gdim)
    if geometry_order > 1:
        gmsh.model.mesh.setOrder(geometry_order)
    

    result = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=gdim)
    mesh, cell_tags = result[0], result[1]
    facet_tags = result[2] if len(result) > 2 else None

    return mesh, cell_tags, facet_tags

def _tag_boundaries_2d(periodic=False):
    """
    Tag boundary surfaces by their normal direction (rectangle edges) or curvature (circle walls).
    Tags:
        1 = x_min edge
        2 = x_max edge
        3 = y_min edge
        4 = y_max edge
        5 = circle walls

    periodic : if True, additionally registers gmsh periodicity constraints
        x_max <- x_min and y_max <- y_min, so the mesher produces matching
        (conforming) node sets on opposite faces. Required for periodic-BC
        cell problems; irrelevant for plain Dirichlet/Neumann DNS solves.
    """
    edges = gmsh.model.getEntities(1)
    x_min, x_max = [], []
    y_min, y_max = [], []
    circle_walls = []

    for _, tag in edges:
        # Count boundary curves: flat edges have 2 curves, circle walls have 1 (the circle itself)
        # This is the most robust classifier, because it relies only on the geometry.
        curves = gmsh.model.getBoundary([(1, tag)], oriented=False)
        n_curves = len(curves)

        if n_curves == 0:
            # Circle wall: bounded only by the circle itself
            circle_walls.append(tag)
        else:
            # Flat edge: classify by which coordinate is at the boundary
            # Use a small tolerance
            eps = 1e-5
            bbox = gmsh.model.getBoundingBox(1, tag)
            # bbox = (xmin, ymin, zmin, xmax, ymax, zmax). It's the bounding box of the edge in 2D space.
            if abs(bbox[0] - bbox[3]) < eps:   # x is constant
                if bbox[0] < 0.5:
                    x_min.append(tag)
                else:
                    x_max.append(tag)
            elif abs(bbox[1] - bbox[4]) < eps:  # y is constant
                if bbox[1] < 0.5:
                    y_min.append(tag)
                else:
                    y_max.append(tag)

    groups = [
        (x_min, 1, "x_min"),
        (x_max, 2, "x_max"),
        (y_min, 3, "y_min"),
        (y_max, 4, "y_max"),
        (circle_walls, 5, "cylinder_walls"),
    ]
    for surfaces, tag, name in groups:
        if surfaces:
            gmsh.model.addPhysicalGroup(1, surfaces, tag=tag)
            gmsh.model.setPhysicalName(1, tag, name)

    if periodic:
        # Domain extent, read off the full model bounding box (holes are internal,
        # so they don't affect the outer bounding box).
        xmin_bb, ymin_bb, _, xmax_bb, ymax_bb, _ = gmsh.model.getBoundingBox(-1, -1)
        Lx = xmax_bb - xmin_bb
        Ly = ymax_bb - ymin_bb

        # Affine transform is a 4x4 row-major matrix acting on homogeneous coords,
        # mapping MASTER (x_min/y_min) -> SLAVE (x_max/y_max).
        if x_min and x_max:
            translation_x = [1, 0, 0, Lx,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, x_max, x_min, translation_x)
        if y_min and y_max:
            translation_y = [1, 0, 0, 0,
                              0, 1, 0, Ly,
                              0, 0, 1, 0,
                              0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, y_max, y_min, translation_y)

def _tag_boundaries_3d():
    """
    Tag boundary surfaces by their normal direction (cube faces) or curvature (cylinder walls).
    Tags:
        1 = x_min face
        2 = x_max face  
        3 = y_min face
        4 = y_max face
        5 = z_min face
        6 = z_max face
        7 = cylinder walls
    """
    surfaces = gmsh.model.getEntities(2)
    
    x_min, x_max = [], []
    y_min, y_max = [], []
    z_min, z_max = [], []
    cylinder_walls = []

    for _, tag in surfaces:

        # Count boundary curves: flat faces have 4 curves, cylinder walls have 2 (top/bottom circles)
        # This is the most robust classifier, because it relises only on the geometry.
        curves = gmsh.model.getBoundary([(2, tag)], oriented=False)
        n_curves = len(curves)

        if n_curves == 2:
            # Cylinder wall: bounded only by top and bottom circles
            cylinder_walls.append(tag)
        else:
            # Flat face: classify by which coordinate is at the boundary
            # Use a small tolerance
            eps = 1e-6
            bbox = gmsh.model.getBoundingBox(2, tag)
            # bbox = (xmin, ymin, zmin, xmax, ymax, zmax). It's the bounding box of the surface in 3D space.
            if abs(bbox[0] - bbox[3]) < eps:   # x is constant
                if bbox[0] < 0.5:
                    x_min.append(tag)
                else:
                    x_max.append(tag)
            elif abs(bbox[1] - bbox[4]) < eps:  # y is constant
                if bbox[1] < 0.5:
                    y_min.append(tag)
                else:
                    y_max.append(tag)
            elif abs(bbox[2] - bbox[5]) < eps:  # z is constant
                if bbox[2] < 0.5:
                    z_min.append(tag)
                else:
                    z_max.append(tag)

    groups = [
        (x_min, 1, "x_min"),
        (x_max, 2, "x_max"),
        (y_min, 3, "y_min"),
        (y_max, 4, "y_max"),
        (z_min, 5, "z_min"),
        (z_max, 6, "z_max"),
        (cylinder_walls, 7, "cylinder_walls"),
    ]
    for surfaces, tag, name in groups:
        if surfaces:
            gmsh.model.addPhysicalGroup(2, surfaces, tag=tag)
            gmsh.model.setPhysicalName(2, tag, name)


def _make_perforated_rectangle_geometry(length, width, num_holes_x, num_holes_y,
                                         hole_radius=0, hole_radius_b=None,
                                         hole_rotation_deg=0.0):
    """
    Create a perforated rectangle gmsh geometry with specified dimensions and hole parameters.

    Parameters:
    length (float): Length of the rectangle.
    width (float): Width of the rectangle.
    hole_radius (float): Semi-axis of the hole along its local x-axis
        (before rotation). Circular hole of this radius if hole_radius_b
        is left as None.
    hole_radius_b (float or None): Semi-axis along the local y-axis. Set
        different from hole_radius for an ellipse (e.g. hole_radius_b <
        hole_radius gives a "long and thin" ellipse aligned with x).
    hole_rotation_deg (float): Rotation of each ellipse about its own
        center, degrees counter-clockwise. No effect for circular holes.
    num_holes_x (int): Number of holes along the x-axis.
    num_holes_y (int): Number of holes along the y-axis.
    """
    # Create the rectangle
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, length, width)

    # Create holes
    hole_spacing_x = length / (num_holes_x)
    hole_spacing_y = width / (num_holes_y)
    if hole_radius == 0:
        radius_a = min(hole_spacing_x, hole_spacing_y) / 4
    else:
        radius_a = hole_radius
    radius_b = radius_a if hole_radius_b is None else hole_radius_b

    holes = []
    for i in range(1, num_holes_x + 1):
        for j in range(1, num_holes_y + 1):
            x = (i - 0.5) * hole_spacing_x
            y = (j - 0.5) * hole_spacing_y
            hole = gmsh.model.occ.addDisk(x, y, 0, radius_a, radius_b)
            if hole_rotation_deg:
                gmsh.model.occ.rotate([(2, hole)], x, y, 0, 0, 0, 1,
                                       np.deg2rad(hole_rotation_deg))
            holes.append(hole)

    # Cut the holes from the rectangle
    gmsh.model.occ.cut([(2,rectangle)], [(2, hole) for hole in holes])




def make_perforated_rectangle(
    length,
    width,
    nx,
    ny,
    hole_radius=0,
    hole_radius_b=None,
    hole_rotation_deg=0.0,
    mesh_ratio=30,
    comm=MPI.COMM_WORLD,
    periodic=False,
    geometry_order=2,
):
    """
    Create a perforated rectangle mesh with specified dimensions and hole parameters, outputted in dolfinx format.
    Mesh size is determined by the mesh_ratio parameter, which controls the number of elements along the diagonal of the rectangle.

    hole_radius, hole_radius_b, hole_rotation_deg : see
        _make_perforated_rectangle_geometry. Leave hole_radius_b=None for
        circular holes; set it different from hole_radius for an ellipse.

    periodic : set True when this mesh will be used for a periodic-BC cell
        problem (opposite faces get matching node layouts). Leave False for
        plain DNS solves with fixed Dirichlet BCs .
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)   # errors and warnings only
    gmsh.model.add("rectangle")

    _make_perforated_rectangle_geometry(
        length,
        width,
        nx,
        ny,
        hole_radius=hole_radius,
        hole_radius_b=hole_radius_b,
        hole_rotation_deg=hole_rotation_deg,
    )
    mesh_size = (length+width/2)/mesh_ratio
    mesh = _build_mesh(
        gdim=2,
        mesh_size=mesh_size,
        comm=comm,
        periodic=periodic,
        geometry_order=geometry_order,
    )

    gmsh.finalize()

    return mesh


def make_unit_cell(hole_radius=0.2, hole_radius_b=None, hole_rotation_deg=0.0,
                    mesh_ratio=30, comm=MPI.COMM_WORLD, geometry_order=2):
    """
    Build the single-inclusion, unit-square periodic cell used for the
    corrector (cell) problems: a [0,1]x[0,1] square with one centered
    hole, with periodicity constraints registered between x_min/x_max and
    y_min/y_max so the mesh is conforming across the cell boundary
    (required by the periodic BC solve in solver.py).

    hole_radius       : semi-axis along local x (before rotation). Circular
        hole of this radius if hole_radius_b is left as None.
    hole_radius_b      : semi-axis along local y. Set < hole_radius for a
        "long and thin" ellipse aligned with x, or > hole_radius for one
        aligned with y.
    hole_rotation_deg  : rotate the ellipse about its center, degrees CCW.
        No effect when the hole is circular.

    geometry_order : 2 builds curved elements (use with P2 finite elements).

    Returns (mesh, cell_tags, facet_tags), same convention as
    make_perforated_rectangle. Facet tags: 1=x_min, 2=x_max, 3=y_min,
    4=y_max, 5=hole wall.
    """
    return make_perforated_rectangle(
        length=1.0, width=1.0,
        nx=1, ny=1,
        hole_radius=hole_radius,
        hole_radius_b=hole_radius_b,
        hole_rotation_deg=hole_rotation_deg,
        mesh_ratio=mesh_ratio,
        comm=comm,
        periodic=True,
        geometry_order=geometry_order,
    )


def _make_periodic_tiling_geometry(epsilon, n_cells_x, n_cells_y, hole_radius_ratio,
                                    hole_radius_ratio_b=None, hole_rotation_deg=0.0):
    """
    Build n_cells_x x n_cells_y exact copies of the unit cell tiled edge to
    edge -- hole centered in every epsilon x epsilon cell, no margins. This
    is the DNS geometry that actually matches what the cell problem assumes,
    unlike make_perforated_rectangle's evenly-spaced-with-margins layout.

    hole_radius_ratio, hole_radius_ratio_b, hole_rotation_deg : same
        ellipse controls as make_unit_cell, but as ratios of epsilon
        (physical semi-axis = ratio * epsilon) rather than of a unit cell.
        MUST match the values passed to make_unit_cell for the DNS and the
        cell problem to represent the same physical material.
    """
    length = n_cells_x * epsilon
    width = n_cells_y * epsilon
    radius_a = hole_radius_ratio * epsilon
    radius_b = radius_a if hole_radius_ratio_b is None else hole_radius_ratio_b * epsilon

    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, length, width)

    holes = []
    for i in range(n_cells_x):
        for j in range(n_cells_y):
            x = (i + 0.5) * epsilon
            y = (j + 0.5) * epsilon
            hole = gmsh.model.occ.addDisk(x, y, 0, radius_a, radius_b)
            if hole_rotation_deg:
                gmsh.model.occ.rotate([(2, hole)], x, y, 0, 0, 0, 1,
                                       np.deg2rad(hole_rotation_deg))
            holes.append(hole)

    gmsh.model.occ.cut([(2, rectangle)], [(2, hole) for hole in holes])


def make_periodic_tiling(epsilon, n_cells_x, n_cells_y, hole_radius_ratio=0.2,
                          hole_radius_ratio_b=None, hole_rotation_deg=0.0,
                          mesh_ratio=30, comm=MPI.COMM_WORLD, geometry_order=2):
    """
    Build a DNS validation mesh: n_cells_x x n_cells_y physical copies of the
    same unit cell used in make_unit_cell (same hole shape/size/rotation
    ratios, so the hole's shape relative to the cell matches exactly), laid
    out as an exact periodic lattice with no edge margins.

    Use this -- not make_perforated_rectangle -- when comparing a direct
    numerical solve against the homogenized K* / two-scale reconstruction,
    since only this layout is geometrically consistent with what the cell
    problem in solve_corrector actually represents.

    hole_radius_ratio, hole_radius_ratio_b, hole_rotation_deg : must match
        whatever was passed to make_unit_cell for this comparison to be valid.

    Domain size comes out to (n_cells_x*epsilon) x (n_cells_y*epsilon).
    Facet tags: 1=x_min, 2=x_max, 3=y_min, 4=y_max, 5=all hole walls
    (not periodic -- this mesh is meant for a plain Dirichlet DNS solve).

    Returns (mesh, cell_tags, facet_tags).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("periodic_tiling")

    _make_periodic_tiling_geometry(
        epsilon, n_cells_x, n_cells_y, hole_radius_ratio,
        hole_radius_ratio_b=hole_radius_ratio_b, hole_rotation_deg=hole_rotation_deg,
    )

    mesh_size = epsilon / mesh_ratio
    mesh = _build_mesh(gdim=2, mesh_size=mesh_size, comm=comm, periodic=False,
                       geometry_order=geometry_order)

    gmsh.finalize()
    return mesh



def _make_perforated_cube_geometry(
    L,
    W,
    H,
    nx,
    ny,
    hole_radius=0
):
    """
    Create a perforated cube gmsh geometry with specified dimensions and hole parameters.
    """

    box = gmsh.model.occ.addBox(
        0,
        0,
        0,
        L,
        W,
        H,
    )

    if not (nx == 0 or ny == 0):
        cylinders = []

        dx = L / (nx + 1)
        dy = W / (ny + 1)
        radius = hole_radius if hole_radius > 0 else min(dx, dy) / 4
    

        for i in range(1, nx + 1):
            for j in range(1, ny + 1):

                x = i * dx
                y = j * dy

                cyl = gmsh.model.occ.addCylinder(
                    x,
                    y,
                    0,
                    0,
                    0,
                    H,
                    radius,
                )

                cylinders.append((3, cyl))

        gmsh.model.occ.cut(
            [(3, box)],
            cylinders,
        )



def make_perforated_cube(
    L,
    W,
    H,
    nx,
    ny,
    hole_radius=0,
    mesh_ratio=30, 
    comm=MPI.COMM_WORLD,
):
    """
    Create a cube perforated with cylinders as a mesh with specified dimensions and hole parameters. Outputted in dolfinx format.
    """

    gmsh.initialize()

    gmsh.model.add("cube")

    _make_perforated_cube_geometry(
        L,
        W,
        H,
        nx,
        ny,
        hole_radius=hole_radius
    )
    mesh_size = (L+W+H)/(3*mesh_ratio)

    mesh = _build_mesh(
        gdim=3,
        mesh_size=mesh_size,
        comm=comm,
        geometry_order=1,
    )

    gmsh.finalize()

    return mesh


