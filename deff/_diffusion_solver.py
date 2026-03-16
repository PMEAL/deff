import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK


__all__ = [
    "DiffusionSolver",
]


@ti.data_oriented
class DiffusionSolver:
    """D3Q7 BGK Lattice-Boltzmann solver for passive scalar diffusion.

    Uses a scalar distribution function (7 components) and a single-relaxation-
    time (BGK) collision operator.  This is the diffusion analogue of the
    D3Q19 MRT flow solver in ``SinglePhaseSolver``.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz), dtype int8
        Binary solid map in the **internal** convention: 1 = solid, 0 = pore.
        (``solve_diffusion`` flips the public 1=pore / 0=solid convention
        before passing the image here.)
    sparse_storage : bool
        If True use Taichi pointer-backed sparse fields; only pore voxels are
        allocated.  Default False (dense storage).
    D : float
        Bulk diffusivity in lattice units.  Default 1/4.
        τ_D = 4D + 0.5 (c_s² = 1/4 for D3Q7), so D=1/4 → τ_D=1.5.
        Steps to convergence scale as L²/D so larger D is faster, but
        accuracy degrades above τ_D ≈ 2.  D=1/4 is the sweet spot.
    """

    def __init__(self, im, sparse_storage=False, D=1.0 / 4.0):
        self.enable_projection = True
        self.sparse_storage = sparse_storage
        self.D = D
        object.__setattr__(self, '_last_vtr', None)

        nx, ny, nz = im.shape
        self.nx, self.ny, self.nz = nx, ny, nz

        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self.solid.from_numpy(im)

        # Boundary condition mode: 0 = periodic, 1 = fixed concentration
        self.bc_x_left = 0;   self.c_bcxl = 0.0
        self.bc_x_right = 0;  self.c_bcxr = 0.0
        self.bc_y_left = 0;   self.c_bcyl = 0.0
        self.bc_y_right = 0;  self.c_bcyr = 0.0
        self.bc_z_left = 0;   self.c_bczl = 0.0
        self.bc_z_right = 0;  self.c_bczr = 0.0

        if not sparse_storage:
            self.g = ti.Vector.field(7, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self.G = ti.Vector.field(7, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self.c = ti.field(ti.f32, shape=(nx, ny, nz))
        else:
            self.g = ti.Vector.field(7, ti.f32)
            self.G = ti.Vector.field(7, ti.f32)
            self.c = ti.field(ti.f32)
            n_mem_partition = 3
            cell1 = ti.root.pointer(
                ti.ijk,
                (
                    nx // n_mem_partition + 1,
                    ny // n_mem_partition + 1,
                    nz // n_mem_partition + 1,
                ),
            )
            cell1.dense(
                ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
            ).place(self.c, self.g, self.G)

        self.e = ti.Vector.field(3, ti.i32, shape=(7,))
        self.w = ti.field(ti.f32, shape=(7,))

        # Bounce-back reverse-direction lookup: LR[s] is the index of -e[s]
        self.LR = [0, 2, 1, 4, 3, 6, 5]

        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)

    def init_simulation(self):
        """Compute τ_D from D and initialise the distribution functions."""
        # D3Q7: c_s² = 1/4  →  D = c_s² (τ_D - 0.5)  →  τ_D = 4D + 0.5
        self.tau_D = 4.0 * self.D + 0.5
        self.static_init()
        self.init()

    # ── Initialisation kernels ─────────────────────────────────────────────────

    @ti.kernel
    def static_init(self):
        """Set D3Q7 lattice velocities and weights (compile-time constants)."""
        if ti.static(self.enable_projection):
            # Rest
            self.e[0] = ti.Vector([0,  0,  0])
            # ±x
            self.e[1] = ti.Vector([ 1,  0,  0])
            self.e[2] = ti.Vector([-1,  0,  0])
            # ±y
            self.e[3] = ti.Vector([0,  1,  0])
            self.e[4] = ti.Vector([0, -1,  0])
            # ±z
            self.e[5] = ti.Vector([0,  0,  1])
            self.e[6] = ti.Vector([0,  0, -1])

            self.w[0] = 1.0 / 4.0   # rest
            self.w[1] = 1.0 / 8.0
            self.w[2] = 1.0 / 8.0
            self.w[3] = 1.0 / 8.0
            self.w[4] = 1.0 / 8.0
            self.w[5] = 1.0 / 8.0
            self.w[6] = 1.0 / 8.0

    @ti.kernel
    def init(self):
        """Initialise concentration and distributions to c = 0.5 everywhere."""
        for i, j, k in self.solid:
            if (self.sparse_storage == False) or (self.solid[i, j, k] == 0):
                self.c[i, j, k] = 0.5
                for s in ti.static(range(7)):
                    self.g[i, j, k][s] = self.w[s] * 0.5
                    self.G[i, j, k][s] = self.w[s] * 0.5

    # ── LBM step kernels ───────────────────────────────────────────────────────

    @ti.func
    def periodic_index(self, i):
        """Wrap index i into the domain (periodic in all directions)."""
        iout = i
        if i[0] < 0:
            iout[0] = self.nx - 1
        if i[0] > self.nx - 1:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = self.ny - 1
        if i[1] > self.ny - 1:
            iout[1] = 0
        if i[2] < 0:
            iout[2] = self.nz - 1
        if i[2] > self.nz - 1:
            iout[2] = 0
        return iout

    @ti.kernel
    def collision(self):
        """BGK collision: relax each distribution toward equilibrium."""
        for i, j, k in self.c:
            if self.solid[i, j, k] == 0:
                c_local = self.c[i, j, k]
                for s in ti.static(range(7)):
                    self.g[i, j, k][s] += -(
                        self.g[i, j, k][s] - self.w[s] * c_local
                    ) / self.tau_D

    @ti.kernel
    def streaming1(self):
        """Stream distributions; apply bounce-back on solid neighbours."""
        for i in ti.grouped(self.c):
            if self.solid[i] == 0:
                for s in ti.static(range(7)):
                    ip = self.periodic_index(i + self.e[s])
                    if self.solid[ip] == 0:
                        self.G[ip][s] = self.g[i][s]
                    else:
                        # Bounce-back: reverse direction stays at i
                        self.G[i][self.LR[s]] = self.g[i][s]

    @ti.kernel
    def boundary_condition(self):
        """Apply fixed-concentration Dirichlet BCs on selected faces.

        Each active face is set to the equilibrium distribution for c_bc,
        which forces the macroscopic concentration to c_bc on that face.
        Inactive faces (bc mode = 0) are left untouched (periodic treatment
        is implicit through ``periodic_index`` in ``streaming1``).
        """
        if ti.static(self.bc_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(7)):
                        self.G[0, j, k][s] = self.w[s] * self.c_bcxl

        if ti.static(self.bc_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx - 1, j, k] == 0:
                    for s in ti.static(range(7)):
                        self.G[self.nx - 1, j, k][s] = self.w[s] * self.c_bcxr

        if ti.static(self.bc_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(7)):
                        self.G[i, 0, k][s] = self.w[s] * self.c_bcyl

        if ti.static(self.bc_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny - 1, k] == 0:
                    for s in ti.static(range(7)):
                        self.G[i, self.ny - 1, k][s] = self.w[s] * self.c_bcyr

        if ti.static(self.bc_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(7)):
                        self.G[i, j, 0][s] = self.w[s] * self.c_bczl

        if ti.static(self.bc_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz - 1] == 0:
                    for s in ti.static(range(7)):
                        self.G[i, j, self.nz - 1][s] = self.w[s] * self.c_bczr

    @ti.kernel
    def streaming3(self):
        """Finalise streaming: copy G → g, recompute macroscopic c."""
        for i in ti.grouped(self.c):
            if self.solid[i] == 0:
                self.g[i] = self.G[i]
                self.c[i] = self.g[i].sum()
            else:
                self.c[i] = 0.0

    def step(self):
        """Advance the simulation by one LBM time step."""
        self.collision()
        self.streaming1()
        self.boundary_condition()
        self.streaming3()

    # ── Setter methods ─────────────────────────────────────────────────────────

    def set_bc_c_x0(self, c):
        self.bc_x_left = 1
        self.c_bcxl = float(c)

    def set_bc_c_x1(self, c):
        self.bc_x_right = 1
        self.c_bcxr = float(c)

    def set_bc_c_y0(self, c):
        self.bc_y_left = 1
        self.c_bcyl = float(c)

    def set_bc_c_y1(self, c):
        self.bc_y_right = 1
        self.c_bcyr = float(c)

    def set_bc_c_z0(self, c):
        self.bc_z_left = 1
        self.c_bczl = float(c)

    def set_bc_c_z1(self, c):
        self.bc_z_right = 1
        self.c_bczr = float(c)

    # ── Output ─────────────────────────────────────────────────────────────────

    def export_VTK(self, path, direction):
        """Write a VTK Rectilinear Grid (.vtr) file.

        Parameters
        ----------
        path : str
            Output path without extension (pyevtk appends ``.vtr``).
        direction : {'x', 'y', 'z'}
            Flow direction used for the simulation.  Stored in the file for
            reference; all three flux components are always written.
        """
        g_np     = self.g.to_numpy()        # (nx, ny, nz, 7)
        e_np     = self.e.to_numpy()        # (7, 3)
        solid_np = self.solid.to_numpy()
        c_np     = self.c.to_numpy().astype(np.float32)

        # Diffusive flux vector: J_d = Σ_s g_s * e_s[d]  for each component d
        flux_vec = np.stack(
            [(g_np * e_np[:, d]).sum(axis=-1).astype(np.float32) for d in range(3)],
            axis=-1,
        )  # shape (nx, ny, nz, 3)
        flux_vec[solid_np > 0] = 0.0  # zero out solid voxels
        # Correct for τ/(τ−0.5) overestimation (see DiffusionResult for derivation)
        flux_vec *= (self.tau_D - 0.5) / self.tau_D

        gridToVTK(
            path,
            self.x,
            self.y,
            self.z,
            pointData={
                "Solid": np.ascontiguousarray(solid_np),
                "c":     np.ascontiguousarray(c_np),
                "flux":  (
                    np.ascontiguousarray(flux_vec[..., 0]),
                    np.ascontiguousarray(flux_vec[..., 1]),
                    np.ascontiguousarray(flux_vec[..., 2]),
                ),
            },
        )
