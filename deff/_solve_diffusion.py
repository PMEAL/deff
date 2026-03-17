"""High-level entry point: run a D3Q7 BGK LBM diffusion simulation."""

import time

import numpy as np

from ._diffusion_solver import DiffusionSolver


__all__ = ["solve_diffusion", "DiffusionResult"]

# Concentration BCs are hardcoded: Δc = 1 keeps the Fick's law formula simple.
# Only the ratio D_eff/D_0 is reported, so the absolute values cancel.
_C_IN  = 1.00
_C_OUT = 0.00

_BC_SETTERS = {
    "x": ("set_bc_c_x0", "set_bc_c_x1"),
    "y": ("set_bc_c_y0", "set_bc_c_y1"),
    "z": ("set_bc_c_z0", "set_bc_c_z1"),
}


class DiffusionResult:
    """Container for a converged D3Q7 BGK LBM diffusion simulation.

    Attributes
    ----------
    solid : np.ndarray, shape (nx, ny, nz), dtype int8
        Solid mask in internal convention: 1 = solid, 0 = pore.
    c : np.ndarray, shape (nx, ny, nz), dtype float32
        Concentration field.
    flux : np.ndarray, shape (nx, ny, nz, 3), dtype float32
        Corrected diffusive flux vector (Jx, Jy, Jz) at each voxel.
        The raw LBM first moment Σ g_s e_s overestimates the true flux by
        τ/(τ−0.5); this array has already been multiplied by (τ−0.5)/τ.
        Solid voxels are set to zero.
    direction : str
        Flow direction used in the simulation ('x', 'y', or 'z').
    D : float
        Bulk diffusivity in lattice units (default 1/4).
    """

    def __init__(self, solver, direction, D):
        self.direction = direction
        self.D = D
        self._solver = solver
        solid_np     = solver.solid.to_numpy()
        self.solid   = solid_np
        self.c       = solver.c.to_numpy().astype(np.float32)
        # Compute diffusive flux vector: J_d = Σ_s g_s * e_s[d]
        g_np = solver.g.to_numpy()   # (nx, ny, nz, 7)
        e_np = solver.e.to_numpy()   # (7, 3)
        flux_vec = np.stack(
            [(g_np * e_np[:, d]).sum(axis=-1).astype(np.float32) for d in range(3)],
            axis=-1,
        )  # shape (nx, ny, nz, 3)
        flux_vec[solid_np > 0] = 0.0
        # Correct for the τ/(τ−0.5) overestimation: the first moment Σ g_s e_s
        # equals τ·c_s²·|∇c| but the true diffusive flux is D_lu·|∇c| = (τ−0.5)·c_s²·|∇c|.
        flux_vec *= (solver.tau_D - 0.5) / solver.tau_D
        self.flux = flux_vec

    def export_to_vtk(self, prefix):
        """Write a VTK Rectilinear Grid (.vtr) file.

        Parameters
        ----------
        prefix : str
            Output path without extension (pyevtk appends ``.vtr``).
        """
        self._solver.export_VTK(prefix, self.direction)


def solve_diffusion(
    im,
    direction="x",
    n_steps=50000,
    D=1.0 / 4.0,
    log_every=500,
    export_vtk=False,
    output_prefix="LB_Diffusion",
    verbose=True,
    sparse=False,
    tol=1e-2,
):
    """
    Run a concentration-driven D3Q7 BGK diffusion simulation to steady state.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz)
        Binary image of the pore space.  1 (or True) = pore, 0 (or False) = solid.
        This matches the PoreSpy convention.
    direction : {'x', 'y', 'z'}
        Axis along which the concentration gradient is applied.  Default ``'x'``.
    n_steps : int
        Maximum number of LBM time steps to run.  Default 50000.
        Diffusion convergence requires ~L²/D steps (e.g. ~40 000 for a
        100-voxel domain with D=1/4); scale up for larger images.
    D : float
        Bulk diffusivity in lattice units.  Default 1/4.
        The BGK relaxation time is τ_D = 4D + 0.5 (c_s² = 1/4 for D3Q7).
        Steps to convergence scale as L²/D so larger D is faster, but
        accuracy degrades above τ_D ≈ 2 (D ≈ 3/8).  D=1/4 (τ_D=1.5) is the
        sweet spot: ~4× faster than D=1/6 with no accuracy penalty.
    log_every : int
        Print a progress line every this many steps.  Default 500.
    export_vtk : bool
        If True, write ``{output_prefix}-{final_step}-{direction}.vtr`` at the
        end.  The file contains Solid, c, and flux arrays.  Default False.
    output_prefix : str
        Filename prefix for the VTR output.  Default ``'LB_Diffusion'``.
    verbose : bool
        Print progress to stdout.  Default True.
    sparse : bool
        If True, use Taichi sparse (pointer-backed) storage.  Only pore cells
        are allocated, reducing memory on high-solid-fraction images.
        Default False.
    tol : float or None
        Convergence tolerance.  The simulation stops early when the relative
        change in the total concentration field between log intervals falls
        below this value: ``delta|c| / |c| < tol``.  Set to ``None`` to
        always run the full ``n_steps``.  Default 1e-2.
        A looser tolerance than the flow solver (1e-3) is appropriate here:
        the effective diffusivity is dominated by the mean flux, which
        converges faster than the pointwise concentration field.

    Returns
    -------
    result : DiffusionResult
        Result object containing ``solid``, ``c``, ``flux``, ``direction``,
        and ``D`` as numpy arrays/values.  Pass directly to
        ``compute_effective_diffusivity()`` or ``compute_diffusive_conductance()``,
        or call ``result.export_to_vtk(prefix)`` to save a VTR file.

    Notes
    -----
    Taichi must be initialised by the caller before invoking this function::

        import taichi as ti
        ti.init(arch=ti.cpu)
    """
    direction = direction.lower()
    if direction not in _BC_SETTERS:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    # Public convention: 1=pore, 0=solid (PoreSpy-compatible).
    # DiffusionSolver uses the opposite, so flip here.
    solid_im = (im == 0).astype(np.int8)
    solver = DiffusionSolver(solid_im, sparse_storage=sparse, D=D)

    set_inlet, set_outlet = _BC_SETTERS[direction]
    getattr(solver, set_inlet)(_C_IN)
    getattr(solver, set_outlet)(_C_OUT)
    solver.init_simulation()

    time_init = time.time()
    time_pre = time_init
    c_prev = None
    final_step = n_steps

    for i in range(n_steps + 1):
        solver.step()

        if i % log_every == 0:
            time_now = time.time()
            diff = int(time_now - time_pre)
            elap = int(time_now - time_init)
            m_d, s_d = divmod(diff, 60)
            h_d, m_d = divmod(m_d, 60)
            m_e, s_e = divmod(elap, 60)
            h_e, m_e = divmod(m_e, 60)

            if verbose:
                print(
                    f"Step {i:6d}/{n_steps}  "
                    f"interval {h_d:02d}h{m_d:02d}m{s_d:02d}s  "
                    f"elapsed {h_e:02d}h{m_e:02d}m{s_e:02d}s"
                )

            c_now = solver.c.to_numpy()
            if c_prev is not None:
                c_total  = np.sum(np.abs(c_now))
                c_change = np.sum(np.abs(c_now - c_prev))
                if verbose:
                    print(f"         |c|={c_total:.4e}  delta|c|={c_change:.4e}")
                if tol is not None and c_total > 0 and c_change / c_total < tol:
                    if verbose:
                        print(
                            f"Converged at step {i} "
                            f"(delta|c|/|c| = {c_change/c_total:.2e} < tol={tol:.2e})"
                        )
                    final_step = i
                    break
            c_prev = c_now
            time_pre = time_now

    result = DiffusionResult(solver, direction, D)

    if export_vtk:
        vtk_path = f"{output_prefix}-{final_step}-{direction}"
        result.export_to_vtk(vtk_path)
        if verbose:
            print(f"Exported {vtk_path}.vtr")

    return result
