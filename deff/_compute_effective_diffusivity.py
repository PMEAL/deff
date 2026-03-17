"""
Compute effective diffusivity and diffusive conductance from D3Q7 BGK LBM
diffusion simulation output.

Accepts a ``DiffusionResult`` object returned by ``solve_diffusion``.

Fick's Law:  D_eff = J * L / Δc

  J      = mean diffusive flux in the flow direction over the interior of the
             domain (inlet/outlet boundary faces excluded; solid voxels are
             zero, so the mean over all voxels naturally accounts for porosity)
  L      = domain length in the flow direction (lattice units)
  Δc     = c_in - c_out = 1.00 - 0.00 = 1.0  (hardcoded in solve_diffusion)

The flux stored in ``DiffusionResult.flux`` is the corrected LBM first moment:
  flux[i,j,k,d] = (τ−0.5)/τ · Σ_s  g_s[i,j,k] * e_s[d]
The (τ−0.5)/τ factor corrects the raw Chapman-Enskog overestimation
(raw first moment = τ·c_s²·|∇c|; true diffusive flux = (τ−0.5)·c_s²·|∇c|).

  D_eff_lu  = mean(flux) * L / Δc          [lattice units]
  D_eff/D_0 = D_eff_lu / D_lu              [dimensionless; primary output]

Formation factor:  F = D_0 / D_eff = 1 / (D_eff/D_0)
Tortuosity:        τ = F / φ = D_0 / (D_eff × φ)  (always > 1)
"""

import re

import numpy as np

from tools.vtr_io import parse_xml_arrays, read_array


__all__ = ["compute_effective_diffusivity", "compute_diffusive_conductance"]


_C_IN  = 1.00
_C_OUT = 0.00


def _read_diffusion_vtr(vtr_file, verbose):
    """Read solid, c, and flux arrays from a diffusion .vtr file."""
    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker       = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1
    xml_header   = raw[:marker].decode("utf-8", errors="replace")
    arrays       = parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} × {ny} × {nz} points")

    solid    = read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    c        = read_array(raw, binary_start, arrays, "c",     nx, ny, nz)
    flux_vec = read_array(raw, binary_start, arrays, "flux",  nx, ny, nz)
    if verbose:
        print("  Arrays loaded.")
    return solid, c, flux_vec


def compute_effective_diffusivity(
    soln,
    direction=None,
    D_lu=None,
    D0_m2s=None,
    verbose=True,
):
    """Compute effective diffusivity from a D3Q7 BGK LBM simulation.

    Parameters
    ----------
    soln : DiffusionResult
        Result object returned by ``solve_diffusion()``.  The flow direction
        and lattice diffusivity are read from ``soln.direction`` and ``soln.D``
        unless overridden by the arguments below.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  Defaults to ``soln.direction``.
    D_lu : float or None
        Bulk diffusivity in lattice units used during the simulation.
        Defaults to ``soln.D`` (typically 1/4).
    D0_m2s : float or None
        Physical bulk diffusivity in m²/s.  If given, ``D_eff_m2s`` is
        also returned.  E.g. for O₂ in air at 25 °C: ``D0_m2s=2.09e-5``.
    verbose : bool
        Print a summary of results to stdout.  Default True.

    Returns
    -------
    dict with keys:
        porosity         – pore volume fraction (dimensionless)
        D_eff_norm       – effective diffusivity ratio D_eff / D_0
        formation_factor – F = D_0 / D_eff  (= 1 / D_eff_norm)
        tortuosity       – τ = F · φ = φ / D_eff_norm  (always ≥ 1)
        D_eff_m2s        – effective diffusivity in m²/s  (None if D0_m2s not given)
    """

    _dir  = direction if direction is not None else soln.direction
    _D_lu = D_lu      if D_lu      is not None else soln.D
    solid    = soln.solid
    c        = soln.c
    flux_vec = soln.flux

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    pore_mask = solid == 0
    porosity  = float(pore_mask.sum()) / pore_mask.size
    nx, ny, nz = solid.shape

    L_dir   = {"x": nx, "y": ny, "z": nz}[_dir]
    dir_idx = {"x": 0,  "y": 1,  "z": 2 }[_dir]
    flux    = flux_vec[..., dir_idx]
    delta_c = _C_IN - _C_OUT  # = 1.0

    J_mean    = float(np.mean(flux))
    D_eff_lu  = J_mean * L_dir / delta_c
    D_eff_norm = D_eff_lu / _D_lu

    formation_factor = 1.0 / D_eff_norm if D_eff_norm > 0 else float("inf")
    tortuosity       = formation_factor * porosity

    D_eff_m2s = None
    if D0_m2s is not None:
        D_eff_m2s = D_eff_norm * D0_m2s

    if verbose:
        print(f"\nFlow direction         = {_dir}")
        print(f"Porosity (φ)           = {porosity:.4f}")
        print(f"Mean flux  J           = {J_mean:.6e}  [lu]")
        print(f"Domain length L        = {L_dir}  [lu]")
        print(f"D_eff  (lattice units) = {D_eff_lu:.6e}")
        print(f"D_bulk (lattice units) = {_D_lu:.6e}")
        print(f"\nD_eff / D_0            = {D_eff_norm:.6f}")
        print(f"Formation factor F     = {formation_factor:.4f}")
        print(f"Tortuosity τ           = {tortuosity:.4f}")
        if D0_m2s is not None:
            print(f"\nWith D_0 = {D0_m2s:.3e} m²/s:")
            print(f"  D_eff = {D_eff_m2s:.4e}  m²/s")
        else:
            print("\nTo get physical D_eff: pass D0_m2s (bulk diffusivity in m²/s).")
        print("\n--- Sanity checks ---")
        c_pore_mean = float(np.mean(c[pore_mask]))
        print(f"Mean c (pore space)    = {c_pore_mean:.6f}  "
              f"(expect ~{(_C_IN + _C_OUT) / 2:.3f} for linear profile)")
        c_flow = {"x": c[:, ny // 2, nz // 2],
                  "y": c[nx // 2, :, nz // 2],
                  "z": c[nx // 2, ny // 2, :]}[_dir]
        print(f"c at domain centreline: min={c_flow.min():.3f}  max={c_flow.max():.3f}")

    return {
        "porosity":         porosity,
        "D_eff_norm":       D_eff_norm,
        "formation_factor": formation_factor,
        "tortuosity":       tortuosity,
        "D_eff_m2s":        D_eff_m2s,
    }


def compute_diffusive_conductance(
    soln,
    direction=None,
    D_lu=None,
    D0_m2s=None,
    voxel_size=None,
    verbose=True,
):
    """Compute diffusive conductance g from a D3Q7 BGK LBM simulation.

    The conductance is defined by  n_a = g · ΔC, where n_a is the diffusive
    flux normalised by the **open (pore) cross-sectional area** of the domain,
    not the total domain area.

    Parameters
    ----------
    soln : DiffusionResult
        Result object returned by ``solve_diffusion()``.  The flow direction
        and lattice diffusivity are read from ``soln.direction`` and ``soln.D``
        unless overridden by the arguments below.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  Defaults to ``soln.direction``.
    D_lu : float or None
        Bulk diffusivity in lattice units.  Defaults to ``soln.D``
        (typically 1/4).
    D0_m2s : float or None
        Physical bulk diffusivity in m²/s (e.g. ``2.09e-5`` for O₂ in air at
        25 °C).  Required to compute ``g_SI``.
    voxel_size : float or None
        Physical side length of one voxel in metres.  Required to compute
        ``g_SI``.
    verbose : bool
        Print a summary to stdout.  Default True.

    Returns
    -------
    dict with keys:
        porosity  – pore volume fraction (dimensionless)
        open_area – mean pore cross-sectional area [voxels²]
        n_a       – diffusive flux normalised by open area [lu]
        g         – diffusive conductance  n_a / ΔC  [lu]
        g_SI      – volumetric conductance [m³/s]
                    (None if D0_m2s or voxel_size is not supplied).
                    Equals D0 · A_open · dx / L for a straight pore.

    Notes
    -----
    The flux is averaged over **interior slices only** (the two
    inlet/outlet boundary faces are excluded).  The equilibrium Dirichlet BC
    forces the flux to zero at those faces; including them would dilute J_mean
    by (L−2)/L and bias g low.

    The open area is computed as porosity × total cross-section area.
    Combining at steady state (conservation of mass guarantees uniform
    cross-sectional flux)::

        J_mean  = mean(flux_dir) over interior slices
        A_open  = porosity × A_total
        n_a     = J_mean / porosity
        g       = n_a / ΔC

    SI conversion (D_physical = D_lu × voxel_size² / Δt)::

        g_SI = g_lu × D0_m2s × A_open × voxel_size / D_lu  [m³/s]
    """
    _dir  = direction if direction is not None else soln.direction
    _D_lu = D_lu      if D_lu      is not None else soln.D

    solid    = soln.solid
    flux_vec = soln.flux

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    pore_mask = solid == 0
    porosity  = float(pore_mask.sum()) / pore_mask.size
    nx, ny, nz = solid.shape

    A_total = {"x": ny * nz, "y": nx * nz, "z": nx * ny}[_dir]
    dir_idx = {"x": 0,  "y": 1,  "z": 2}[_dir]

    flux    = flux_vec[..., dir_idx]
    delta_c = _C_IN - _C_OUT  # = 1.0

    # Average flux over interior slices only (exclude inlet/outlet faces).
    # The equilibrium BC forces flux=0 at the two boundary faces; including them
    # dilutes J_mean by a factor of (L-2)/L and biases the conductance low.
    flux_interior = {"x": flux[1:-1, :, :],
                     "y": flux[:, 1:-1, :],
                     "z": flux[:, :, 1:-1]}[_dir]
    J_mean    = float(np.mean(flux_interior))
    open_area = porosity * A_total       # mean pore cross-section area [voxels²]
    n_a       = J_mean / porosity        # flux normalized by open area
    g         = n_a / delta_c

    g_SI = None
    if D0_m2s is not None and voxel_size is not None:
        g_SI = g * D0_m2s * open_area * voxel_size / _D_lu

    if verbose:
        print(f"\nFlow direction         = {_dir}")
        print(f"Porosity (φ)           = {porosity:.4f}")
        print(f"Total cross-section    = {A_total}  [voxels²]")
        print(f"Open area (φ × A)      = {open_area:.2f}  [voxels²]")
        print(f"Mean flux  J           = {J_mean:.6e}  [lu]")
        print(f"n_a  (flux / A_open)   = {n_a:.6e}  [lu]")
        print(f"g  = n_a / ΔC          = {g:.6e}  [lu]")
        if g_SI is not None:
            print(f"g  (SI)                = {g_SI:.6e}  [m³/s]")
        else:
            print("\nTo get g in SI units: pass D0_m2s (m²/s) and voxel_size (m).")

    return {
        "porosity":   porosity,
        "open_area":  open_area,
        "n_a":        n_a,
        "g":          g,
        "g_SI":       g_SI,
    }
