"""
Compute effective diffusivity from D3Q7 BGK LBM diffusion simulation output.

Accepts either a ``DiffusionResult`` object (returned by ``solve_diffusion``) or
a path to a .vtr file written by pyevtk.  When passed a ``DiffusionResult`` the
VTR round-trip is skipped entirely.

Fick's Law:  D_eff = J * L / Δc

  J      = mean diffusive flux in the flow direction over the entire domain
             (solid voxels have flux=0, so this naturally accounts for porosity)
  L      = domain length in the flow direction (lattice units)
  Δc     = c_in - c_out = 1.00 - 0.00 = 1.0  (hardcoded in solve_diffusion)

The flux stored in the VTR is the raw LBM flux:
  flux[i,j,k] = Σ_s  g_s[i,j,k] * e_s[d]
which has units of D_0 * (concentration / length).  Therefore:

  D_eff_lu  = mean(flux) * L / Δc          [lattice units]
  D_eff/D_0 = D_eff_lu / D_lu              [dimensionless; primary output]

Formation factor:  F = D_0 / D_eff = 1 / (D_eff/D_0)
Tortuosity:        τ = F / φ = D_0 / (D_eff × φ)  (always > 1)
"""

import re

import numpy as np

from tools.vtr_io import _parse_xml_arrays, _read_array


__all__ = ["compute_effective_diffusivity"]


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
    arrays       = _parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} × {ny} × {nz} points")

    solid    = _read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    c        = _read_array(raw, binary_start, arrays, "c",     nx, ny, nz)
    flux_vec = _read_array(raw, binary_start, arrays, "flux",  nx, ny, nz)
    if verbose:
        print("  Arrays loaded.")
    return solid, c, flux_vec


def compute_effective_diffusivity(
    source,
    direction=None,
    D_lu=None,
    D0_m2s=None,
    verbose=True,
):
    """Compute effective diffusivity from a D3Q7 BGK LBM simulation.

    Parameters
    ----------
    source : DiffusionResult or str/path-like
        Either a ``DiffusionResult`` returned by ``solve_diffusion()``, or a
        path to a ``.vtr`` file written by ``DiffusionSolver.export_VTK()``.
        When a ``DiffusionResult`` is given, ``direction`` and ``D_lu`` default
        to the values stored in the result.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  If *None* and ``source`` is a ``DiffusionResult``,
        taken from ``source.direction``; otherwise defaults to ``'x'``.
    D_lu : float or None
        Bulk diffusivity in lattice units used during the simulation.
        If *None* and ``source`` is a ``DiffusionResult``, taken from
        ``source.D``; otherwise defaults to 1/4.
    D0_m2s : float or None
        Physical bulk diffusivity in m²/s.  If given, the effective
        diffusivity is also reported in m²/s.  E.g. for O₂ in air at
        25 °C: ``D0_m2s=2.1e-5``.
    verbose : bool
        Print a summary of results to stdout.  Default True.

    Returns
    -------
    dict with keys:
        porosity         – pore volume fraction (dimensionless)
        D_eff_norm       – effective diffusivity ratio D_eff / D_0
        formation_factor – F = D_0 / D_eff  (= 1 / D_eff_norm)
        tortuosity       – τ = F / φ = D_0 / (D_eff × φ)  (always > 1)
        D_eff_m2s        – effective diffusivity in m²/s  (None if D0_m2s is None)
    """
    from ._solve_diffusion import DiffusionResult

    if isinstance(source, DiffusionResult):
        _dir  = direction if direction is not None else source.direction
        _D_lu = D_lu      if D_lu      is not None else source.D
        solid    = source.solid
        c        = source.c
        flux_vec = source.flux
    else:
        _dir  = direction if direction is not None else "x"
        _D_lu = D_lu      if D_lu      is not None else 1.0 / 4.0
        solid, c, flux_vec = _read_diffusion_vtr(source, verbose)

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
    tortuosity       = formation_factor / porosity

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
