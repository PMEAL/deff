"""
Compute effective diffusivity from D3Q7 BGK LBM diffusion simulation output.

Reads the .vtr file written by pyevtk using only numpy (no vtk/pyvista needed).

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
import struct

import numpy as np


__all__ = ["compute_effective_diffusivity"]


def _parse_xml_arrays(xml):
    """Return a dict of {name: (offset, dtype, n_components)} from a VTK XML header."""
    arrays = {}
    for m in re.finditer(
        r'<DataArray Name="(\w+)"[^>]*NumberOfComponents="(\d+)"'
        r'[^>]*type="(\w+)"[^>]*offset="(\d+)"',
        xml,
    ):
        name, n_comp, dtype_str, offset = m.groups()
        dtype_map = {
            "Float32": np.float32,
            "Float64": np.float64,
            "Int8":    np.int8,
            "Int32":   np.int32,
            "UInt64":  np.uint64,
        }
        arrays[name] = (int(offset), dtype_map[dtype_str], int(n_comp))
    return arrays


def _read_array(raw, binary_start, arrays, name, nx, ny, nz):
    """Read one named array from the raw VTR bytes.

    pyevtk writes scalar arrays in Fortran-order spatial indexing.
    """
    offset, dtype, n_comp = arrays[name]
    pos = binary_start + offset
    # 8-byte UInt64 header = number of *bytes* in this array
    n_bytes, = struct.unpack_from("<Q", raw, pos)
    pos += 8
    data = np.frombuffer(raw, dtype=dtype, count=n_bytes // dtype().itemsize, offset=pos)
    if n_comp > 1:
        data_pts = data.reshape(nx * ny * nz, n_comp)
        data = np.stack(
            [data_pts[:, c].reshape(nx, ny, nz, order="F") for c in range(n_comp)],
            axis=-1,
        )
    else:
        data = data.reshape((nx, ny, nz), order="F")
    return data


_C_IN  = 1.00
_C_OUT = 0.00


def compute_effective_diffusivity(
    vtr_file,
    direction="x",
    D_lu=1.0 / 4.0,
    D0_m2s=None,
    verbose=True,
):
    """Compute effective diffusivity from a D3Q7 BGK LBM .vtr output file.

    Parameters
    ----------
    vtr_file : str or path-like
        Path to the VTR file written by ``DiffusionSolver.export_VTK()``.
    direction : {'x', 'y', 'z'}
        Flow direction used in the simulation.  Determines which domain
        length is used.  Default ``'x'``.
    D_lu : float
        Bulk diffusivity in lattice units used during the simulation.
        Must match the ``D`` argument passed to ``solve_diffusion``.
        Default 1/4.
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
    direction = direction.lower()
    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    # ── Read VTR file ──────────────────────────────────────────────────────────
    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker       = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1

    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays     = _parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} × {ny} × {nz} points")

    solid = _read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    c     = _read_array(raw, binary_start, arrays, "c",     nx, ny, nz)
    flux  = _read_array(raw, binary_start, arrays, "flux",  nx, ny, nz)
    if verbose:
        print("  Arrays loaded.")

    # ── Porosity ───────────────────────────────────────────────────────────────
    pore_mask = solid == 0
    porosity  = float(pore_mask.sum()) / pore_mask.size

    # ── Effective diffusivity via Fick's Law ───────────────────────────────────
    # J_mean = mean flux over the whole domain (solid voxels already 0 in VTR)
    # D_eff_lu = J_mean * L / Δc
    L_dir  = {"x": nx, "y": ny, "z": nz}[direction]
    delta_c = _C_IN - _C_OUT  # = 1.0

    J_mean    = float(np.mean(flux))
    D_eff_lu  = J_mean * L_dir / delta_c
    D_eff_norm = D_eff_lu / D_lu

    # ── Derived quantities ─────────────────────────────────────────────────────
    # Formation factor: F = D_0 / D_eff  (always > 1)
    # Tortuosity: τ = F / φ = D_0 / (D_eff × φ)  (always > 1 since F > 1 and φ ≤ 1)
    formation_factor = 1.0 / D_eff_norm if D_eff_norm > 0 else float("inf")
    tortuosity       = formation_factor / porosity

    # ── Physical units ─────────────────────────────────────────────────────────
    D_eff_m2s = None
    if D0_m2s is not None:
        D_eff_m2s = D_eff_norm * D0_m2s

    # ── Verbose output ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\nFlow direction         = {direction}")
        print(f"Porosity (φ)           = {porosity:.4f}")
        print(f"Mean flux  J           = {J_mean:.6e}  [lu]")
        print(f"Domain length L        = {L_dir}  [lu]")
        print(f"D_eff  (lattice units) = {D_eff_lu:.6e}")
        print(f"D_bulk (lattice units) = {D_lu:.6e}")
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
                  "z": c[nx // 2, ny // 2, :]}[direction]
        print(f"c at domain centreline: min={c_flow.min():.3f}  max={c_flow.max():.3f}")

    return {
        "porosity":         porosity,
        "D_eff_norm":       D_eff_norm,
        "formation_factor": formation_factor,
        "tortuosity":       tortuosity,
        "D_eff_m2s":        D_eff_m2s,
    }
