"""
Internal helpers for parsing VTR (VTK Rectilinear Grid) binary files.

These utilities underpin :func:`~deff.utils.vtr_to_array`, which reads
exported ``.vtr`` files directly.  The primary public API
(:func:`~deff.solve_diffusion`, :func:`~deff.compute_effective_diffusivity`,
etc.) works exclusively with :class:`~deff._solve_diffusion.DiffusionResult`
objects and does not require file I/O.
"""

import re
import struct

import numpy as np


__all__ = [
    "vtr_to_array",
    "parse_xml_arrays",
    "read_array",
    "read_diffusion_vtr",
]


def vtr_to_array(filename):
    r"""
    Read the velocity field from a VTR file and return it as a numpy array.

    Parameters
    ----------
    filename : str
        Path to the ``.vtr`` file produced by the simulation.

    Returns
    -------
    velocity : ndarray, shape (nx, ny, nz, 3)
        Velocity field with the final axis indexing the x, y, and z components.
        For example, ``velocity[..., 0]`` is a 3D image of the x-component.
    """
    with open(filename, "rb") as fh:
        raw = fh.read()
    marker = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1
    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays = parse_xml_arrays(xml_header)
    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    velocity = read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)
    return velocity


def parse_xml_arrays(xml):
    """
    Parse ``DataArray`` entries from a VTK XML header string.

    Parameters
    ----------
    xml : str
        The XML header section of a ``.vtr`` file (everything before the
        ``<AppendedData>`` block).

    Returns
    -------
    arrays : dict
        Mapping of ``{name: (offset, dtype, n_components)}`` for each
        ``DataArray`` element found in the header.
    """
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
            "Int8": np.int8,
            "Int32": np.int32,
            "UInt64": np.uint64,
        }
        arrays[name] = (int(offset), dtype_map[dtype_str], int(n_comp))
    return arrays


def read_array(raw, binary_start, arrays, name, nx, ny, nz):
    """
    Decode one named array from the raw VTR binary blob.

    pyevtk writes vector data in AoS layout (components interleaved per point)
    with Fortran-order spatial indexing::

        vx[0,0,0], vy[0,0,0], vz[0,0,0], vx[1,0,0], ...

    Scalar arrays are written in Fortran order directly.

    Parameters
    ----------
    raw : bytes
        The full raw contents of a ``.vtr`` file.
    binary_start : int
        Byte offset of the ``_`` marker that begins the ``AppendedData`` block.
    arrays : dict
        Metadata dict as returned by :func:`parse_xml_arrays`.
    name : str
        Name of the ``DataArray`` to read.
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    data : ndarray
        For scalars: shape ``(nx, ny, nz)``.
        For vectors: shape ``(nx, ny, nz, n_components)``.
    """
    offset, dtype, n_comp = arrays[name]
    pos = binary_start + offset
    # 8-byte UInt64 header = number of *bytes* in this array
    (n_bytes,) = struct.unpack_from("<Q", raw, pos)
    pos += 8
    data = np.frombuffer(
        raw, dtype=dtype, count=n_bytes // dtype().itemsize, offset=pos
    )
    if n_comp > 1:
        data_pts = data.reshape(nx * ny * nz, n_comp)
        data = np.stack(
            [data_pts[:, c].reshape(nx, ny, nz, order="F") for c in range(n_comp)],
            axis=-1,
        )
    else:
        data = data.reshape((nx, ny, nz), order="F")
    return data


def read_diffusion_vtr(vtr_file, verbose):
    """
    Read the solid mask, concentration, and flux arrays from a diffusion ``.vtr`` file.

    Parameters
    ----------
    vtr_file : str
        Path to the ``.vtr`` file written by :meth:`DiffusionResult.export_to_vtk`.
    verbose : bool
        If ``True``, print progress messages to stdout.

    Returns
    -------
    solid : ndarray, shape (nx, ny, nz), dtype int8
        Solid mask (1 = solid, 0 = pore).
    c : ndarray, shape (nx, ny, nz)
        Concentration field.
    flux_vec : ndarray, shape (nx, ny, nz, 3)
        Diffusive flux vector (Jx, Jy, Jz) at each voxel.
    """
    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1
    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays = parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} × {ny} × {nz} points")

    solid = read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    c = read_array(raw, binary_start, arrays, "c", nx, ny, nz)
    flux_vec = read_array(raw, binary_start, arrays, "flux", nx, ny, nz)
    if verbose:
        print("  Arrays loaded.")
    return solid, c, flux_vec
