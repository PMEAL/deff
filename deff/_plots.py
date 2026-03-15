from tools.vtr_io import parse_xml_arrays, read_array
import re
import numpy as np
import matplotlib.pyplot as plt
from tools.vtr_io import vtr_to_array


__all__ = [
    'plot_cross_section',
    'add_streamlines',
]


def plot_cross_section(soln, axis=2):
    r"""
    Generate a 2D image of the concentration field for plotting.

    Parameters
    ----------
    filename : str
        The VTR file produced by ``solve_diffusion`` / ``DiffusionSolver.export_VTK()``.
    axis : int
        The axis along which the mid-plane slice is taken.
        ``0`` → y-z plane, ``1`` → x-z plane, ``2`` (default) → x-y plane.

    Returns
    -------
    c_slice : ndarray
        A 2D array of concentration values at the mid-plane slice.
    """
    from ._solve_diffusion import DiffusionResult
    c = soln.c
    nx, ny, nz = c.shape[:3]
    if axis == 0:
        c_slice = c[int(nx / 2), :, :]
    elif axis == 1:
        c_slice = c[:, int(ny / 2), :].T
    elif axis == 2:
        c_slice = c[:, :, int(nz / 2)].T
    return c_slice


def add_streamlines(source, ax, axis, **kwargs):
    r"""
    Add diffusive flux streamlines to a concentration plot.

    Reads the 3-component ``flux`` vector field from a diffusion VTR file and
    calls ``plt.streamplot`` using the two in-plane components at the mid-plane
    slice, matching the slice taken by :func:`plot_concentration`.

    Parameters
    ----------
    source : DiffusionResult or str
        Either a ``DiffusionResult`` returned by ``solve_diffusion()``, or a
        path to a ``.vtr`` file written by ``DiffusionSolver.export_VTK()``.
    ax : matplotlib.axes.Axes
        The axes object to draw streamlines on.
    axis : int
        Slice axis, must match the ``axis`` passed to :func:`plot_concentration`.
        ``0`` → y-z plane, ``1`` → x-z plane, ``2`` → x-y plane.
    **kwargs
        Forwarded to ``plt.streamplot`` (e.g. ``color='white'``, ``density=1.5``).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from ._solve_diffusion import DiffusionResult
    if isinstance(source, DiffusionResult):
        flux = source.flux
        nx, ny, nz = flux.shape[:3]
    else:
        with open(source, "rb") as fh:
            raw = fh.read()
        marker = raw.index(b'<AppendedData encoding="raw">')
        binary_start = raw.index(b"_", marker) + 1
        xml_header = raw[:marker].decode("utf-8", errors="replace")
        arrays = parse_xml_arrays(xml_header)
        m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
        x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
        nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
        flux = read_array(raw, binary_start, arrays, "flux", nx, ny, nz)
    mid = [int(nx / 2), int(ny / 2), int(nz / 2)]
    if axis == 0:
        U = flux[mid[0], :, :, 1]
        V = flux[mid[0], :, :, 2]
    elif axis == 1:
        U = flux[:, mid[1], :, 0].T
        V = flux[:, mid[1], :, 2].T
    elif axis == 2:
        U = flux[:, :, mid[2], 0].T
        V = flux[:, :, mid[2], 1].T
    nrows, ncols = U.shape
    X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    ax.streamplot(X, Y, U, V, **kwargs)
    return ax


# def add_streamlines(source, ax, axis, **kwargs):
#     from ._solve_diffusion import DiffusionResult
#     if isinstance(source, DiffusionResult):
#         velocity = source.flux
#     else:
#         velocity = vtr_to_array(source)
#     mid = [int(s / 2) for s in velocity.shape[:3]]
#     if axis == 0:
#         U = velocity[mid[0], :, :, 1]
#         V = velocity[mid[0], :, :, 2]
#     elif axis == 1:
#         U = velocity[:, mid[1], :, 0].T
#         V = velocity[:, mid[1], :, 2].T
#     elif axis == 2:
#         U = velocity[:, :, mid[2], 0].T
#         V = velocity[:, :, mid[2], 1].T
#     nrows, ncols = U.shape
#     X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
#     ax.streamplot(X, Y, U, V, **kwargs)
#     return ax
