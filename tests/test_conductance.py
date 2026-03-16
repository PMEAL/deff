"""
Validate compute_diffusive_conductance against the analytical solution for a
straight cylinder:

    G = D0 * pi * R^2 / L          [m^3/s]

Tests vary L and R independently so that systematic biases in the ratio
G_LBM / G_analytical can be spotted by pattern (constant offset vs. L- or
R-dependent drift).
"""

import numpy as np
import pytest
from deff import solve_diffusion, compute_diffusive_conductance

D0_M2S  = 2.09e-5   # O2 in air at 25 C  [m^2/s]
DX_M    = 1e-6      # voxel size          [m]
W = H   = 40        # cross-section box   [voxels]
TOL_SIM = 1e-4      # LBM convergence tolerance


def _make_cylinder(L, R, box_w=W, box_h=H):
    """Binary image (1=pore) of a straight cylinder along x."""
    box = np.zeros([L, box_w, box_h], dtype=int)
    cy, cz = box_w // 2, box_h // 2
    for j in range(box_w):
        for k in range(box_h):
            if (j - cy) ** 2 + (k - cz) ** 2 <= R ** 2:
                box[:, j, k] = 1
    return box


def _analytical_G(L, R, dx=DX_M):
    """Analytical conductance G = D0 * pi * R^2 / L  [m^3/s]."""
    return D0_M2S * np.pi * (R * dx) ** 2 / (L * dx)


def _lbm_G(L, R, box_w=W, box_h=H, dx=DX_M):
    """Run LBM and return g_SI [m^3/s]."""
    im = _make_cylinder(L, R, box_w=box_w, box_h=box_h)
    soln = solve_diffusion(im, direction="x", tol=TOL_SIM, verbose=False)
    result = compute_diffusive_conductance(
        soln, direction="x", D0_m2s=D0_M2S, voxel_size=dx, verbose=False
    )
    return result["g_SI"]


# ---------------------------------------------------------------------------
# Vary tube LENGTH (fixed R=8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [30, 50, 80])
def test_conductance_vs_length(L):
    """G_LBM / G_analytical should be within 5 % for a straight cylinder."""
    R = 8
    G_lbm  = _lbm_G(L, R)
    G_anal = _analytical_G(L, R)
    ratio  = G_lbm / G_anal
    print(f"\nL={L}, R={R}: G_LBM={G_lbm:.4e}  G_anal={G_anal:.4e}  ratio={ratio:.4f}")
    assert abs(ratio - 1.0) < 0.05, f"ratio={ratio:.4f}, expected 1.0 ± 0.05"


# ---------------------------------------------------------------------------
# Vary tube RADIUS (fixed L=50)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R", [5, 8, 12])
def test_conductance_vs_radius(R):
    """G_LBM / G_analytical should be within 5 % for a straight cylinder."""
    L = 50
    G_lbm  = _lbm_G(L, R)
    G_anal = _analytical_G(L, R)
    ratio  = G_lbm / G_anal
    print(f"\nL={L}, R={R}: G_LBM={G_lbm:.4e}  G_anal={G_anal:.4e}  ratio={ratio:.4f}")
    assert abs(ratio - 1.0) < 0.05, f"ratio={ratio:.4f}, expected 1.0 ± 0.05"


# ---------------------------------------------------------------------------
# Sanity: concentration profile should be radially flat at mid-axial slice
# ---------------------------------------------------------------------------

def test_radial_concentration_flat():
    """
    At the mid-axial cross-section the concentration should be uniform
    across the pore space (no radial gradient → walls are no-flux, not sinks).
    Expected value: ~0.5  (midpoint between c_in=1 and c_out=0).
    """
    L, R = 50, 8
    im   = _make_cylinder(L, R)
    soln = solve_diffusion(im, direction="x", tol=TOL_SIM, verbose=False)

    nx, ny, nz = soln.c.shape
    cy, cz = ny // 2, nz // 2
    mid_x  = nx // 2

    solid_slice = soln.solid[mid_x, :, :]
    c_slice     = soln.c[mid_x, :, :]
    js, ks      = np.where(solid_slice == 0)
    c_pore      = c_slice[js, ks]

    c_mean = float(c_pore.mean())
    c_std  = float(c_pore.std())

    print(f"\nMid-slice c: mean={c_mean:.4f}  std={c_std:.6f}")
    # BCs at nodes i=0 (c=1) and i=nx-1 (c=0); mid_x=nx//2=25 sits at
    # c = 1 - 25/49 = 24/49 ≈ 0.490, not exactly 0.5.
    c_expected = 1.0 - mid_x / (nx - 1)
    assert abs(c_mean - c_expected) < 0.01, f"mean c={c_mean:.4f}, expected ~{c_expected:.4f}"
    assert c_std < 0.005, f"std c={c_std:.6f}, expected < 0.005 (radially flat)"


# ---------------------------------------------------------------------------
# Resolution convergence: same physical geometry at 1× and 2× voxel density
# ---------------------------------------------------------------------------

def test_conductance_resolution_convergence():
    """
    Keep the physical cylinder fixed (R_phys=8µm, L_phys=50µm) and double
    the voxel resolution (halve dx), so the LBM uses more voxels to
    represent the same geometry.  The ratio G_LBM / G_analytical should move
    closer to 1.0 at higher resolution if residual errors are discretisation
    artefacts.

    Geometry at each resolution:
      1×: L=50, R=8,  W=H=40, dx=1e-6 m
      2×: L=100, R=16, W=H=80, dx=5e-7 m
    """
    # Base resolution
    L1, R1, W1, dx1 = 50, 8, 40, 1e-6
    G_lbm_1  = _lbm_G(L1, R1, box_w=W1, box_h=W1, dx=dx1)
    G_anal_1 = _analytical_G(L1, R1, dx=dx1)
    ratio_1  = G_lbm_1 / G_anal_1

    # 2× resolution (same physical size)
    L2, R2, W2, dx2 = 100, 16, 80, 5e-7
    G_lbm_2  = _lbm_G(L2, R2, box_w=W2, box_h=W2, dx=dx2)
    G_anal_2 = _analytical_G(L2, R2, dx=dx2)
    ratio_2  = G_lbm_2 / G_anal_2

    print(
        f"\n1× (L={L1}, R={R1}, dx={dx1:.0e}): ratio={ratio_1:.4f}"
        f"\n2× (L={L2}, R={R2}, dx={dx2:.0e}): ratio={ratio_2:.4f}"
        f"\n|Δratio| = {abs(ratio_2 - ratio_1):.4f}"
    )

    # Both resolutions should pass the 5 % tolerance
    assert abs(ratio_1 - 1.0) < 0.05, f"1× ratio={ratio_1:.4f}"
    assert abs(ratio_2 - 1.0) < 0.05, f"2× ratio={ratio_2:.4f}"
    # Higher resolution should not be worse (allow 1 % slack for noise)
    assert abs(ratio_2 - 1.0) <= abs(ratio_1 - 1.0) + 0.01, (
        f"2× resolution (ratio={ratio_2:.4f}) is not more accurate than "
        f"1× (ratio={ratio_1:.4f})"
    )
