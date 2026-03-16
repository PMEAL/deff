# %%
import taichi as ti
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from deff import (
    solve_diffusion, 
    compute_diffusive_conductance,
    plot_cross_section,
    add_streamlines,
)
from porespy.tools import (
    get_edt,
)

edt = get_edt()

ti.init(arch=ti.cpu)
Rp = 20
R_lu = 10
L_lu = 50
W = 50
H = 50
box = np.zeros([L_lu, W, H], dtype=int)
cy, cz = int(W/2), int(H/2)
for i in range(L_lu):
    for j in range(W):
        for k in range(H):
            if (j - cy)**2 + (k - cz)**2 < R_lu**2:
                box[i, j, k] = 1
# Add spheres to the ends
balls = np.ones_like(box, dtype=bool)
balls[0, cy, cz] = False
balls[-1, cy, cz] = False
balls = edt(balls) < Rp
# box[balls] = True

soln = solve_diffusion(
    im=box,
    direction="x",
    tol=1e-4,
)

# %%
fig, ax = plt.subplots()
c_slice = plot_cross_section(soln, axis=1)
ax.imshow(c_slice, origin="lower", cmap="turbo", vmin=0, vmax=1)
add_streamlines(soln, ax, axis=1, color="white", density=1.0)
plt.show()

# %%
results = compute_diffusive_conductance(
    soln,
    direction="x",
    dx_m=1e-6,
)
print(f"g_LBM = {results['g_SI']:.4e} m^3/s")

# Analytical check: straight cylinder  g_d = D0 * pi * R^2 / L
D0_m2s = 2.1e-5
dx_m = 1e-6
g_d_analytical_lu = (1/4) * np.pi * R_lu**2 / L_lu
g_d_analytical_SI = g_d_analytical_lu * D0_m2s * dx_m / (1/4)
print(f"\nAnalytical (cylinder): g_d = {g_d_analytical_SI:.4e} m^3/s")
print(f"Ratio LBM / analytical = {results['g_SI'] / g_d_analytical_SI:.4f}")
# %%
