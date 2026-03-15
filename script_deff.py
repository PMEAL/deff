# %%
import taichi as ti
import porespy as ps
from deff import (
    solve_diffusion, 
    compute_effective_diffusivity,
    plot_concentration,
    add_diffusion_streamlines,
    compute_diffusive_conductance,
)
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)
im = ps.generators.cylinders(
    shape=[100, 100, 100], 
    r=10, 
    porosity=0.7,
    seed=0,
)

soln = solve_diffusion(
    im=im,
    direction="x",
    tol=1e-2,
)
res = compute_effective_diffusivity(
    soln,
    direction="x",
)
print(f"D_eff/D_0 = {res['D_eff_norm']:.4f}")
print(f"Tortuosity = {res['tortuosity']:.4f}")

# %%
fig, ax = plt.subplots()
c_slice = plot_concentration(soln._last_vtr, axis=1)
ax.imshow(c_slice, origin="lower", cmap="turbo", vmin=0, vmax=1)
add_diffusion_streamlines(soln._last_vtr, ax, axis=1, color="white", density=1.0)
plt.show()
# %%
soln.export_VTK("sample_diff", direction='x')
results = compute_diffusive_conductance(
    "sample_diff.vtr",
    direction="x",
    D_lu=1/4,
    D0_m2s=2.1e-5,   # e.g. O2 in air at 25°C
    dx_m=2.85e-6,
)
print(f"g_d = {results['g_SI']:.4e} m^3/s")
# %%
