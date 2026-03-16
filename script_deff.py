# %%
import taichi as ti
import porespy as ps
from deff import (
    solve_diffusion, 
    compute_effective_diffusivity,
    plot_cross_section,
    add_streamlines,
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
c_slice = plot_cross_section(soln, axis=1)
ax.imshow(c_slice, origin="lower", cmap="turbo", vmin=0, vmax=1)
add_streamlines(soln, ax, axis=1, color="white", density=1.0)
plt.show()
