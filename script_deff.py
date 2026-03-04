import taichi as ti
import porespy as ps
from deff import solve_diffusion, compute_effective_diffusivity

ti.init(arch=ti.cpu)
im = ps.generators.cylinders(
    shape=[100, 100, 100], 
    r=10, 
    porosity=0.7,
    seed=0,
)

solver = solve_diffusion(
    im=im,
    direction="x",
    tol=1e-2,
)
res = compute_effective_diffusivity(
    solver._last_vtr,
    direction="x",
)
print(f"D_eff/D_0 = {res['D_eff_norm']:.4f}")
print(f"Tortuosity = {res['tortuosity']:.4f}")
