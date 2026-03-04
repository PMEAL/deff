# `deff`

`deff` computes the effective diffusivity of a porous material from its 3D tomographic image using the Lattice Boltzmann Method (LBM). Given a binary voxel image of the pore space, it solves steady-state diffusion (Fick's law), returning the normalised effective diffusivity D_eff/D_0, formation factor, and tortuosity.

The LBM implementation is adapted from
[Taichi-LBM3D](https://github.com/yjhp1016/taichi_LBM3D)
([DOI](https://doi.org/10.3390/fluids7080270)) by Jianhui Yang.

---

## Installation

```bash
git clone https://github.com/PMEAL/deff.git
cd deff
pip install -e .
```

Key dependencies: `taichi` (GPU/CPU acceleration), `numpy`, `pyevtk`.
Optional: `porespy` (used in the examples below to generate synthetic images).

---

## Quick start

```python
import taichi as ti
import porespy as ps
from deff import solve_diffusion, compute_effective_diffusivity

ti.init(arch=ti.cpu)  # use ti.gpu for GPU acceleration

# Generate a synthetic test image (1 = pore, 0 = solid)
im = ps.generators.cylinders([200, 200, 200], r=10, porosity=0.7)

# Run the LBM simulation. Saves result to a .vtr file when done.
solver = solve_diffusion(im, direction="x")

# Compute effective diffusivity from the saved file
results = compute_effective_diffusivity(solver._last_vtr, direction="x")
print(f"D_eff/D_0  = {results['D_eff_norm']:.4f}")
print(f"Tortuosity = {results['tortuosity']:.4f}")
```

---

## Common usage patterns

### GPU acceleration

Taichi supports CUDA, Metal, and Vulkan backends. Switch by changing `ti.init`:

```python
ti.init(arch=ti.gpu)   # picks the best available GPU backend
ti.init(arch=ti.cuda)  # CUDA explicitly
```

### Physical units

Pass the bulk diffusivity `D0_m2s` (in m²/s) to `compute_effective_diffusivity` to get results in m²/s:

```python
results = compute_effective_diffusivity(
    solver._last_vtr,
    direction="x",
    D0_m2s=2.1e-5,   # O₂ in air at 25 °C
)
print(f"D_eff = {results['D_eff_m2s']:.4e} m²/s")
```

### Convergence

By default `solve_diffusion` stops early once the concentration field has converged to within a relative tolerance of 1e-2 (i.e. `delta|c| / |c| < 1e-2`). The actual number of steps run is reflected in the auto-generated VTR filename.

```python
# Tighten or loosen the tolerance
solver = solve_diffusion(im, direction="x", tol=1e-3)  # tighter
solver = solve_diffusion(im, direction="x", tol=5e-2)  # faster, coarser

# Disable early stopping and always run n_steps
n = 10000
solver = solve_diffusion(im, direction="x", n_steps=n, tol=None)
results = compute_effective_diffusivity(solver._last_vtr, direction="x")
```

The convergence check fires every `log_every` steps (default 500), so the true stopping point is rounded to that interval.

### Full diffusivity tensor

For anisotropic materials, run all three directions:

```python
results = {}
for ax in ("x", "y", "z"):
    solver = solve_diffusion(im, direction=ax)
    results[ax] = compute_effective_diffusivity(solver._last_vtr, direction=ax)

for ax in ("x", "y", "z"):
    print(f"D_eff/D_0 ({ax}) = {results[ax]['D_eff_norm']:.4f}")
```

### Memory-efficient sparse storage

For images with a high solid fraction, enable sparse storage so only pore voxels are allocated in GPU memory:

```python
solver = solve_diffusion(im, direction="x", sparse=True)
```

---

## Return values

### `solve_diffusion`

Returns the `DiffusionSolver` object after the run. The path to the exported VTR file is stored in `solver._last_vtr` (or `None` if `export_vtk=False`).

### `compute_effective_diffusivity`

Returns a dict:

| Key               | Description                                            |
|-------------------|--------------------------------------------------------|
| `porosity`        | Pore volume fraction φ (dimensionless)                 |
| `D_eff_norm`      | Normalised effective diffusivity D_eff / D_0           |
| `formation_factor`| F = D_0 / D_eff  (= 1 / D_eff_norm)                  |
| `tortuosity`      | τ = F / φ = D_0 / (D_eff × φ)  (always ≥ 1)          |
| `D_eff_m2s`       | Effective diffusivity in m²/s  (`None` if no `D0_m2s`)|

---

## How it works

1. A concentration gradient is imposed by fixing concentration (c_in = 1.0, c_out = 0.0) on opposite faces of the domain along the chosen axis; the other four faces are periodic.
2. The D3Q7 BGK-LBM collision operator evolves the distribution functions to steady state. Solid voxels use bounce-back boundary conditions.
3. Fick's law is applied to the converged flux field:

   **D_eff = J · L / Δc**

   where J is the volume-averaged diffusive flux, L is the domain length, and Δc = 1.0.

4. The normalised result D_eff/D_0 is independent of the lattice diffusivity. If a physical bulk diffusivity D_0 is supplied, D_eff is also reported in m²/s.
