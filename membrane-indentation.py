# Membrane indentation model (axisymmetric)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------
# Parameters
# -------------------------
R = 5.0           # membrane radius (Micro-m)
N = 200           # radial grid points
kappa = 20.0      # bending rigidity (pN·Micro-m)
gamma = 0.5       # surface tension (pN/Micro-m)
F = 10.0          # point force at r=0 (pN)
C0 = 0.0          # spontaneous curvature

# Radial grid
r = np.linspace(0, R, N)
dr = r[1] - r[0]

# functions
# -------------------------
def compute_derivatives(h):
    """Compute first and second derivatives using finite differences."""
    dh = np.zeros_like(h)
    d2h = np.zeros_like(h)

    # First derivative
    dh[1:-1] = (h[2:] - h[:-2]) / (2 * dr)
    dh[0]    = (h[1] - h[0]) / dr
    dh[-1]   = (h[-1] - h[-2]) / dr

    # Second derivative
    d2h[1:-1] = (h[2:] - 2 * h[1:-1] + h[:-2]) / (dr**2)
    d2h[0]    = 2 * (h[1] - h[0]) / (dr**2)
    d2h[-1]   = (h[-1] - 2 * h[-2] + h[-3]) / (dr**2)

    return dh, d2h


def mean_curvature_approx(h):
    """Approximate mean curvature H."""
    dh, d2h = compute_derivatives(h)
    H = np.zeros_like(h)

    with np.errstate(divide='ignore', invalid='ignore'):
        H[1:] = 0.5 * (d2h[1:] + dh[1:] / r[1:])

    H[0] = 0.5 * d2h[0]
    return H


def total_energy(h_internal):
    """Compute total energy = bending + tension - work of point force."""
    # Add boundary condition
    h = np.concatenate([h_internal, [0.0]])

    dh, d2h = compute_derivatives(h)
    H = mean_curvature_approx(h)

    area = 2 * np.pi * r * dr

    bending_density = 0.5 * kappa * (2 * H - C0)**2
    tension_density = 0.5 * gamma * (dh**2)

    Ebend = np.sum(bending_density * area)
    Etens = np.sum(tension_density * area)
    Eforce = -F * h[0]

    return Ebend + Etens + Eforce


# Optimization
# -------------------------
# Initial guess: Gaussian-shaped indentation
h0_full = np.exp(-(r / (0.5 * R))**2) * (-0.1)
h0_internal = h0_full[:-1]

res = minimize(total_energy, h0_internal, method='Powell',
               options={'maxiter': 10000, 'ftol': 1e-9})

h_opt_internal = res.x
h_opt = np.concatenate([h_opt_internal, [0.0]])
H_opt = mean_curvature_approx(h_opt)

# Results
# -------------------------
print("Minimizer success:", res.success, res.message)
print(f"Center deflection = {np.min(h_opt):.4f} µm, "
      f"center curvature ≈ {H_opt[0]:.4f} 1/µm")


# Plot results
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(r, h_opt, label='h(r)')
plt.xlabel('r (µm)')
plt.ylabel('h(r) (µm)')
plt.title('Axisymmetric membrane shape')
plt.grid(True)
plt.tight_layout()
plt.show()

