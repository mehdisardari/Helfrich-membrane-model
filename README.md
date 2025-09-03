# Helfrich-membrane-model

Python implementation of an axisymmetric membrane indentation model based on Helfrich energy.  
The model estimates the equilibrium shape of a circular membrane under a localized point force by minimizing the total energy (bending + tension – external work).

---
## Features
- Axisymmetric geometry with radial discretization
- Finite difference approximation for derivatives
- Mean curvature calculation
- Energy minimization using `scipy.optimize.minimize`
- Visualization of the deflected membrane shape
