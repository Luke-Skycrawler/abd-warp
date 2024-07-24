import warp as wp
import numpy as np

vec12 = wp.types.vector(length=12, dtype=wp.float32)
mat12 = wp.types.matrix(shape = (12, 12), dtype = wp.float32)

dt = 0.01
dhat = 1e-3
max_iter = 100
tol = 1e-4
kappa = 1e7
stiffness = 1e7
c1 = 1e-4
gravity = -9.8

