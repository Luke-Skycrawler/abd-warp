import warp as wp
import numpy as np

vec12 = wp.types.vector(length=12, dtype=wp.float32)
mat12 = wp.types.matrix(shape = (12, 12), dtype = wp.float32)
vec5i = wp.types.vector(length = 5, dtype = int)
mat23 = wp.types.matrix(shape = (2, 3), dtype = float)


dt = 0.01
max_iter = 100
tol = 1e-4
stiffness = 1e9
c1 = 1e-4
# gravity = -9.8
# gravity = wp.constant(-0.0)
gravity = wp.constant(-9.8)
d2hat = wp.constant(1e-4)
dhat = wp.constant(1e-2)
kappa = wp.constant(1e-3)


mass = 1e4
I0 = 4e4
