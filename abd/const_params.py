import warp as wp
import numpy as np

vec12 = wp.types.vector(length=12, dtype=wp.float32)
mat12 = wp.types.matrix(shape = (12, 12), dtype = wp.float32)
vec5i = wp.types.vector(length = 5, dtype = int)

dt = 0.01
d2hat = 1e-3
_dhat = np.sqrt(d2hat)
dhat = wp.constant(_dhat)
max_iter = 100
tol = 1e-4
kappa = 1e9
stiffness = 1e9
c1 = 1e-4
# gravity = -9.8
# gravity = wp.constant(-0.0)
gravity = wp.constant(-9.8)


mass = 1e4
I0 = 4e4
