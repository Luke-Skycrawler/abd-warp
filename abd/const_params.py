import warp as wp
import numpy as np

# vec3 = wp.vec3
# mat33 = wp.mat33
# scalar = wp.float32
# vec4 = wp.vec4
# vec2 = wp.vec2
# mat22 = wp.mat22
vec3 = wp.vec3d
mat33 = wp.mat33d
scalar = wp.float64
vec4 = wp.vec4d
vec2 = wp.vec2d
mat22 = wp.mat22d

vec12 = wp.types.vector(length=12, dtype=scalar)
mat12 = wp.types.matrix(shape = (12, 12), dtype = scalar)
vec5i = wp.types.vector(length = 5, dtype = int)
mat23 = wp.types.matrix(shape = (2, 3), dtype = scalar)
mat34 = wp.types.matrix(shape = (3, 4), dtype = scalar)
mat24 = wp.types.matrix(shape = (2, 4), dtype = scalar)

dt = scalar(0.01)
max_iter = 10
tol = 1e1
stiffness = scalar(1e9)
c1 = 1e-4
alpha_min = 1e-3
# gravity = -9.8
# gravity = scalar(-0.0)
gravity = scalar(-9.8)
d2hat = scalar(1e-4)
dhat = scalar(1e-2)
kappa = scalar(1e-3)
ground = scalar(-0.5)

mass = scalar(1e3)
I0 = scalar(1e2)

# switches to enable/disable contacts
pt = True
ee = True
vg = True