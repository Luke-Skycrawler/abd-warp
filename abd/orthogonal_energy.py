from abd.const_params import *
from abd.affine_body import AffineBody

class InertialEnergy:
    def __init__(self):
        pass
    def energy(self, inputs):
        A, p, scene_objects = inputs
        scene_objects
    
    def gradient(self, inputs):
        A, p = inputs

    def hessian(self, inputs):
        A, p = inputs


@wp.func
def energy_ortho(A: wp.mat33) -> float:
    e = float(0.0)
    for i in range(3):
        for j in range(3):
            term = wp.dot(A[i], A[j]) - wp.select(i == j, 0.0, 1.0)
            e += term * term
    return e * kappa
            
            

wp.func
def grad_ortho(i: int, A: wp.mat33) -> wp.vec3:
    grad = wp.vec3(0.0)
    for j in range(3):
        grad += wp.dot(A[i], A[j]) * A[j]

    grad -= A[i]
    return grad * 4 * kappa

@wp.func
def hessian_ortho(i: int, j: int, A: wp.mat33) -> wp.mat33:
    hess = wp.mat33(0.0)
    if i == j:
        qiqiT = wp.outer(A[i], A[i]) 
        qiTqi = wp.dot(A[i], A[i]) - 1
        term2 = wp.diag(wp.vec3(qiTqi))

        for k in range(3):
            hess += wp.outer(A[k], A[k])
        hess += qiqiT + term2
    else:
        hess = wp.outer(A[j], A[i])
    return hess * 4 * kappa