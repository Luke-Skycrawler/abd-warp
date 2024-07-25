from const_params import *
from affine_body import AffineBody, AffineBodyStates, affine_body_states_empty
from bsr_utils import bsr_cg
from sparse import BSR, bsr_empty
from warp.sparse import bsr_set_from_triplets, bsr_zeros
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
            
            

@wp.func
def grad_ortho(i: int, A: wp.mat33) -> wp.vec3:
    grad = wp.vec3(0.0)
    for j in range(3):
        grad += wp.dot(A[i], A[j]) * A[j]

    grad -= A[i]
    return grad * 4.0 * kappa

@wp.func
def hessian_ortho(i: int, j: int, A: wp.mat33) -> wp.mat33:
    hess = wp.mat33(0.0)
    if i == j:
        qiqiT = wp.outer(A[i], A[i]) 
        qiTqi = wp.dot(A[i], A[i]) - 1.0
        term2 = wp.diag(wp.vec3(qiTqi))

        for k in range(3):
            hess += wp.outer(A[k], A[k])
        hess += qiqiT + term2
    else:
        hess = wp.outer(A[j], A[i])
    return hess * 4.0 * kappa

@wp.func
def offset(i: int, j: int, bsr: BSR) -> int:

    '''diag for now'''
    return i * 16

@wp.kernel
def bsr_hessian_inertia(bsr: BSR, states: AffineBodyStates):
    i = wp.tid()
    os = offset(i, i, bsr)
    for ii in range(4):
        for jj in range(4):
            m = wp.select(ii == jj, 0.0, wp.select(ii == 0, I0, mass))
            I = wp.vec3(m)
            dh =wp.diag(I)
            if ii > 0 and jj > 0:
                dh += hessian_ortho(ii - 1, jj - 1, states.A[i]) * dt * dt
            bsr.blocks[os + ii + jj * 4] = bsr.blocks[os + ii + jj * 4] + dh


@wp.kernel
def flattened_gradient_inertia(g: wp.array(dtype = wp.vec3), states: AffineBodyStates):
    i = wp.tid()
    for ii in range(1, 4):
        g[ii + i * 4] = dt * dt * grad_ortho(ii - 1, states.A[i])

    A_tilde = tildeA(states.A0[i], states.Adot[i])
    p_tilde = tildep(states.p0[i], states.pdot[i])
    q0, q1, q2, q3 = norm_M(states.A[i], states.p[i], A_tilde, p_tilde)

    g[0 + i * 4] = g[0 + i * 4] + q0
    g[1 + i * 4] = g[1 + i * 4] + q1
    g[2 + i * 4] = g[2 + i * 4] + q2
    g[3 + i * 4] = g[3 + i * 4] + q3

@wp.kernel
def _init(states: AffineBodyStates):
    states.Adot[0] = wp.skew(wp.vec3(1.0, 0.0, 0.0))
    states.A[0] = wp.diag(wp.vec3(1.0))
    states.A0[0] = wp.diag(wp.vec3(1.0))

@wp.func
def norm_M(A: wp.mat33, p: wp.vec3, A_tilde: wp.mat33, p_tilde: wp.vec3):
    q0 = p - p_tilde
    q1 = A[0] - A_tilde[0]
    q2 = A[1] - A_tilde[1]
    q3 = A[2] - A_tilde[2]
    return q0 * mass, q1 * I0, q2 * I0, q3 * I0

@wp.func
def tildeA(A0: wp.mat33, Adot: wp.mat33) -> wp.mat33:
    return A0 + dt * Adot

@wp.func
def tildep(p0: wp.vec3, pdot: wp.vec3) -> wp.vec3:
    return p0 + dt * pdot + dt * dt * wp.vec3(0.0, gravity, 0.0)

@wp.kernel
def _set_triplets(rows: wp.array(dtype = int), cols: wp.array(dtype = int)):
    for i in range(4):
        for j in range(4):
            rows[i + j * 4] = i
            cols[i + j * 4] = j

if __name__ == "__main__":
    wp.init()
    bsr = bsr_empty(1) 
    states = affine_body_states_empty(1)
    # A = wp.zeros(1, dtype = wp.mat33)
    g = wp.zeros(4, dtype = wp.vec3)
    dq = wp.zeros_like(g)
    wp.launch(_init, 1, inputs = [states])
    wp.launch(flattened_gradient_inertia, 1, inputs = [g, states])
    wp.launch(bsr_hessian_inertia, 1, inputs = [bsr, states])
    hess = bsr_zeros(4, 4, wp.mat33)
    rows = wp.zeros(16, dtype = int)
    cols = wp.zeros(16, dtype = int)
    values = wp.zeros(16, dtype = wp.mat33)

    values.assign(bsr.blocks.flatten())
    wp.launch(_set_triplets, 1, inputs = [rows, cols])
    bsr_set_from_triplets(hess, rows, cols, values)
    bsr_cg(hess, dq, g, 100, 1e-4)    

    # print(bsr.blocks.numpy())
    # print(hess.values.numpy())
    print(g.numpy())
    print(dq.numpy())
    
    
    

    
    
    
