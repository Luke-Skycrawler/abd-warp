from const_params import *
from affine_body import AffineBody, AffineBodyStates, affine_body_states_empty
# from bsr_utils import bsr_cg
from warp.optim.linear import cg, bicgstab
from sparse import BSR, bsr_empty
from warp.sparse import bsr_set_from_triplets, bsr_zeros
class InertialEnergy:
    def __init__(self):
        pass

    def energy(self, e, states):
        e.zero_()
        wp.launch(energy_inertia, states.A.shape, inputs = [states, e])
    
    def gradient(self, g, states):
        g.zero_()
        wp.launch(flattened_gradient_inertia, states.A.shape, inputs = [g, states])

    def hessian(self, blocks, states):
        wp.launch(bsr_hessian_inertia, states.A.shape, inputs = [blocks, states])

@wp.kernel
def energy_inertia(states: AffineBodyStates, e: wp.array(dtype = float)):
    i = wp.tid()
    A_tilde = tildeA(states.A0[i], states.Adot[i])
    p_tilde = tildep(states.p0[i], states.pdot[i])
    
    dqTMdq = norm_M(states.A[i], states.p[i], A_tilde, p_tilde)
    de = energy_ortho(states.A[i]) * dt * dt + 0.5 * dqTMdq
    wp.atomic_add(e, 0, de)

@wp.func
def norm_M(A: wp.mat33, p: wp.vec3, A_tilde: wp.mat33, p_tilde: wp.vec3) -> float:
    dq0 = p - p_tilde
    dq1 = A[0] - A_tilde[0]
    dq2 = A[1] - A_tilde[1]
    dq3 = A[2] - A_tilde[2]

    return wp.dot(dq0, dq0) * mass + (wp.dot(dq1, dq1) + wp.dot(dq2, dq2) + wp.dot(dq3, dq3)) * I0


@wp.func
def energy_ortho(A: wp.mat33) -> float:
    e = float(0.0)
    for i in range(3):
        for j in range(3):
            term = wp.dot(A[i], A[j]) - wp.select(i == j, 0.0, 1.0)
            e += term * term
    return e * stiffness
            
            

@wp.func
def grad_ortho(i: int, A: wp.mat33) -> wp.vec3:
    grad = -A[i]
    for j in range(3):
        grad += wp.dot(A[i], A[j]) * A[j]

    return grad * 4.0 * stiffness

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
        hess = wp.outer(A[j], A[i]) + wp.diag(wp.vec3(wp.dot(A[j], A[i])))
    return hess * 4.0 * stiffness

@wp.func
def offset(i: int, j: int, bsr: BSR) -> int:

    '''diag for now'''
    return i * 16

@wp.kernel
def bsr_hessian_inertia(blocks: wp.array(dtype = wp.mat33), states: AffineBodyStates):
    i = wp.tid()
    # os = offset(i, i, bsr)
    os = i * 16
    for ii in range(4):
        for jj in range(4):
            m = wp.select(ii == jj, 0.0, wp.select(ii == 0, I0, mass))
            I = wp.vec3(m)
            dh =wp.diag(I)
            if ii > 0 and jj > 0:
                dh += hessian_ortho(ii - 1, jj - 1, states.A[i]) * dt * dt
            blocks[os + ii + jj * 4] += dh


@wp.kernel
def flattened_gradient_inertia(g: wp.array(dtype = wp.vec3), states: AffineBodyStates):
    i = wp.tid()
    for ii in range(1, 4):
        g[ii + i * 4] = dt * dt * grad_ortho(ii - 1, states.A[i])

    A_tilde = tildeA(states.A0[i], states.Adot[i])
    p_tilde = tildep(states.p0[i], states.pdot[i])
    q0, q1, q2, q3 = Mdq(states.A[i], states.p[i], A_tilde, p_tilde)

    g[0 + i * 4] += q0
    g[1 + i * 4] += q1
    g[2 + i * 4] += q2
    g[3 + i * 4] += q3

@wp.kernel
def _init(states: AffineBodyStates):
    states.Adot[0] = wp.skew(wp.vec3(1.0, 0.0, 0.0))
    states.A[0] = wp.diag(wp.vec3(1.0))
    states.A0[0] = wp.diag(wp.vec3(1.0))

@wp.func
def Mdq(A: wp.mat33, p: wp.vec3, A_tilde: wp.mat33, p_tilde: wp.vec3):
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
def _set_triplets(n_bodies: int, n_ij: int, ij_list: wp.array(dtype = wp.vec2i), rows: wp.array(dtype = int), cols: wp.array(dtype = int)):
    i = wp.tid()
    os = i * 16

    I = int(0)
    J = int(0)
    if i < n_bodies:
        # diagonal blocks
        I = i
        J = i
    else:
        # off-diagonal blocks, upper triangle index range [n_ij, n_ij + n_bodies), lower triangle at [n_ij + n_bodies, n_bodies + 2 n_ij)
        idx = wp.select(i < n_bodies + n_ij, i - n_bodies - n_ij, i - n_bodies)
        I = ij_list[idx][0]
        J = ij_list[idx][1]

    for ii in range(4):
        for jj in range(4):
            rows[os + ii + jj  *4] = ii + 4 * I
            cols[os + ii + jj * 4] = jj + 4 * J




@wp.kernel
def _update_q(states: AffineBodyStates, dq: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    states.p[i] = states.p[i] - dq[i * 4 + 0]
    q1 = states.A[i][0] - dq[i * 4 + 1]
    q2 = states.A[i][1] - dq[i * 4 + 2]
    q3 = states.A[i][2] - dq[i * 4 + 3]
    states.A[i] = wp.transpose(wp.mat33(q1, q2, q3))
    # states.A[i] = wp.mat33(q1, q2, q3)

@wp.kernel
def _update_q0qdot(states: AffineBodyStates):
    i = wp.tid()
    states.pdot[i] = (states.p[i] - states.p0[i]) / dt
    states.Adot[i] = (states.A[i] - states.A0[i]) / dt

    states.p0[i] = states.p[i]
    states.A0[i] = states.A[i]


if __name__ == "__main__":
    wp.init()
    bsr = bsr_empty(1) 
    states = affine_body_states_empty(1)
    # A = wp.zeros(1, dtype = wp.mat33)
    g = wp.zeros(4, dtype = wp.vec3)
    dq = wp.zeros_like(g)
    wp.launch(_init, 1, inputs = [states])
    inertia = InertialEnergy()


    hess = bsr_zeros(4, 4, wp.mat33)
    rows = wp.zeros(16, dtype = int)
    cols = wp.zeros(16, dtype = int)
    values = wp.zeros(16, dtype = wp.mat33)

    for frame in range(10):
        
        wp.copy(states.A, states.A0)
        wp.copy(states.p, states.p0)

        it = 0 
        while True:
            inertia.gradient(g, states)
            inertia.hessian(bsr, states)


            values.assign(bsr.blocks.flatten())
            # print(bsr.blocks.numpy())

            wp.launch(_set_triplets, 1, inputs = [rows, cols])
            bsr_set_from_triplets(hess, rows, cols, values)
            bicgstab(hess, g, dq, 1e-4)

            # print(bsr.blocks.numpy())
            # print(hess.values.numpy())
            # print(g.numpy())
            # print(dq.numpy())

            wp.launch(_update_q, 1, inputs = [states, dq])
            
            it += 1
            if it > 1:
                inertia.gradient(g, states)
                # print("residue gradient: ", g.numpy())
                break
        wp.launch(_update_q0qdot, 1, inputs = [states])
        print("a dot = ", states.Adot.numpy())

    
    
    

    
    
    
