from const_params import *
from affine_body import AffineBody, AffineBodyStates
from sparse import BSR, bsr_empty

from warp.sparse import bsr_set_from_triplets, bsr_zeros
from warp.optim.linear import cg, bicgstab
from psd.ee import beta_gamma_ee, C_ee
from psd.hl import signed_distance
from psd.vf import beta_gamma_pt, C_vf

# temp
from orthogonal_energy import offset


class IPCContactEnergy:
    def __init__(self) -> None:
        pass

    def energy(self, inputs):
        ee_list = inputs[0]
        pt_list = inputs[1]
        vg_list = inputs[2]
        E = wp.array(dtype = wp.zeros((1, ), dtype = float))
        inputs.append(E)
        wp.launch(ipc_energy_ee, dim = ee_list.shape, inputs = inputs)
        wp.launch(ipc_energy_pt, dim = pt_list.shape, inputs = inputs)
        wp.launch(ipc_energy_vg, dim = vg_list.shape, inputs = inputs)
        return E.numpy()[0]

    def gradient(self, inputs):
        pass

    def hessian(self, inputs):
        pass

@wp.func
def barrier(d: float) -> float:
    ret = 0.0

    if d < d2hat:
        dbydhat = d / d2hat
        ret = kappa * - wp.pow((dbydhat - 1.0), 2.0) * wp.log(dbydhat)
    return ret

@wp.func
def barrier_derivative(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = kappa * (d2hat - d) * (2.0 * wp.log(d / d2hat) + (d - d2hat) / d) / (d2hat * d2hat)

    return ret

@wp.func
def barrier_derivative2(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = -kappa * (2.0 * wp.log(d / d2hat) + (d - d2hat) / d + (d - d2hat) * (2.0 / d + d2hat / (d * d))) / (d2hat * d2hat)
    return ret

@wp.func
def vg_distance(v: wp.vec3) -> float:
    return v[1]

@wp.func
def verify_root_pt(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    beta, gamma = beta_gamma_ee(x0, x1, x2, x3)
    cond = 0.0 < beta < 1.0 and 0.0 < gamma < 1.0 # edge edge  distance
    return cond

@wp.func
def verify_root_ee(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    beta, gamma = beta_gamma_ee(x0, x1, x2, x3)
    cond = 0.0 < beta < 1.0 and 0.0 < gamma < 1.0
    

@wp.kernel
def ipc_energy_ee(ee_list: wp.array(dtype = wp.vec2i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), E: wp.array(dtype = float), bodies: wp.array(dtype = AffineBody)):
    i = wp.tid()
    ijee = ee_list[i]
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eaid = ijee[2]
    ebid = ijee[3]
    E0 = bi.edges[eaid]
    E1 = bj.edges[ebid]

    ea0 = bi.x[E0[0]]
    ea1 = bi.x[E0[1]]
    eb0 = bj.x[E1[0]]
    eb1 = bj.x[E1[1]]

    beta, gamma = beta_gamma_ee(ea0, ea1, eb0, eb1)

    cond = 0.0 < beta < 1.0 and 0.0 < gamma < 1.0 # edge edge  distance
    if cond:
        e0p, e1p, e2p = C_ee(ea0, ea1, eb0, eb1)
        d = signed_distance(e0p, e1p, e2p)
        d2 = d * d
        wp.atomic_add(E, 0, barrier(d2))
    else:
        pass
        # do nothing. Catched by point-triangle distance instead 

@wp.kernel
def ipc_energy_vg(ee_list: wp.array(dtype = wp.vec2i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), E: wp.array(dtype = float), bodies: wp.array(dtype = AffineBody)):
    i = wp.tid()
    ip = vg_list[i]
    I = ip[0]
    bi = bodies[I]
    pid = ip[1]
    p = bi.x[pid]

    d = vg_distance(p)
    wp.atomic_add(E, 0, barrier(d * d))

@wp.kernel
def ipc_energy_pt(ee_list: wp.array(dtype = wp.vec2i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), E: wp.array(dtype = float), bodies: wp.array(dtype = AffineBody)):
    i = wp.tid()
    ijpt = pt_list[i]
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[2]
    tid = ijpt[3]

    T = bj.triangles[tid]

    p = bi.x[pid]
    t0 = bj.x[T[0]]
    t1 = bj.x[T[1]]
    t2 = bj.x[T[2]]

    beta, gamma = beta_gamma_pt(p, t0, t1, t2)
    cond = 0.0 < beta < 1.0 and 0.0 < gamma < 1.0 and (beta + gamma) < 1.0  # point triangle. the projection of the point is inside the triangle
    e0p, e1p, e2p = C_vf(p, t0, t1, t2) 
    d2 = wp.length_sq(e2p)
    if cond:
        wp.atomic_add(E, 0, barrier(d2))
    else:
        # fixme: ignore point-line and point-point for now
        pass





@wp.kernel
def ipc_term_vg(vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = AffineBody), g: wp.array(dtype = wp.vec3), bsr: BSR, states: AffineBodyStates):
    i = wp.tid()

    col = vg_list[i]
    b = col[0]
    vid = col[1]

    v = bodies[b].x[vid]
    vtile = bodies[b].x0[vid]

    nabla_d = wp.vec3(0.0, 1.0, 0.0)
    d = vg_distance(v)
    d2bdb2 = barrier_derivative2(d * d)

    os = offset(i, i, bsr)
    for ii in range(4):
        theta_ii = wp.select(ii == 0, vtile[ii - 1], 1.0)
        g[4 * b + ii] += d * 2.0 * nabla_d * theta_ii 
        for jj in range(4):
            theta_jj = wp.select(jj == 0, vtile[jj - 1], 1.0)
            
            nabla_d_nabla_dT = wp.outer(nabla_d, nabla_d)

            dh = nabla_d_nabla_dT * theta_ii * theta_jj * d2bdb2
            bsr.blocks[os + ii + jj * 4] += dh
            


@wp.func
def ipc_hess(pt: wp.vec2, ij: wp.vec2, pt_type: int, d: float):
    pass

@wp.kernel
def ipc_term_pt(pt_list: wp.array(dtype = wp.vec2), ij_list: wp.array(dtype = wp.vec2), bodies: wp.array(dtype = AffineBody), g: wp.array(dtype = wp.vec3), bsr: BSR, states: AffineBodyStates):
    i = wp.tid()




    
    