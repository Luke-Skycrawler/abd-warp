from const_params import *
from affine_body import AffineBody, AffineBodyStates
from sparse import BSR, bsr_empty

from warp.sparse import bsr_set_from_triplets, bsr_zeros
from warp.optim.linear import cg, bicgstab

# temp
from orthogonal_energy import offset
@wp.func
def barrier(d: float) -> float:
    ret = 0.0

    if d < dhat:
        dbydhat = d / dhat
        ret = kappa * - wp.pow((dbydhat - 1.0), 2.0) * wp.log(dbydhat)
    return ret

@wp.func
def barrier_derivative(d: float) -> float:
    ret = 0.0
    if d < dhat:
        ret = kappa * (dhat - d) * (2.0 * wp.log(d / dhat) + (d - dhat) / d) / (dhat * dhat)

    return ret

@wp.func
def barrier_derivative2(d: float) -> float:
    ret = 0.0
    if d < dhat:
        ret = -kappa * (2.0 * wp.log(d / dhat) + (d - dhat) / d + (d - dhat) * (2.0 / d + dhat / (d * d))) / (dhat * dhat)
    return ret

@wp.func
def vg_distance(v: wp.vec3) -> float:
    return v[1]

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
    
@wp.kernel
def ipc_term_pt(pt_list: wp.array(dtype = wp.vec2), ij_list: wp.array(dtype = wp.vec2), bodies: wp.array(dtype = AffineBody), g: wp.array(dtype = wp.vec3), bsr: BSR, states: AffineBodyStates):
    i = wp.tid()




    
    