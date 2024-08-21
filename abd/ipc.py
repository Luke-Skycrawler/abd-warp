from const_params import *
from typing import Any
from psd.ee import beta_gamma_ee, C_ee
from psd.hl import signed_distance
from psd.vf import beta_gamma_pt, C_vf
from affine_body import AffineBody, fetch_ee, fetch_pt, vg_distance


class IPCContactEnergy:
    def __init__(self) -> None:
        pass

    def energy(self, inputs):
        ee_list = inputs[0]
        pt_list = inputs[1]
        vg_list = inputs[2]
        E = wp.zeros((1, ), dtype =float)
        inputs.append(E)
        wp.launch(ipc_energy_ee, dim = ee_list.shape, inputs = inputs)
        wp.launch(ipc_energy_pt, dim = pt_list.shape, inputs = inputs)
        wp.launch(ipc_energy_vg, dim = vg_list.shape, inputs = inputs)
        return E.numpy()[0]

    def gradient(self, inputs):
        pass

    def hessian(self, inputs):
        pass

    def gh(self, inputs):

        vg_list, bodies, g, blocks = inputs
        dim = vg_list.shape[0]
        wp.launch(ipc_term_vg, dim = (dim, ), inputs = inputs)
        return dim

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

    

@wp.kernel
def ipc_energy_ee(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    ijee = ee_list[i]
    ea0, ea1, eb0, eb1 = fetch_ee(ijee, bodies)
    beta, gamma = beta_gamma_ee(ea0, ea1, eb0, eb1)

    cond = 0.0 < beta < 1.0 and 0.0 < gamma < 1.0 # edge edge  distance
    if cond:
        e0p, e1p, e2p = C_ee(ea0, ea1, eb0, eb1)
        d = signed_distance(e0p, e1p, e2p)
        d2 = d * d
        wp.atomic_add(E, 0, barrier(d2))
    else:
        pass
        # do nothing. Caught by point-triangle distance instead 

@wp.kernel
def ipc_energy_vg(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    ip = vg_list[i]
    I = ip[0]
    bi = bodies[I]
    pid = ip[1]
    p = bi.x[pid]

    d = vg_distance(p)
    wp.atomic_add(E, 0, barrier(d * d))

@wp.kernel
def ipc_energy_pt(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)

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
def ipc_term_vg(vg_list: wp.array(dtype = wp.vec2i), 
                bodies: wp.array(dtype = Any), 
                g: wp.array(dtype = wp.vec3), blocks: wp.array(dtype = wp.mat33)):
    i = wp.tid()

    col = vg_list[i]
    b = col[0]
    vid = col[1]

    v = bodies[b].x[vid]
    vtile = bodies[b].x0[vid]

    nabla_d = wp.vec3(0.0, 1.0, 0.0)
    d = vg_distance(v)
    d2bdb2 = barrier_derivative2(d * d)
    dbdd = barrier_derivative(d * d)

    os = b * 16
    for ii in range(4):
        theta_ii = wp.select(ii == 0, vtile[ii - 1], 1.0)
        wp.atomic_add(g, 4 * b + ii, d * 2.0 * nabla_d * theta_ii * dbdd)
        # g[4 * b + ii] += d * 2.0 * nabla_d * theta_ii 
        for jj in range(4):
            theta_jj = wp.select(jj == 0, vtile[jj - 1], 1.0)
            
            nabla_d_nabla_dT = wp.outer(nabla_d, nabla_d)

            dh = nabla_d_nabla_dT * (theta_ii * theta_jj * d2bdb2 * 4.0 * d * d)
            # blocks[os + ii + jj * 4] += dh
            wp.atomic_add(blocks, os + ii + jj * 4, dh)
            

@wp.kernel
def ipc_term_pt(nij: int, pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), g: wp.array(dtype = wp.vec3), blocks: wp.array(dtype = wp.mat33)):
    i = wp.tid()

    n_bodies = bodies.shape[0]

    idx = pt_list[i][2]

    offset_upper = 16 * (n_bodies + idx)
    offset_lower = 16 * (n_bodies + idx + nij)

    p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)
    





if __name__ == "__main__":

    wp.init()
    vg_list = wp.zeros((1,), dtype = wp.vec2i)
    _bodies = []
    a = AffineBody()
    a.x = wp.zeros((1, ), dtype = wp.vec3)
    a.x0 = wp.zeros((1, ), dtype = wp.vec3)
    a.xk = wp.zeros((1, ), dtype = wp.vec3)
    a.x_view = wp.zeros((1, ), dtype = wp.vec3)
    a.triangles = wp.zeros((1, 3), dtype = int)
    a.edges = wp.zeros((1, 2), dtype = int)

    _bodies.append(a)
    bodies = wp.array(_bodies, dtype = AffineBody)

    g = wp.zeros((4, ), dtype = wp.vec3)
    blocks = wp.zeros((16, ), dtype = wp.mat33)

    wp.launch(ipc_term_vg, (1, ), inputs = [vg_list, bodies, g, blocks])
    # wp.launch(ipc_term_vg, (1, ), inputs = [vg_list, g, blocks])


    
    