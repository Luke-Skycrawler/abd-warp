from ipcsk.barrier import *
from typing import Any
from psd.ee import beta_gamma_ee, C_ee, dceedx_s
from psd.vf import beta_gamma_pt, C_vf, dcvfdx_s
from affine_body import AffineBody, fetch_ee, fetch_pt, vg_distance, fetch_vertex, fetch_pt_xk, fetch_ee_xk, fetch_pt_x0, fetch_ee_x0
from psd.hl import signed_distance
from ccd import verify_root_pt, verify_root_ee

@wp.kernel
def ipc_energy_ee(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    ijee = ee_list[i]
    ea0, ea1, eb0, eb1 = fetch_ee_xk(ijee, bodies)
    cond = verify_root_ee(ea0, ea1, eb0, eb1)
    e0p, e1p, e2p = C_ee(ea0, ea1, eb0, eb1)
    d = signed_distance(e0p, e1p, e2p)
    d2 = d * d
    if cond:
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
    p = bi.xk[pid]

    d = vg_distance(p)
    wp.atomic_add(E, 0, barrier(d * d))

@wp.kernel
def ipc_energy_pt(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt_xk(pt_list[i], bodies)

    cond = verify_root_pt(p, t0, t1, t2)
    e0p, e1p, e2p = C_vf(p, t0, t1, t2) 
    d2 = wp.length_sq(e2p)
    if cond:
        wp.atomic_add(E, 0, barrier(d2))
    else:
        # fixme: ignore point-line and point-point for now
        pass



            
