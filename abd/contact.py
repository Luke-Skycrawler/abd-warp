from const_params import *
from affine_body import AffineBody
from fetch_utils import fetch_vertex, fetch_xk, fetch_ee, fetch_pt, fetch_pt_xk, fetch_ee_xk
from ccd import vg_collison_time, pt_collision_time, ee_collision_time

@wp.kernel
def toi_vg(bodies: wp.array(dtype = AffineBody), vg_list: wp.array(dtype = wp.vec2i), toi: wp.array(dtype = float)):

    tid = wp.tid()
    xt0 = fetch_vertex(vg_list[tid], bodies)
    xt1 = fetch_xk(vg_list[tid], bodies)

    t = vg_collison_time(xt0, xt1)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)

@wp.kernel
def toi_pt(bodies: wp.array(dtype = AffineBody), pt_list: wp.array(dtype = vec5i), toi: wp.array(dtype = float)):

    tid = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[tid], bodies)
    pk, t0k, t1k, t2k = fetch_pt_xk(pt_list[tid], bodies)
    
    t = pt_collision_time(p, t0, t1, t2, pk, t0k, t1k, t2k)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)

@wp.kernel
def toi_ee(bodies: wp.array(dtype = AffineBody), ee_list: wp.array(dtype = vec5i), toi: wp.array(dtype = float)):

    tid = wp.tid()
    ea0, ea1, eb0, eb1 = fetch_ee(ee_list[tid], bodies)
    ea0k, ea1k, eb0k, eb1k = fetch_ee_xk(ee_list[tid], bodies)
    
    t = ee_collision_time(ea0, ea1, eb0, eb1, ea0k, ea1k, eb0k, eb1k)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)


        