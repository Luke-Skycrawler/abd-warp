from const_params import *
from cubic_roots import cubic_roots
from psd.ee import beta_gamma_ee
from psd.vf import beta_gamma_pt
from affine_body import fetch_vertex, fetch_xk, fetch_ee, fetch_pt, fetch_pt_xk, fetch_ee_xk, vg_distance
from typing import Any
import abdtk

@wp.kernel
def toi_vg(bodies: wp.array(dtype = Any), vg_list: wp.array(dtype = wp.vec2i), toi: wp.array(dtype = scalar)):

    tid = wp.tid()
    xt0 = fetch_vertex(vg_list[tid], bodies)
    xt1 = fetch_xk(vg_list[tid], bodies)

    t = vg_collison_time(xt0, xt1)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)

@wp.kernel
def toi_pt(bodies: wp.array(dtype = Any), pt_list: wp.array(dtype = vec5i), toi: wp.array(dtype = scalar)):

    tid = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[tid], bodies)
    pk, t0k, t1k, t2k = fetch_pt_xk(pt_list[tid], bodies)
    
    t = pt_collision_time(p, t0, t1, t2, pk, t0k, t1k, t2k)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)

@wp.kernel
def toi_ee(bodies: wp.array(dtype = Any), ee_list: wp.array(dtype = vec5i), toi: wp.array(dtype = scalar)):

    tid = wp.tid()
    ea0, ea1, eb0, eb1 = fetch_ee(ee_list[tid], bodies)
    ea0k, ea1k, eb0k, eb1k = fetch_ee_xk(ee_list[tid], bodies)
    
    t = ee_collision_time(ea0, ea1, eb0, eb1, ea0k, ea1k, eb0k, eb1k)
    if t < 1.0:
        wp.atomic_min(toi, 0, t)


        
@wp.func
def verify_root_pt(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    beta, gamma = beta_gamma_pt(x0, x1, x2, x3)
    cond = scalar(0.0) < beta < scalar(1.0) and scalar(0.0) < gamma < scalar(1.0) and scalar(0.0) < beta + gamma < scalar(1.0)
    return cond

@wp.func
def verify_root_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    beta, gamma = beta_gamma_ee(x0, x1, x2, x3)
    cond = scalar(0.0) < beta < scalar(1.0) and scalar(0.0) < -gamma < scalar(1.0)
    return cond

@wp.func
def build_and_solve_4_points_coplanar(
    p0_t0: vec3, p1_t0: vec3, p2_t0: vec3, p3_t0: vec3,
    p0_t1: vec3, p1_t1: vec3, p2_t1: vec3, p3_t1: vec3
):
    a1 = mat33(p1_t1, p2_t1, p3_t1)
    a2 = mat33(p0_t1, p2_t1, p3_t1)
    a3 = mat33(p0_t1, p1_t1, p3_t1)
    a4 = mat33(p0_t1, p1_t1, p2_t1)

    b1 = mat33(p1_t0, p2_t0, p3_t0)
    b2 = mat33(p0_t0, p2_t0, p3_t0)
    b3 = mat33(p0_t0, p1_t0, p3_t0)
    b4 = mat33(p0_t0, p1_t0, p2_t0)

    a1 -= b1
    a2 -= b2
    a3 -= b3
    a4 -= b4

    t = det_polynomial(a1, b1) - det_polynomial(a2, b2) + det_polynomial(a3, b3) - det_polynomial(a4, b4)

    found, roots = cubic_roots(t, 0.0, 1.0)
    return found, roots


@wp.func
def det_polynomial(a: mat33, b: mat33) -> vec4:
    pos_polynomial = vec4(0.0, 0.0, 0.0, 0.0)
    neg_polynomial = vec4(0.0, 0.0, 0.0, 0.0)

    c11c22c33 = mat23(
        a[0, 0], a[1, 1], a[2, 2],
        b[0, 0], b[1, 1], b[2, 2]
    )
    c12c23c31 = mat23(
        a[0, 1], a[1, 2], a[2, 0],
        b[0, 1], b[1, 2], b[2, 0]
    )
    c13c21c32 = mat23(
        a[0, 2], a[1, 0], a[2, 1],
        b[0, 2], b[1, 0], b[2, 1]
    )
    c11c23c32 = mat23(
        a[0, 0], a[1, 2], a[2, 1],
        b[0, 0], b[1, 2], b[2, 1]
    )
    c12c21c33 = mat23(
        a[0, 1], a[1, 0], a[2, 2],
        b[0, 1], b[1, 0], b[2, 2]
    )
    c13c22c31 = mat23(
        a[0, 2], a[1, 1], a[2, 0],
        b[0, 2], b[1, 1], b[2, 0]
    )

    pos_polynomial += cubic_binomial(c11c22c33[0], c11c22c33[1])
    pos_polynomial += cubic_binomial(c12c23c31[0], c12c23c31[1])
    pos_polynomial += cubic_binomial(c13c21c32[0], c13c21c32[1])
    neg_polynomial += cubic_binomial(c11c23c32[0], c11c23c32[1])
    neg_polynomial += cubic_binomial(c12c21c33[0], c12c21c33[1])
    neg_polynomial += cubic_binomial(c13c22c31[0], c13c22c31[1])

    return pos_polynomial - neg_polynomial


@wp.func
def cubic_binomial(a: vec3, b:vec3):
    return vec4(
        b[0] * b[1] * b[2],
        a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2],
        a[0] * a[1] * b[2] + a[0] * b[1] * a[2] + b[0] * a[1] * a[2],
        a[0] * a[1] * a[2]
    )



@wp.func
def pt_collision_time(
    p0_t0: vec3, p1_t0: vec3, p2_t0: vec3, p3_t0: vec3,

    p0_t1: vec3, p1_t1: vec3, p2_t1: vec3, p3_t1: vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(p0_t0, p1_t0, p2_t0, p3_t0, p0_t1, p1_t1, p2_t1, p3_t1)

    root = scalar(1.0)
    true_root = bool(False)
    for i in range(n_roots):
        root = roots[i]
        p0t = wp.lerp(p0_t0, p0_t1, root)
        p1t = wp.lerp(p1_t0, p1_t1, root)
        p2t = wp.lerp(p2_t0, p2_t1, root)
        p3t = wp.lerp(p3_t0, p3_t1, root)
        true_root = verify_root_pt(p0t, p1t, p2t, p3t)
        if true_root:
            break

    if not true_root:
        root = 1.0

    return root

@wp.func
def ee_collision_time(
    ei0_t0: vec3,  
    ei1_t0: vec3, 
    ej0_t0: vec3, 
    ej1_t0: vec3, 
    ei0_t1: vec3,
    ei1_t1: vec3,
    ej0_t1: vec3,
    ej1_t1: vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(ei0_t0, ei1_t0, ej0_t0, ej1_t0, ei0_t1, ei1_t1, ej0_t1, ej1_t1)

    root = scalar(1.0)
    true_root = bool(False)
    for i in range(n_roots):
        root = roots[i]
        ei0 = wp.lerp(ei0_t0, ei0_t1, root)
        ei1 = wp.lerp(ei1_t0, ei1_t1, root)
        ej0 = wp.lerp(ej0_t0, ej0_t1, root)
        ej1 = wp.lerp(ej1_t0, ej1_t1, root)
        true_root = verify_root_ee(ei0, ei1, ej0, ej1)
        if true_root:
            break

    if not true_root:
        root = 1.0

    return root
        
@wp.func
def vg_collison_time(pt0: vec3, pt1: vec3):
    toi = scalar(1.0)
    d1 = vg_distance(pt1)
    if d1 < 0:
        d0 = vg_distance(pt0)
        toi = d0 / (d0 - d1)
    return toi

@wp.kernel
def get_vertices_pt(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = vec3)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)
    x[i, 0] = p
    x[i, 1] = t0
    x[i, 2] = t1
    x[i, 3] = t2

@wp.kernel
def get_vertices_pt_xk(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = vec3)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt_xk(pt_list[i], bodies)
    x[i, 0] = p
    x[i, 1] = t0
    x[i, 2] = t1
    x[i, 3] = t2

@wp.kernel
def get_vertices_ee_xk(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = vec3)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_ee_xk(ee_list[i], bodies)
    x[i, 0] = p
    x[i, 1] = t0
    x[i, 2] = t1
    x[i, 3] = t2

@wp.kernel
def get_vertices_ee(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = vec3)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_ee(ee_list[i], bodies)
    x[i, 0] = p
    x[i, 1] = t0
    x[i, 2] = t1
    x[i, 3] = t2



def toi_pt_abdtk(bodies, pt_list):
    npt = pt_list.shape[0]
    x_t0 = wp.zeros((pt_list.shape[0], 4), dtype = vec3)
    x_t1 = wp.zeros_like(x_t0)
    wp.launch(get_vertices_pt, dim = pt_list.shape, inputs = [pt_list, bodies, x_t0])

    wp.launch(get_vertices_pt_xk, dim = pt_list.shape, inputs = [pt_list, bodies, x_t1])    
    
    _t0 = x_t0.numpy()
    _t1 = x_t1.numpy()

    toi = 1.0

    for i in range(npt):
        t0 = _t0[i]
        t1 = _t1[i]
        t = abdtk.pt_collision_time(t0[0], t0[1], t0[2], t0[3], t1[0], t1[1], t1[2], t1[3])
        toi = min(toi, t)
        
    return toi

def toi_ee_abdtk(bodies, ee_list):  
    nee = ee_list.shape[0]
    x_t0 = wp.zeros((ee_list.shape[0], 4), dtype = vec3)
    x_t1 = wp.zeros_like(x_t0)
    wp.launch(get_vertices_ee, dim = ee_list.shape, inputs = [ee_list, bodies, x_t0])

    wp.launch(get_vertices_ee_xk, dim = ee_list.shape, inputs = [ee_list, bodies, x_t1])    
    
    _t0 = x_t0.numpy()
    _t1 = x_t1.numpy()

    toi = 1.0

    for i in range(nee):
        t0 = _t0[i]
        t1 = _t1[i]
        t = abdtk.pt_collision_time(t0[0], t0[1], t0[2], t0[3], t1[0], t1[1], t1[2], t1[3])
        toi = min(toi, t)
        
    return toi

        
    
    
