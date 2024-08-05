from const_params import *
from cubic_roots import cubic_roots
from ipc import verify_root_ee, verify_root_pt, vg_distance
mat23 = wp.types.matrix(2, 3, float)
@wp.func
def build_and_solve_4_points_coplanar(
    p0_t0: wp.vec3, p1_t0: wp.vec3, p2_t0: wp.vec3, p3_t0: wp.vec3,
    p0_t1: wp.vec3, p1_t1: wp.vec3, p2_t1: wp.vec3, p3_t1: wp.vec3
):
    a1 = wp.mat33(p1_t1, p2_t1, p3_t1)
    a2 = wp.mat33(p0_t1, p2_t1, p3_t1)
    a3 = wp.mat33(p0_t1, p1_t1, p3_t1)
    a4 = wp.mat33(p0_t1, p1_t1, p2_t1)

    b1 = wp.mat33(p1_t0, p2_t0, p3_t0)
    b2 = wp.mat33(p0_t0, p2_t0, p3_t0)
    b3 = wp.mat33(p0_t0, p1_t0, p3_t0)
    b4 = wp.mat33(p0_t0, p1_t0, p2_t0)

    a1 -= b1
    a2 -= b2
    a3 -= b3
    a4 -= b4

    t = det_polynomial(a1, b1) - det_polynomial(a2, b2) + det_polynomial(a3, b3) - det_polynomial(a4, b4)

    found, roots = cubic_roots(t, 0.0, 1.0)
    return found, roots


@wp.func
def det_polynomial(a: wp.mat33, b: wp.mat33) -> wp.vec4:
    pos_polynomial = wp.vec4(0.0, 0.0, 0.0, 0.0)
    neg_polynomial = wp.vec4(0.0, 0.0, 0.0, 0.0)

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
def cubic_binomial(a: wp.vec3, b:wp.vec3):
    return wp.vec4(
        b[0] * b[1] * b[2],
        a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2],
        a[0] * a[1] * b[2] + a[0] * b[1] * a[2] + b[0] * a[1] * a[2],
        a[0] * a[1] * a[2]
    )



@wp.func
def collision_time(
    p0_t0: wp.vec3, p1_t0: wp.vec3, p2_t0: wp.vec3, p3_t0: wp.vec3,

    p0_t1: wp.vec3, p1_t1: wp.vec3, p2_t1: wp.vec3, p3_t1: wp.vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(p0_t0, p1_t0, p2_t0, p3_t0, p0_t1, p1_t1, p2_t1, p3_t1)

    root = float(0.0)
    true_root = False
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
    ei0_e0: wp.vec3, ei0_e1: wp.vec3, 
    ej0_e0: wp.vec3, ej0_e1: wp.vec3,
    ei1_e0: wp.vec3, ei1_e1: wp.vec3,
    ej1_e0: wp.vec3, ej1_e1: wp.vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(ei0_e0, ei0_e1, ei1_e0, ei1_e1, ej0_e0, ej0_e1, ej1_e0, ej1_e1)

    root = float(0.0)
    true_root = wp.bool(False)
    for i in range(n_roots):
        root = roots[i]
        ei0 = wp.lerp(ei0_e0, ei0_e1, root)
        ei1 = wp.lerp(ei1_e0, ei1_e1, root)
        ej0 = wp.lerp(ej0_e0, ej0_e1, root)
        ej1 = wp.lerp(ej1_e0, ej1_e1, root)
        true_root = verify_root_ee(ei0, ei1, ej0, ej1)
        if true_root:
            break

    if not true_root:
        root = 1.0

    return root
        
@wp.func
def vg_collison_time(pt0: wp.vec3, pt1: wp.vec3):
    toi = float(1.0)
    d1 = vg_distance(pt1)
    if d1 < 0:
        d0 = vg_distance(pt0)
        toi = d0 / (d0 - d1)
    return toi
    
