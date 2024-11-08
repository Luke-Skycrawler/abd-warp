import warp as wp
import numpy as np
from const_params import scalar, vec3, mat33, mat22, vec2

@wp.func
def C_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    e0 = x1 - x0
    e1 = x3 - x2
    e2 = x2 - x0

    # e0x12 = wp.dot(wp.cross(e0, e1), wp.cross(e0, e1))

    # al = wp.dot(e0, e1) / wp.dot(e0, e0)
    # bet = (wp.dot(e1, e1) * wp.dot(e0, e2) - wp.dot(e2, e1) * wp.dot(e0, e1)) / e0x12

    # gam = (wp.dot(e0, e0) * wp.dot(e2, e1) - wp.dot(e2, e0) * wp.dot(e0, e1)) / e0x12

    # return e0, e1 - al * e0, e2 - bet * e0 - gam * e1
    alpha = wp.dot(e0, e1) / wp.dot(e0, e0)

    e1perp = e1 - alpha * e0
    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b

    e2perp = e2 - beta_gamma[0] * e0 - beta_gamma[1] * e1
    return e0, e1perp, e2perp

@wp.func
def dadx(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    e0 = x1 - x0
    e1 = x3 - x2

    al = wp.dot(e1, e0) / wp.dot(e0, e0)

    e0n2 = wp.dot(e0, e0)

    dadx0 = scalar(2.0) * al * e0 - e1
    dadx1 = -dadx0
    dadx2 = -e0
    dadx3 = -dadx2

    return dadx0 / e0n2, dadx1 / e0n2, dadx2 / e0n2, dadx3 / e0n2

@wp.func
def dbdx(x0: vec3, x1: vec3, x2: vec3, x3: vec3):

    e0 = x1 - x0
    e1 = x3 - x2
    e2 = x2 - x0

    bet, _ = beta_gamma_ee(x0, x1, x2, x3)
    
    e1n2 = wp.dot(e1, e1)
    e0n2 = wp.dot(e0, e0)
    e0d1 = wp.dot(e0, e1)
    e0d2 = wp.dot(e0, e2)
    e1d2 = wp.dot(e1, e2)
    e0x12 = wp.dot(wp.cross(e0, e1), wp.cross(e0, e1))

    dbdx0 = scalar(2.0) * bet * (e1n2 * e0 - e0d1 * e1) + e1 * (e0d1 + e1d2) - e1n2 * (e2 + e0)
    dbdx1 = scalar(2.0) * bet * (e0d1 * e1 - e1n2 * e0) - e1 * e1d2 + e1n2 * e2
    dbdx2 = scalar(2.0) * bet * (e0n2 * e1 - e0d1 * e0) + e0 * (e1n2 + e1d2) - scalar(2.0) * e0d2 * e1 - e0d1 * (e1 - e2)
    dbdx3 = scalar(2.0) * bet * (e0d1 * e0 - e0n2 * e1) - e0 * e1d2 + scalar(2.0) * e1 * e0d2 - e0d1 * e2

    return dbdx0 / e0x12, dbdx1 / e0x12, dbdx2 / e0x12, dbdx3 / e0x12

@wp.func
def dgdx(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    e0 = x1 - x0
    e1 = x3 - x2
    e2 = x2 - x0

    gam = wp.dot(e0, e1) / (wp.dot(e0, e0) * wp.dot(e1, e1) - wp.dot(e0, e1) * wp.dot(e0, e1))

    e1n2 = wp.dot(e1, e1)
    e0n2 = wp.dot(e0, e0)
    e0d1 = wp.dot(e0, e1)
    e0d2 = wp.dot(e0, e2)
    e1d2 = wp.dot(e1, e2)
    e0x12 = wp.dot(wp.cross(e0, e1), wp.cross(e0, e1))

    dgdv0 = scalar(2.0) * gam * (e1n2 * e0 - e0d1 * e1) - scalar(2.0) * e1d2 * e0 - e0n2 * e1 + e0d2 * e1 + e0d1 * (e2 + e0)
    dgdv1 = scalar(2.0) * gam * (e0d1 * e1 - e1n2 * e0) + scalar(2.0) * e1d2 * e0 - e0d2 * e1 - e0d1 * e2
    dgdv2 = scalar(2.0) * gam * (e0n2 * e1 - e0d1 * e0) - e0d1 * e0 + e0d2 * e0 + e0n2 * (e1 - e2)
    dgdv3 = scalar(2.0) * gam * (e0d1 * e0 - e0n2 * e1) - e0d2 * e0 + e0n2 * e2

    return dgdv0 / e0x12, dgdv1 / e0x12, dgdv2 / e0x12, dgdv3 / e0x12

@wp.func
def dcdx_delta_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3, ret: wp.array2d(dtype = mat33)):
    
    e0 = x1 - x0
    e1 = x3 - x2
 
    z3 = mat33(scalar(0.0))
    dadx0, dadx1, dadx2, dadx3 = dadx(x0, x1, x2, x3)
    dbdx0, dbdx1, dbdx2, dbdx3 = dbdx(x0, x1, x2, x3)
    dgdx0, dgdx1, dgdx2, dgdx3 = dgdx(x0, x1, x2, x3)

    ret[0, 0] = z3
    ret[0, 1] = z3
    ret[0, 2] = z3
    ret[0, 3] = z3
    ret[1, 0] = -wp.outer(e0, dadx0)
    ret[1, 1] = -wp.outer(e0, dadx1)
    ret[1, 2] = -wp.outer(e0, dadx2)
    ret[1, 3] = -wp.outer(e0, dadx3)
    ret[2, 0] = -wp.outer(e0, dbdx0) - wp.outer(e1, dgdx0)
    ret[2, 1] = -wp.outer(e0, dbdx1) - wp.outer(e1, dgdx1)
    ret[2, 2] = -wp.outer(e0, dbdx2) - wp.outer(e1, dgdx2)
    ret[2, 3] = -wp.outer(e0, dbdx3) - wp.outer(e1, dgdx3)

@wp.func
def beta_gamma_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    e0 = x1 - x0
    e1 = x3 - x2
    e2 = x2 - x0

    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b

    return beta_gamma[0], beta_gamma[1]

@wp.func
def dceedx_s(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    # edge-edge
    
    e0 = x1 - x0
    e1 = x3 - x2
    e2 = x2 - x0

    e0x12 = wp.dot(wp.cross(e0, e1), wp.cross(e0, e1))

    al = wp.dot(e0, e1) / wp.dot(e0, e0)
    bet, gam = beta_gamma_ee(x0, x1, x2, x3)
    mat34 = wp.matrix(dtype = scalar, shape = (3, 4))

    mat34[0, 0] = -scalar(1.0)
    mat34[0, 1] = scalar(1.0)
    mat34[0, 2] = scalar(0.0)
    mat34[0, 3] = scalar(0.0)

    mat34[1, 0] = al
    mat34[1, 1] = -al
    mat34[1, 2] = -scalar(1.0)
    mat34[1, 3] = scalar(1.0)

    mat34[2, 0] = bet - scalar(1.0)
    mat34[2, 1] = -bet
    mat34[2, 2] = gam + scalar(1.0)
    mat34[2, 3] = -gam

    return mat34
