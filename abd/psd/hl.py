import warp as wp 
import numpy as np
from .vf import C_vf, dcvfdx_s, dcdx_delta_vf
from .ee import C_ee, dceedx_s, dcdx_delta_ee
from .matrix_prod import *
from .dcdx_delta import *

@wp.func
def distance(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    ei = x1 - x2
    ej = x3 - x2
    ek = x0 - x2
    return wp.dot(ek, wp.cross(ei, ej)) / wp.length(wp.cross(ei, ej))

@wp.func
def signed_distance(e0perp: vec3, e1perp: vec3, e2perp: vec3):
    return wp.dot(e2perp, wp.cross(e0perp, e1perp)) / wp.length(wp.cross(e0perp, e1perp))

@wp.func
def Hl(e0o: vec3, e1o: vec3, e2o: vec3):
    l = signed_distance(e0o, e1o, e2o)
    z33 = mat33(0.0)
    i33 = wp.diag(vec3(scalar(scalar(1.0))))

    l02 = -l * wp.outer(e2o, e0o) / (wp.length_sq(e2o) * wp.length_sq(e0o))
    l12 = -l * wp.outer(e2o, e1o) / (wp.length_sq(e2o) * wp.length_sq(e1o))

    e0o_unit = wp.normalize(e0o)
    e1o_unit = wp.normalize(e1o)
    block = (wp.diag(vec3(scalar(scalar(1.0)))) - wp.outer(e0o_unit, e0o_unit) - wp.outer(e1o_unit, e1o_unit))
    l00 = -l * block / wp.length_sq(e0o)
    l11 = -l * block / wp.length_sq(e1o)

    return l00, l11, l02, l12

@wp.func
def gl(l: scalar, e2o: vec3):
    z3 = vec3(scalar(0.0))
    return z3, z3, e2o * l / wp.length_sq(e2o)


@wp.kernel
def test_vf(x: wp.array(dtype = vec3), dcdx_delta: wp.array2d(dtype = mat33), ret: wp.array2d(dtype = mat33), d2Psi: wp.array2d(dtype = mat33), mat34: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i * 4 + 0]
    x1 = x[i * 4 + 1]
    x2 = x[i * 4 + 2]
    x3 = x[i * 4 + 3]

    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
    dcdx_delta_vf(x0, x1, x2, x3, dcdx_delta)
    dcdx_simple = dcvfdx_s(x0, x1, x2, x3)
    for ii in range(3):
        for jj in range(4):
            mat34[ii, jj] = dcdx_simple[ii, jj]
    l = signed_distance(e0p, e1p, e2p)
    Hl00, Hl11, Hl02, Hl12 = Hl(e0p, e1p, e2p)
    gl0, gl1, gl2 = gl(l, e2p)
    # gx = simpleTerm' * gl * 2 * l;

    # d2Psi = 2 * l * Hl + 2 * gl * gl.T
    d2Psi[0, 0] = scalar(scalar(2.0)) * l * Hl00 + scalar(scalar(2.0)) * wp.outer(gl0, gl0)
    d2Psi[0, 1] = scalar(scalar(2.0)) * wp.outer(gl0, gl1)
    d2Psi[0, 2] = scalar(scalar(2.0)) * l * Hl02 + scalar(scalar(2.0)) * wp.outer(gl0, gl2)
    d2Psi[1, 0] = scalar(scalar(2.0)) * wp.outer(gl1, gl0)
    d2Psi[1, 1] = scalar(scalar(2.0)) * l * Hl11 + scalar(scalar(2.0)) * wp.outer(gl1, gl1)
    d2Psi[1, 2] = scalar(scalar(2.0)) * l * Hl12 + scalar(scalar(2.0)) * wp.outer(gl1, gl2)
    d2Psi[2, 0] = scalar(scalar(2.0)) * l * wp.transpose(Hl02) + scalar(scalar(2.0)) * wp.outer(gl2, gl0)
    d2Psi[2, 1] = scalar(scalar(2.0)) * l * wp.transpose(Hl12) + scalar(scalar(2.0)) * wp.outer(gl2, gl1)
    d2Psi[2, 2] = scalar(scalar(2.0)) * wp.outer(gl2, gl2)

@wp.kernel
def test_ee(x: wp.array(dtype = vec3), dcdx_delta: wp.array2d(dtype = mat33), ret: wp.array2d(dtype = mat33), d2Psi: wp.array2d(dtype = mat33), mat34: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i * 4 + 0]
    x1 = x[i * 4 + 1]
    x2 = x[i * 4 + 2]
    x3 = x[i * 4 + 3]

    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)
    gl0, gl1, gl2 = gl(l, e2p)

    dcdx_delta_ee(x0, x1, x2, x3, dcdx_delta)
    dcdx_simple = dceedx_s(x0, x1, x2, x3)
    for ii in range(3):
        for jj in range(4):
            mat34[ii, jj] = dcdx_simple[ii, jj]
    Hl00, Hl11, Hl02, Hl12 = Hl(e0p, e1p, e2p)

    # d2Psi = 2 * l * Hl + 2 * gl * gl.T
    d2Psi[0, 0] = scalar(scalar(2.0)) * l * Hl00 + scalar(scalar(2.0)) * wp.outer(gl0, gl0)
    d2Psi[0, 1] = scalar(scalar(2.0)) * wp.outer(gl0, gl1)
    d2Psi[0, 2] = scalar(scalar(2.0)) * l * Hl02 + scalar(scalar(2.0)) * wp.outer(gl0, gl2)
    d2Psi[1, 0] = scalar(scalar(2.0)) * wp.outer(gl1, gl0)
    d2Psi[1, 1] = scalar(scalar(2.0)) * l * Hl11 + scalar(scalar(2.0)) * wp.outer(gl1, gl1)
    d2Psi[1, 2] = scalar(scalar(2.0)) * l * Hl12 + scalar(scalar(2.0)) * wp.outer(gl1, gl2)
    d2Psi[2, 0] = scalar(scalar(2.0)) * l * wp.transpose(Hl02) + scalar(scalar(2.0)) * wp.outer(gl2, gl0)
    d2Psi[2, 1] = scalar(scalar(2.0)) * l * wp.transpose(Hl12) + scalar(scalar(2.0)) * wp.outer(gl2, gl1)
    d2Psi[2, 2] = scalar(scalar(2.0)) * wp.outer(gl2, gl2)
    
    # for iill in range(16):
    #     ii = iill // 4
    #     ll = iill % 4
    #     hil = mat33(0.0)
    #     for jjkk in range(9):
    #         jj = jjkk // 3
    #         kk = jjkk % 3
    #         hil += dcdx_simple[jj, ii] * d2Psi[jj, kk] * dcdx_simple[kk, ll] - wp.transpose(dcdx_delta[jj, ii]) @ d2Psi[jj, kk] @ dcdx_delta[kk, ll] 

    #         wp.atomic_add(ret, ii, ll, hil)
                
@wp.func
def eig_Hl(e0p: vec3, e1p: vec3, e2p: vec3, q: wp.array2d(dtype = vec3)):
    l = signed_distance(e0p, e1p, e2p)

    e0pn = wp.length_sq(e0p)
    e1pn = wp.length_sq(e1p)
    e2pn = wp.length_sq(e2p)
    f12 = wp.sqrt(scalar(scalar(1.0)) + scalar(scalar(4.0)) * (e1pn / e2pn))
    f02 = wp.sqrt(scalar(scalar(1.0)) + scalar(scalar(4.0)) * (e0pn / e2pn))

    lam0 = -l / (scalar(scalar(2.0)) * e1pn) * (scalar(scalar(1.0)) + f12)
    lam1 = -l / (scalar(scalar(2.0)) * e1pn) * (scalar(scalar(1.0)) - f12)
    lam2 = -l / (scalar(scalar(2.0)) * e0pn) * (scalar(scalar(1.0)) + f02)
    lam3 = -l / (scalar(scalar(2.0)) * e0pn) * (scalar(scalar(1.0)) - f02)

    z31 = vec3(0.0)

    omega0 = lam0 / (lam0 - l / e2pn)
    omega1 = lam1 / (lam1 - l / e2pn)
    omega2 = lam2 / (lam2 - l / e2pn)
    omega3 = lam3 / (lam3 - l / e2pn)
    
    # bundles to a eigen vector every 3 items
    q[0, 0] = z31
    q[0, 1] = e2p
    q[0, 2] = omega0 * e1p
    q[1, 0] = z31
    q[1, 1] = e2p
    q[1, 2] = omega1 * e1p
    q[2, 0] = e2p
    q[2, 1] = z31
    q[2, 2] = omega2 * e0p
    q[3, 0] = e2p
    q[3, 1] = z31
    q[3, 2] = omega3 * e0p

    return lam0 * scalar(scalar(2.0)) * l, lam1 * scalar(scalar(2.0)) * l, lam2 * scalar(scalar(2.0)) * l, lam3 * scalar(scalar(2.0)) * l
    # return  z31, e2p, omega0 * e1p,\
    #         z31, e2p, omega1 * e1p,\
    #         e2p, z31, omega2 * e0p,\
    #         e2p, z31, omega3 * e0p

@wp.func
def eig_Hl_tid(e0p: vec3, e1p: vec3, e2p: vec3, q: wp.array2d(dtype = vec3), tid: int):
    l = signed_distance(e0p, e1p, e2p)

    e0pn = wp.length_sq(e0p)
    e1pn = wp.length_sq(e1p)
    e2pn = wp.length_sq(e2p)
    f12 = wp.sqrt(scalar(scalar(1.0)) + scalar(scalar(4.0)) * (e1pn / e2pn))
    f02 = wp.sqrt(scalar(scalar(1.0)) + scalar(scalar(4.0)) * (e0pn / e2pn))

    lam0 = -l / (scalar(scalar(2.0)) * e1pn) * (scalar(scalar(1.0)) + f12)
    lam1 = -l / (scalar(scalar(2.0)) * e1pn) * (scalar(scalar(1.0)) - f12)
    lam2 = -l / (scalar(scalar(2.0)) * e0pn) * (scalar(scalar(1.0)) + f02)
    lam3 = -l / (scalar(scalar(2.0)) * e0pn) * (scalar(scalar(1.0)) - f02)

    z31 = vec3(scalar(scalar(0.0)))

    omega0 = lam0 / (lam0 - l / e2pn)
    omega1 = lam1 / (lam1 - l / e2pn)
    omega2 = lam2 / (lam2 - l / e2pn)
    omega3 = lam3 / (lam3 - l / e2pn)
    
    # bundles to a eigen vector every 3 items
    q[tid, 0 * 3 + 0] = z31
    q[tid, 0 * 3 + 1] = e2p
    q[tid, 0 * 3 + 2] = omega0 * e1p
    q[tid, 1 * 3 + 0] = z31
    q[tid, 1 * 3 + 1] = e2p
    q[tid, 1 * 3 + 2] = omega1 * e1p
    q[tid, 2 * 3 + 0] = e2p
    q[tid, 2 * 3 + 1] = z31
    q[tid, 2 * 3 + 2] = omega2 * e0p
    q[tid, 3 * 3 + 0] = e2p
    q[tid, 3 * 3 + 1] = z31
    q[tid, 3 * 3 + 2] = omega3 * e0p

    return lam0 * scalar(scalar(2.0)) * l, lam1 * scalar(scalar(2.0)) * l, lam2 * scalar(scalar(2.0)) * l, lam3 * scalar(scalar(2.0)) * l
    # return  z31, e2p, omega0 * e1p,\
    #         z31, e2p, omega1 * e1p,\
    #         e2p, z31, omega2 * e0p,\
    #         e2p, z31, omega3 * e0p

@wp.kernel
def verify_eig_sys_ee(x: wp.array(dtype = vec3), q: wp.array2d(dtype = vec3), lam: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i * 9 + 0]
    x1 = x[i * 9 + 1]
    x2 = x[i * 9 + 2]
    x3 = x[i * 9 + 3]

    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    lam0, lam1, lam2, lam3 = eig_Hl(e0p, e1p, e2p, q)
    l = signed_distance(e0p, e1p, e2p)

    lam[i, 0] = lam0
    lam[i, 1] = lam1
    lam[i, 2] = lam2
    lam[i, 3] = lam3
    lam[i, 4] = scalar(scalar(2.0))

    gl0, gl1, gl2 = gl(l, e2p)
    q[4, 0] = gl0
    q[4, 1] = gl1
    q[4, 2] = gl2

@wp.kernel
def verify_eig_sys_vf(x: wp.array(dtype = vec3), q: wp.array2d(dtype = vec3), lam: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i * 9 + 0]
    x1 = x[i * 9 + 1]
    x2 = x[i * 9 + 2]
    x3 = x[i * 9 + 3]

    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
    lam0, lam1, lam2, lam3 = eig_Hl(e0p, e1p, e2p, q)
    l = signed_distance(e0p, e1p, e2p)

    lam[i, 0] = lam0
    lam[i, 1] = lam1
    lam[i, 2] = lam2
    lam[i, 3] = lam3
    lam[i, 4] = scalar(scalar(2.0))

    gl0, gl1, gl2 = gl(l, e2p)
    q[4, 0] = gl0
    q[4, 1] = gl1
    q[4, 2] = gl2

def project_psd(A, Q, Lambda):
    n = A.shape[0]
    ret = np.copy(A)
    for i in range(n):
        if Lambda[i, i] < scalar(scalar(0.0)):
            qi = Q[:, i].reshape((n))
            term = (Lambda[i, i] / np.dot(qi, qi))
            ret -= np.outer(qi, qi) * term
    return ret

@wp.kernel
def test_distance(x: wp.array(dtype = vec3)):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)
    l2 = l * l
    e2n = wp.length_sq(e2p)
    print(l2)
    print(e2n)
    # verified l2 is indeed e2n

