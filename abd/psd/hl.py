import warp as wp 
import numpy as np
from vf import C_vf, dcvfdx_s, dcdx_delta_vf
from ee import C_ee, dceedx_s, dcdx_delta_ee

@wp.func
def distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    ei = x1 - x2
    ej = x3 - x2
    ek = x0 - x2
    return wp.dot(ek, wp.cross(ei, ej)) / wp.length(wp.cross(ei, ej))

@wp.func
def signed_distance(e0perp: wp.vec3, e1perp: wp.vec3, e2perp: wp.vec3):
    return wp.dot(e2perp, wp.cross(e0perp, e1perp)) / wp.length(wp.cross(e0perp, e1perp))

@wp.func
def Hl(e0o: wp.vec3, e1o: wp.vec3, e2o: wp.vec3):
    l = signed_distance(e0o, e1o, e2o)
    z33 = wp.mat33(0.0)
    i33 = wp.diag(wp.vec3(1.0))

    l02 = -l * wp.outer(e2o, e0o) / (wp.length_sq(e2o) * wp.length_sq(e0o))
    l12 = -l * wp.outer(e2o, e1o) / (wp.length_sq(e2o) * wp.length_sq(e1o))

    e0o_unit = wp.normalize(e0o)
    e1o_unit = wp.normalize(e1o)
    block = (wp.diag(wp.vec3(1.0)) - wp.outer(e0o_unit, e0o_unit) - wp.outer(e1o_unit, e1o_unit))
    l00 = -l * block / wp.length_sq(e0o)
    l11 = -l * block / wp.length_sq(e1o)

    return l00, l11, l02, l12

@wp.func
def gl(l: float, e2o: wp.vec3):
    z3 = wp.vec3(0.0)
    return z3, z3, e2o * l / wp.length_sq(e2o)


@wp.kernel
def test_vf(x: wp.array(dtype = wp.vec3), dcdx_delta: wp.array2d(dtype = wp.mat33), ret: wp.array2d(dtype = wp.mat33), d2Psi: wp.array2d(dtype = wp.mat33), mat34: wp.array2d(dtype = float)):
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
    d2Psi[0, 0] = 2.0 * l * Hl00 + 2.0 * wp.outer(gl0, gl0)
    d2Psi[0, 1] = 2.0 * wp.outer(gl0, gl1)
    d2Psi[0, 2] = 2.0 * l * Hl02 + 2.0 * wp.outer(gl0, gl2)
    d2Psi[1, 0] = 2.0 * wp.outer(gl1, gl0)
    d2Psi[1, 1] = 2.0 * l * Hl11 + 2.0 * wp.outer(gl1, gl1)
    d2Psi[1, 2] = 2.0 * l * Hl12 + 2.0 * wp.outer(gl1, gl2)
    d2Psi[2, 0] = 2.0 * l * wp.transpose(Hl02) + 2.0 * wp.outer(gl2, gl0)
    d2Psi[2, 1] = 2.0 * l * wp.transpose(Hl12) + 2.0 * wp.outer(gl2, gl1)
    d2Psi[2, 2] = 2.0 * wp.outer(gl2, gl2)

@wp.kernel
def test_ee(x: wp.array(dtype = wp.vec3), dcdx_delta: wp.array2d(dtype = wp.mat33), ret: wp.array2d(dtype = wp.mat33), d2Psi: wp.array2d(dtype = wp.mat33), mat34: wp.array2d(dtype = float)):
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
    d2Psi[0, 0] = 2.0 * l * Hl00 + 2.0 * wp.outer(gl0, gl0)
    d2Psi[0, 1] = 2.0 * wp.outer(gl0, gl1)
    d2Psi[0, 2] = 2.0 * l * Hl02 + 2.0 * wp.outer(gl0, gl2)
    d2Psi[1, 0] = 2.0 * wp.outer(gl1, gl0)
    d2Psi[1, 1] = 2.0 * l * Hl11 + 2.0 * wp.outer(gl1, gl1)
    d2Psi[1, 2] = 2.0 * l * Hl12 + 2.0 * wp.outer(gl1, gl2)
    d2Psi[2, 0] = 2.0 * l * wp.transpose(Hl02) + 2.0 * wp.outer(gl2, gl0)
    d2Psi[2, 1] = 2.0 * l * wp.transpose(Hl12) + 2.0 * wp.outer(gl2, gl1)
    d2Psi[2, 2] = 2.0 * wp.outer(gl2, gl2)
    
    # for iill in range(16):
    #     ii = iill // 4
    #     ll = iill % 4
    #     hil = wp.mat33(0.0)
    #     for jjkk in range(9):
    #         jj = jjkk // 3
    #         kk = jjkk % 3
    #         hil += dcdx_simple[jj, ii] * d2Psi[jj, kk] * dcdx_simple[kk, ll] - wp.transpose(dcdx_delta[jj, ii]) @ d2Psi[jj, kk] @ dcdx_delta[kk, ll] 

    #         wp.atomic_add(ret, ii, ll, hil)
    
@wp.kernel
def d2Psidx2(ret: wp.array2d(dtype = wp.mat33), d2Psi: wp.array2d(dtype = wp.mat33), dcdx_simple: wp.array2d(dtype = float), dcdx_delta: wp.array2d(dtype = wp.mat33)):
    ii, ll = wp.tid()
    h = wp.mat33(0.0)
    # H = dcdx_simple.T @ d2Psi @ dcdx_simple - dcdx_delta.T @ d2Psi @ dcdx_delta
    for jj in range(3):
        for kk in range(4):
            h += dcdx_simple[jj, ii] * d2Psi[jj, kk] * dcdx_simple[kk, ll] - wp.transpose(dcdx_delta[jj, ii]) @ d2Psi[jj, kk] @ dcdx_delta[kk, ll]
    ret[ii, ll] = h
        
    
@wp.func
def eig_Hl(e0p: wp.vec3, e1p: wp.vec3, e2p: wp.vec3, ret: wp.array2d(dtype = wp.vec3)):
    l = signed_distance(e0p, e1p, e2p)

    e0pn = wp.length_sq(e0p)
    e1pn = wp.length_sq(e1p)
    e2pn = wp.length_sq(e2p)
    f12 = wp.sqrt(1.0 + 4.0 * (e1pn / e2pn))
    f02 = wp.sqrt(1.0 + 4.0 * (e0pn / e2pn))

    lam0 = -l / (2.0 * e1pn) * (1.0 + f12)
    lam1 = -l / (2.0 * e1pn) * (1.0 - f12)
    lam2 = -l / (2.0 * e0pn) * (1.0 + f02)
    lam3 = -l / (2.0 * e0pn) * (1.0 - f02)

    z31 = wp.vec3(0.0)

    omega0 = lam0 / (lam0 - l / e2pn)
    omega1 = lam1 / (lam1 - l / e2pn)
    omega2 = lam2 / (lam2 - l / e2pn)
    omega3 = lam3 / (lam3 - l / e2pn)
    
    ret[0, 0] = z31
    ret[0, 1] = e2p
    ret[0, 2] = omega0 * e1p
    ret[1, 0] = z31
    ret[1, 1] = e2p
    ret[1, 2] = omega1 * e1p
    ret[2, 0] = e2p
    ret[2, 1] = z31
    ret[2, 2] = omega2 * e0p
    ret[3, 0] = e2p
    ret[3, 1] = z31
    ret[3, 2] = omega3 * e0p

    return lam0 * 2.0 * l, lam1 * 2.0 * l, lam2 * 2.0 * l, lam3 * 2.0 * l
    # return  z31, e2p, omega0 * e1p,\
    #         z31, e2p, omega1 * e1p,\
    #         e2p, z31, omega2 * e0p,\
    #         e2p, z31, omega3 * e0p

@wp.kernel
def verify_eig_sys_ee(x: wp.array(dtype = wp.vec3), q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
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
    lam[i, 4] = 2.0

    gl0, gl1, gl2 = gl(l, e2p)
    q[4, 0] = gl0
    q[4, 1] = gl1
    q[4, 2] = gl2

@wp.kernel
def verify_eig_sys_vf(x: wp.array(dtype = wp.vec3), q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
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
    lam[i, 4] = 2.0

    gl0, gl1, gl2 = gl(l, e2p)
    q[4, 0] = gl0
    q[4, 1] = gl1
    q[4, 2] = gl2

    
def project_psd(A, Q, Lambda):
    n = A.shape[0]
    ret = np.copy(A)
    for i in range(n):
        if Lambda[i, i] < 0.0:
            qi = Q[:, i].reshape((n))
            term = (Lambda[i, i] / np.dot(qi, qi))
            ret -= np.outer(qi, qi) * term
    return ret

if __name__ == "__main__":
    wp.config.enable_backward = False
    wp.config.max_unroll = 0
    wp.init()
    np.set_printoptions(precision=4, suppress=True)    
    arr = np.array([
        [0.4360, 0.0259, 0.5497],
        [0.4353, 0.4204, 0.3303],
        [0.2046, 0.6193, 0.2997],
        [0.2668, 0.6211, 0.5291]
    ])

    dcdx_simple = wp.zeros((3, 4), dtype = float)
    x = wp.from_numpy(arr, dtype = wp.vec3, shape = (4))
    dcdx_delta = wp.zeros((3, 4), dtype = wp.mat33)
    ret = wp.zeros((4, 4), dtype = wp.mat33)
    d2Psi = wp.zeros((3, 3), dtype = wp.mat33)
    wp.launch(test_vf, 1, inputs = [x, dcdx_delta, ret, d2Psi, dcdx_simple])
    # wp.launch(test_ee, 1, inputs = [x, dcdx_delta, ret, d2Psi, dcdx_simple])
    print(ret.numpy())
    ret.zero_()
    wp.launch(d2Psidx2, (4, 4), inputs = [ret, d2Psi, dcdx_simple, dcdx_delta])
    print(ret.numpy())

    q = wp.zeros((9,3), dtype = wp.vec3)
    lam = wp.zeros((1, 9), dtype = float)

    wp.launch(verify_eig_sys_vf, 1, inputs = [x, q, lam])
    Q = q.numpy().reshape(9, 9).T
    Lambda = np.diag(lam.numpy().reshape(9))
    # print(Q)
    # print(Lambda)
    # print(Q @ Lambda)
    # assemble H from numpy
    _dc_s = dcdx_simple.numpy().reshape(3, 4)
    _d2Psi = d2Psi.numpy()
    d2Psidc = np.zeros((9, 9))
    for ii in range(3):
        for jj in range(3):
            d2Psidc[ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3] = _d2Psi[ii, jj]
    dc_s = np.kron(_dc_s, np.eye(3))
    dc_d = np.zeros((9, 12))
    _dc_d = dcdx_delta.numpy()
    for ii in range(3):
        for jj in range(4):
            dc_d[ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3] = _dc_d[ii, jj]
    
    H = dc_s.T @ d2Psidc @ dc_s - dc_d.T @ d2Psidc @ dc_d

    # print(d2Psidc @ Q - Q @ Lambda)
    d2Psidc1 = project_psd(d2Psidc, Q, Lambda)
    d2Psidc2 = project_psd(-d2Psidc, Q, -Lambda)
    QTQ = Q.T @ Q
    diag_inv = np.array([(0.0 if i >= 5 else (1.0 / QTQ[i, i])) for i in range(9)])

    diaginv = np.diag(diag_inv)
    Q_inv = diaginv @ Q.T

    # print(Q_inv @ Q)
    print(Q_inv @ d2Psidc @ Q)
    print(Q_inv @ d2Psidc1 @ Q)
    print(Q_inv @ d2Psidc2 @ Q)
    # print(np.diag(Q.T @ d2Psidc1 @ Q))
    # print(Q.T @ Q)
