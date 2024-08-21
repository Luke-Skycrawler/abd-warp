from const_params import *
from typing import Any
from psd.ee import beta_gamma_ee, C_ee, dceedx_s
from psd.hl import signed_distance, eig_Hl_tid, gl
from psd.vf import beta_gamma_pt, C_vf, dcvfdx_s
from affine_body import AffineBody, fetch_ee, fetch_pt, vg_distance, fetch_vertex, fetch_pt_xk, fetch_ee_xk, fetch_pt_x0, fetch_ee_x0


class IPCContactEnergy:
    def __init__(self) -> None:
        pass

    def energy(self, inputs):
        ee_list = inputs[0]
        pt_list = inputs[1]
        vg_list = inputs[2]
        E = wp.zeros((1, ), dtype =float)
        inputs.append(E)
        # wp.launch(ipc_energy_ee, dim = ee_list.shape, inputs = inputs)
        # wp.launch(ipc_energy_pt, dim = pt_list.shape, inputs = inputs)
        wp.launch(ipc_energy_vg, dim = vg_list.shape, inputs = inputs)
        return E.numpy()[0]

    def gradient(self, inputs):
        pass

    def hessian(self, inputs):
        pass

    def gh(self, inputs):

        pt_list, ee_list, vg_list, nij, bodies, g, blocks = inputs
        dim = vg_list.shape[0]

        ipc_term_pt(nij, pt_list, bodies, g, blocks)
        ipc_term_ee(nij, ee_list, bodies, g, blocks)
        wp.launch(ipc_term_vg, dim = (dim, ), inputs = [vg_list, bodies, g, blocks])
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
    ea0, ea1, eb0, eb1 = fetch_ee_xk(ijee, bodies)
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
    p = bi.xk[pid]

    d = vg_distance(p)
    wp.atomic_add(E, 0, barrier(d * d))

@wp.kernel
def ipc_energy_pt(ee_list: wp.array(dtype = vec5i), pt_list: wp.array(dtype = vec5i), vg_list: wp.array(dtype = wp.vec2i), bodies: wp.array(dtype = Any), E: wp.array(dtype = float)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt_xk(pt_list[i], bodies)

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
            

# @wp.kernel
# def ipc_term_pt(nij: int, pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), g: wp.array(dtype = wp.vec3), blocks: wp.array(dtype = wp.mat33)):
#     i = wp.tid()

#     n_bodies = bodies.shape[0]

#     idx = pt_list[i][2]

#     offset_upper = 16 * (n_bodies + idx)
#     offset_lower = 16 * (n_bodies + idx + nij)

#     p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)

def extract_g(Q, dcdx, Jp, Jt):
    gl = np.zeros(9)
    for i in range(3):
        gl[3 * i: 3 * i + 3] = Q[4 * 3 + i]
    g12 = np.kron(dcdx.T, np.eye(3)) @ gl
    Jp = Jp.reshape(1, 4)
    gp = np.kron(Jp.T, np.eye(3)) @ g12[:3]
    gt = np.kron(Jt.T, np.eye(3)) @ g12[3:]
    g = np.zeros(24)
    g[:12] = gp
    g[12:] = gt
    return g


def ipc_term_pt(nij, pt_list, bodies, grad, blocks):
    npt = pt_list.shape[0]
    n_bodies = bodies.shape[0]

    Q = wp.zeros((npt, 5 * 3), dtype = wp.vec3)
    Lam = wp.zeros((npt, 5), dtype = float)
    dcdx = wp.zeros((npt, ), dtype = mat34)
    Jt = wp.zeros((npt, ), dtype = mat34)
    Jp = wp.zeros((npt, ), dtype = wp.vec4) 

    wp.launch(_Q_lambda_pt, dim = (npt, ), inputs = [pt_list, bodies, Q, Lam])
    wp.launch(_dcdx, dim = (npt, ), inputs = [pt_list, bodies, dcdx])
    wp.launch(extract_JpJt, dim = (npt,), inputs=[pt_list, bodies, Jp, Jt])

    Qnp = Q.numpy()
    Jpnp = Jp.numpy()
    Jtnp = Jt.numpy()
    Lamnp = Lam.numpy()
    dcdxnp = dcdx.numpy()
    
    gnp = np.zeros((npt, 24))
    Hinp = np.zeros((npt, 12, 12))
    Hjnp = np.zeros((npt, 12, 12))
    Hijnp = np.zeros((npt, 12, 12))

    for i in range(npt):
        g = extract_g(Qnp[i], dcdxnp[i], Jpnp[i], Jtnp[i])

        qq = extract_Q(Qnp[i])
        Hl_pos = QLQinv(qq, Lamnp[i])
        H12 = dcTHldc(dcdxnp[i], Hl_pos)
        # H12 tested
        Hi, Hj, Hij = JTH12J(H12, Jpnp[i], Jtnp[i])
        gnp[i] = g
        Hinp[i] = Hi
        Hjnp[i] = Hj
        Hijnp[i] = Hij

    put_grad(grad, gnp, pt_list)
    put_hess(blocks, Hinp, Hjnp, Hijnp, pt_list, nij, n_bodies)
    return gnp, Hinp, Hjnp, Hijnp

def put_grad(g, gnp, pt_list):
    npt = pt_list.shape[0]
    gwp = wp.from_numpy(gnp.reshape(npt, 8, 3), dtype = wp.vec3, shape = (npt, 8))
    wp.launch(_put_grad, dim = (npt, ), inputs = [gwp, pt_list, g])

def put_hess(blocks, Hinp, Hjnp, Hijnp, pt_list, nij, n_bodies):
    npt = pt_list.shape[0]
    Hjwp = to_wp(Hjnp)
    Hijwp = to_wp(Hijnp)
    Hiwp = to_wp(Hinp)
    wp.launch(_put_hess, dim = (npt, ), inputs = [blocks, Hiwp, Hjwp, Hijwp, pt_list, nij, n_bodies])

def to_wp(H):
    npt = H.shape[0]
    Hn = np.zeros((npt, 4, 4, 3, 3))
    for i in range(npt):
        for ii in range(3):
            for jj in range(3):
                Hn[i, ii, jj] = H[i, ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3]

    Hwp = wp.from_numpy(Hn, dtype = wp.mat33, shape = (npt, 4, 4))
    return Hwp

@wp.kernel
def _put_hess(blocks: wp.array(dtype = wp.mat33), Hi: wp.array3d(dtype = wp.mat33), Hj: wp.array3d(dtype = wp.mat33), Hij: wp.array3d(dtype = wp.mat33), pt_list: wp.array(dtype = vec5i), nij: int, n_bodies: int):
    i = wp.tid()
    I = pt_list[i][0]
    J = pt_list[i][1]
    idx = pt_list[i][2]
    
    for ii in range(4):
        for jj in range(4):
            blocks[16 * I + ii + jj * 4] = Hi[i, ii, jj]
            blocks[16 * J + ii + jj * 4] = Hj[i, ii, jj]

            blocks[16 * (n_bodies + idx) + ii + jj * 4] = Hij[i, ii, jj]
            blocks[16 * (n_bodies + idx + nij) + jj + ii * 4] = wp.transpose(Hij[i, ii, jj])

@wp.kernel
def _put_grad(gnp: wp.array2d(dtype = wp.vec3), pt_list: wp.array(dtype = vec5i), g: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    I = pt_list[i][0]
    J = pt_list[i][1]

    wp.atomic_add(g, 4 * I + 0, gnp[i, 0])
    wp.atomic_add(g, 4 * I + 1, gnp[i, 1])
    wp.atomic_add(g, 4 * I + 2, gnp[i, 2])
    wp.atomic_add(g, 4 * I + 3, gnp[i, 3])

    wp.atomic_add(g, 4 * J + 0, gnp[i, 4])
    wp.atomic_add(g, 4 * J + 1, gnp[i, 5])
    wp.atomic_add(g, 4 * J + 2, gnp[i, 6])
    wp.atomic_add(g, 4 * J + 3, gnp[i, 7])

def extract_Q(q):
    Q = q.reshape(5, 9).T
    return Q
    
def QLQinv(Q, lam):
    QTQ = Q.T @ Q
    diag_inv = np.array([(1.0 / QTQ[i, i]) for i in range(5)])
    diag_inv = np.diag(diag_inv)
    Q_inv = diag_inv @ Q.T
    Lam = np.diag(lam)
    return Q @ Lam @ Q_inv

def dcTHldc(_dcdx, Hl_pos):
    dcdx = np.kron(_dcdx, np.eye(3))
    return dcdx.T @ Hl_pos @ dcdx

def JTH12J(H12, Jp, Jt):
    Jp = np.kron(Jp.reshape(1, 4), np.eye(3))
    Jt = np.kron(Jt, np.eye(3))
    Hpp = H12[:3, :3]
    Hpt = H12[:3, 3:]
    Htt = H12[3:, 3:]
    Hi = Jp.T @ Hpp @ Jp
    Hj = Jt.T @ Htt @ Jt
    Hij = Jp.T @ Hpt @ Jt
    return Hi, Hj, Hij
    
@wp.kernel
def extract_JpJt(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), Jp: wp.array(dtype = wp.vec4), Jt: wp.array(dtype = mat34)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_pt_x0(pt_list[i], bodies)
    Jp[i] = wp.vec4(1.0, x0[0], x0[1], x0[2])
    Jt[i] = mat34(1.0, x1[0], x1[1], x1[2], 
               1.0, x2[0], x2[1], x2[2], 
               1.0, x3[0], x3[1], x3[2])
    

@wp.kernel
def _Q_lambda_pt(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype =Any), q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_pt(pt_list[i], bodies)
    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
    lam0, lam1, lam2, lam3 = eig_Hl_tid(e0p, e1p, e2p, q, i)
    l = signed_distance(e0p, e1p, e2p)

    lam0 = wp.max(lam0, 0.0)
    lam1 = wp.max(lam1, 0.0)
    lam2 = wp.max(lam2, 0.0)
    lam3 = wp.max(lam3, 0.0)

    lam[i, 0] = lam0
    lam[i, 1] = lam1
    lam[i, 2] = lam2
    lam[i, 3] = lam3
    lam[i, 4] = 2.0

    gl0, gl1, gl2 = gl(l, e2p)
    q[i, 4 * 3 + 0] = gl0
    q[i, 4 * 3 + 1] = gl1
    q[i, 4 * 3 + 2] = gl2

@wp.kernel
def _dcdx(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), ret: wp.array2d(dtype = mat34)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_pt(pt_list[i], bodies)
    dcdx_simple = dcvfdx_s(x0, x1, x2, x3)
    ret[i] = dcdx_simple
    
    
def ipc_term_ee(nij, ee_list, bodies, g, blocks):
    return




if __name__ == "__main__":

    wp.init()
    np.set_printoptions(precision=4, suppress=True)

    def test0():
            
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

    def test1(): 
        pt_list = wp.zeros((1,), dtype = vec5i)
        arr = np.array([
            [0.4360, 0.0259, 0.5497],
            [0.4353, 0.4204, 0.3303],
            [0.2046, 0.6193, 0.2997],
            [0.2668, 0.6211, 0.5291]
        ])

        _bodies = []
        a = AffineBody()
        a.x = wp.zeros((1, ), dtype = wp.vec3)
        a.x0 = wp.zeros((1, ), dtype = wp.vec3)
        a.xk = wp.zeros((1, ), dtype = wp.vec3)
        a.x_view = wp.zeros((1, ), dtype = wp.vec3)
        a.triangles = wp.zeros((1, 3), dtype = int)
        a.edges = wp.zeros((1, 2), dtype = int)
        ax = arr[0]
        a.x.assign(ax)
        a.x0.assign(ax)
        a.xk.assign(ax)
        a.x_view.assign(ax)
        _bodies.append(a)

        b = AffineBody()
        b.x = wp.zeros((3, ), dtype = wp.vec3)
        b.x0 = wp.zeros_like(b.x)
        b.xk = wp.zeros_like(b.x)
        b.x_view = wp.zeros_like(b.x)
        tindex = np.array([0, 1, 2]).reshape(1, 3)
        b.triangles = wp.from_numpy(tindex, dtype = int, shape = (1, 3))
        b.edges = wp.zeros((1, 2), dtype = int)
        b.triangles.assign(tindex)

        bx = arr[1:]
        b.x.assign(bx)
        b.x0.assign(bx)
        b.xk.assign(bx)
        b.x_view.assign(bx)
        _bodies.append(b)

        bodies = wp.array(_bodies, dtype = AffineBody)

        g = wp.zeros((8,), dtype = wp.vec3)
        H = wp.zeros((4 * 16), dtype = wp.mat33)
        ptnp = np.array([0, 1, 0, 0, 0])
        pt_list.assign(ptnp)

        ipc_term_pt(1, pt_list, bodies, g, H)

    test1()