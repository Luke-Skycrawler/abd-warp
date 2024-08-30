from const_params import *
from psd.hl import signed_distance, eig_Hl_tid, gl

from psd.ee import C_ee, dceedx_s
from affine_body import fetch_ee, fetch_ee_x0
from ccd import verify_root_ee
from typing import Any
from ipcsk.barrier import *
from ipcsk.put import put_grad, put_hess
import ipctk
ipctk_ref = False

@wp.kernel
def _mask_valid(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), valid: wp.array(dtype = wp.bool), d: wp.array(dtype = float)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_ee(ee_list[i], bodies)

    true_root = verify_root_ee(p, t0, t1, t2)
    valid[i] = true_root
    e0p, e1p, e2p = C_ee(p, t0, t1, t2)
    # d0 = signed_distance(e0p, e1p, e2p)
    # d[i] = d0 * d0
    d[i] = wp.length_sq(e2p)

@wp.kernel
def _Q_lambda_ee(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype =Any), q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_ee(pt_list[i], bodies)
    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    lam0, lam1, lam2, lam3 = eig_Hl_tid(e0p, e1p, e2p, q, i)
    l = signed_distance(e0p, e1p, e2p)

    # lam0 = wp.max(lam0, 0.0)
    # lam1 = wp.max(lam1, 0.0)
    # lam2 = wp.max(lam2, 0.0)
    # lam3 = wp.max(lam3, 0.0)

    lam[i, 0] = lam0
    lam[i, 1] = lam1
    lam[i, 2] = lam2
    lam[i, 3] = lam3
    lam[i, 4] = 2.0

    gl0, gl1, gl2 = gl(l, e2p)
    q[i, 4 * 3 + 0] = gl0 * 2.0 * l
    q[i, 4 * 3 + 1] = gl1 * 2.0 * l
    q[i, 4 * 3 + 2] = gl2 * 2.0 * l


def ipc_term_ee(nij, ee_list, bodies, grad, blocks):
    nee = ee_list.shape[0]
    n_bodies = bodies.shape[0]
    highlight = np.array([False for _ in range(n_bodies)])

    if not ee:
        return highlight

    Q = wp.zeros((nee, 5 * 3), dtype = wp.vec3)
    Lam = wp.zeros((nee, 5), dtype = float)
    dcdx = wp.zeros((nee, ), dtype = mat34)
    Jei = wp.zeros((nee, ), dtype = mat24)
    Jej = wp.zeros((nee, ), dtype = mat24)
    valid = wp.zeros((nee, ), dtype = wp.bool)
    d2 = wp.zeros((nee,), dtype = float)    
    x = wp.zeros((nee, 4), dtype = wp.vec3)

    wp.launch(_mask_valid, dim = (nee, ), inputs = [ee_list, bodies, valid, d2])

    wp.launch(_Q_lambda_ee, dim = (nee, ), inputs = [ee_list, bodies, Q, Lam])
    
    wp.launch(_dcdx, dim = (nee, ), inputs = [ee_list, bodies, dcdx])

    wp.launch(extract_JeiJej, dim = (nee, ), inputs = [ee_list, bodies, Jei, Jej])
    
    wp.launch(get_vertices, dim = (nee, ), inputs = [ee_list, bodies, x])

    Qnp = Q.numpy()
    Jeinp = Jei.numpy()
    Jejnp = Jej.numpy()
    Lamnp = Lam.numpy()
    dcdxnp = dcdx.numpy()
    validnp = valid.numpy()
    d2np = d2.numpy()
    eenp = ee_list.numpy()
    xnp = x.numpy()

    gnp = np.zeros((nee, 24))

    Hinp = np.zeros((nee, 12, 12))  
    Hjnp = np.zeros((nee, 12, 12))
    Hijnp = np.zeros((nee, 12, 12))
    
    # print(f"d2 min= {np.min(d2np)}, nee = {nee}, d2hat = {d2hat}")
    for i in range(nee):
        if d2np[i] < d2hat and validnp[i]:
            with wp.ScopedTimer(f"ee contact {i}"):
                J = eenp[i, 1]
                highlight[J] = True
                print(f"ee contact {i}, d2 = {d2np[i]}")
                B_ = barrier_derivative_np(d2np[i])
                B__ = barrier_derivative2_np(d2np[i])
                ee_grad, g = extract_g(Qnp[i], dcdxnp[i], Jeinp[i], Jejnp[i])
                qq = extract_Q(Qnp[i])
                Hl = QLQinv(qq, Lamnp[i])
                Hee = dcTHldc(dcdxnp[i], Hl)

                if ipctk_ref:
                    ei0 = xnp[i, 0]
                    ei1 = xnp[i, 1]
                    ej0 = xnp[i, 2]
                    ej1 = xnp[i, 3]

                    gee_ipc = ipctk.line_line_distance_gradient(ei0, ei1, ej0, ej1)

                    Hee_ipc = ipctk.line_line_distance_hessian(ei0, ei1, ej0, ej1)
                    d2_ipc = ipctk.line_line_distance(ei0, ei1, ej0, ej1)
                    # print(f"valid = {validnp[i]}, d2 = {d2np[i]}, d2_ipc = {d2_ipc}")
                    # print(f"ee_grad = {ee_grad}\nref = {gee_ipc}\ndiff = {(ee_grad - gee_ipc)}")
                    # print(f"H = {Hpt}\nref = {Hee_ipc}, diff = {(Hpt - Hee_ipc)}")


                    B_ = barrier_derivative(d2_ipc)
                    B__ = barrier_derivative2(d2_ipc)
                    Hee = Hee_ipc
                    ee_grad = gee_ipc
                            
                            
                            
                Hipc = Hee * B_ + np.outer(ee_grad, ee_grad) * B__ 

                g = JTg(Jeinp[i], Jejnp[i], ee_grad)
                g *= B_
                # H12 tested
                Hi, Hj, Hij = JTH12J(Hipc, Jeinp[i], Jejnp[i])

                gnp[i] = g
                Hinp[i] = Hi
                Hjnp[i] = Hj
                Hijnp[i] = Hij

    put_grad(grad, gnp, ee_list)
    put_hess(blocks, Hinp, Hjnp, Hijnp, ee_list, nij, n_bodies)
    return highlight
  

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

def extract_Q(q):
    Q = q.reshape(5, 9).T
    return Q
 
def extract_g(Q, dcdx, Jei, Jej):
    gl = np.zeros(9)
    for i in range(3):
        gl[3 * i: 3 * i + 3] = Q[4 * 3 + i]
    g12 = np.kron(dcdx.T, np.eye(3)) @ gl
    g = JTg(Jei, Jej, g12)
    return g12, g

def JTg(Jei, Jej, g12):
    gei = np.kron(Jei.T, np.eye(3)) @ g12[:6]
    gej = np.kron(Jej.T, np.eye(3)) @ g12[6:]
    g = np.zeros(24)
    g[:12] = gei
    g[12:] = gej
    return g

def JTH12J(H12, Jei, Jej):
    Jei = np.kron(Jei, np.eye(3))
    Jej = np.kron(Jej, np.eye(3))
    Hpp = H12[:6, :6]
    Hpt = H12[:6, 6:]
    Htt = H12[6:, 6:]
    Hi = Jei.T @ Hpp @ Jei
    Hj = Jej.T @ Htt @ Jej
    Hij = Jei.T @ Hpt @ Jej
    return Hi, Hj, Hij



@wp.kernel
def get_vertices(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = wp.vec3)):
    i = wp.tid()
    ei0, ei1, ej0, ej1 = fetch_ee(ee_list[i], bodies)
    x[i, 0] = ei0
    x[i, 1] = ei1
    x[i, 2] = ej0
    x[i, 3] = ej1

@wp.kernel
def extract_JeiJej(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), Jei: wp.array(dtype = mat24), Jej: wp.array(dtype = mat24)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_ee_x0(ee_list[i], bodies)
    Jei[i] = mat24(
        1.0, x0[0], x0[1], x0[2],
        1.0, x1[0], x1[1], x1[2]
    )
    Jej[i] = mat24(
        1.0, x2[0], x2[1], x2[2], 
        1.0, x3[0], x3[1], x3[2]
    )

@wp.kernel
def _dcdx(ee_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), ret: wp.array2d(dtype = mat34)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_ee(ee_list[i], bodies)
    dcdx_simple = dceedx_s(x0, x1, x2, x3)
    ret[i] = dcdx_simple
