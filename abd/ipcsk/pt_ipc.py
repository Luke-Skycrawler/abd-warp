from const_params import *
from ipcsk.barrier import *
from ipcsk.put import put_grad, put_hess
from psd.vf import beta_gamma_pt, C_vf, dcvfdx_s
from affine_body import AffineBody, fetch_ee, fetch_pt, vg_distance, fetch_vertex, fetch_pt_xk, fetch_ee_xk, fetch_pt_x0, fetch_ee_x0
from ccd import verify_root_pt, verify_root_ee
from typing import Any
from psd.hl import signed_distance, eig_Hl_tid, gl
import ipctk
import abdtk

ipctk_ref = False
@wp.kernel
def _mask_valid(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), valid: wp.array(dtype = wp.bool), d: wp.array(dtype = float)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)

    true_root = verify_root_pt(p, t0, t1, t2)
    valid[i] = true_root
    e0p, e1p, e2p = C_vf(p, t0, t1, t2)
    d0 = signed_distance(e0p, e1p, e2p)
    d[i] = d0 * d0

def ipc_term_pt(nij, pt_list, bodies, grad, blocks):
    npt = pt_list.shape[0]
    n_bodies = bodies.shape[0]
    highlight = np.array([False for _ in range(n_bodies)])

    if not pt:
        return highlight 

    Q = wp.zeros((npt, 5 * 3), dtype = wp.vec3)
    Lam = wp.zeros((npt, 5), dtype = float)
    dcdx = wp.zeros((npt, ), dtype = mat34)
    Jt = wp.zeros((npt, ), dtype = mat34)
    Jp = wp.zeros((npt, ), dtype = wp.vec4) 
    valid = wp.zeros((npt, ), dtype = wp.bool)
    d2 = wp.zeros((npt,), dtype = float)
    x = wp.zeros((npt, 4), dtype = wp.vec3)

    wp.launch(_mask_valid, dim = (npt, ), inputs = [pt_list, bodies, valid, d2])
    wp.launch(_Q_lambda_pt, dim = (npt, ), inputs = [pt_list, bodies, Q, Lam])
    wp.launch(_dcdx, dim = (npt, ), inputs = [pt_list, bodies, dcdx])
    wp.launch(extract_JpJt, dim = (npt,), inputs=[pt_list, bodies, Jp, Jt])
    wp.launch(get_vertices, dim = (npt, ), inputs = [pt_list, bodies, x])

    Qnp = Q.numpy()
    Jpnp = Jp.numpy()
    Jtnp = Jt.numpy()
    Lamnp = Lam.numpy()
    dcdxnp = dcdx.numpy()
    validnp = valid.numpy()
    d2np = d2.numpy()
    xnp = x.numpy()
    
    gnp = np.zeros((npt, 24))
    Hinp = np.zeros((npt, 12, 12))
    Hjnp = np.zeros((npt, 12, 12))
    Hijnp = np.zeros((npt, 12, 12))
    ptnp = pt_list.numpy()
    ijnp = np.delete(ptnp, 2, 1)
    d2min = np.min(d2np) if len(d2np) else 1.0
    # if npt > 0 and d2min < d2hat:
    #     # print(f"colision detected, list = {ptnp}")
    #     print(f"colision detected, d2 min = {d2min}")
    #     # print(f"d2 < dhat, valid: {validnp}")
    #     # print(f"d2 = {d2np}")

    # hess = np.zeros((24, 24))
    # gradient = np.zeros(24)

    # for i in range(npt):
    #     p, t0, t1, t2 = xnp[i]
    #     d2_ipc = ipctk.point_triangle_distance(p, t0, t1, t2)
    #     if d2_ipc < d2hat:
    #         I = ptnp[i, 0]
    #         J = ptnp[i, 1]
    #         highlight[J] = True
    #         B_ = barrier_derivative_np(d2_ipc)
    #         B__ = barrier_derivative2_np(d2_ipc)

    #         pt_grad = ipctk.point_triangle_distance_gradient(p, t0, t1, t2)

    #         pt_hess = ipctk.point_triangle_distance_hessian(p, t0, t1, t2)
    #         _Q, _Lam = eigh(pt_hess)
    #         _Lam = np.max(_Lam, 0)
    #         pt_hess = _Q @ np.diag(_Lam) @ _Q.T
            
    #         Hipc = pt_hess * B_ + np.outer(pt_grad, pt_grad) * B__ 
    #         Hi, Hj, Hij = JTH12J(Hipc, Jpnp[i], Jtnp[i])
    #         pt_grad *= B_            

    #         g = JTg(Jpnp[i], Jtnp[i], pt_grad)
            
    #         gnp[i] = g
    #         Hinp[i] = Hi
    #         Hjnp[i] = Hj
    #         Hijnp[i] = Hij
            
    #         # hess[I * 12: I * 12 + 12, I * 12: I * 12 + 12] += Hi
    #         # hess[J * 12: J * 12 + 12, J * 12: J * 12 + 12] += Hj
    #         # if I < J:
    #         #     hess[I * 12: I * 12 + 12, J * 12: J * 12 + 12] += Hij
    #         #     hess[J * 12: J * 12 + 12, I * 12: I * 12 + 12] += Hij.T
    #         # else:
    #         #     hess[J * 12: J * 12 + 12, I * 12: I * 12 + 12] += Hij
    #         #     hess[I * 12: I * 12 + 12, J * 12: J * 12 + 12] += Hij.T
    #         # gradient[I * 12: I * 12 + 12] += g[:12]
    #         # gradient[J * 12: J * 12 + 12] += g[12:]
            
    for i in range(npt):
        # if validnp[i]:
        # if True:
        if d2np[i] < d2hat and validnp[i]:
        # if d2np[i] < d2hat:
            with wp.ScopedTimer(f"pt contact {i}"):
                I = ptnp[i, 1]
                highlight[I] = True

                B_ = barrier_derivative_np(d2np[i])
                B__ = barrier_derivative2_np(d2np[i])
                print(f"pt contact {i}, d^2 = {d2np[i]}")
                pt_grad, g = extract_g(Qnp[i], dcdxnp[i], Jpnp[i], Jtnp[i])
                
                qq = extract_Q(Qnp[i])
                Hl_pos = QLQinv(qq, Lamnp[i])


                Hpt = dcTHldc(dcdxnp[i], Hl_pos)

                if ipctk_ref:
                    p = xnp[i, 0]
                    t0 = xnp[i, 1]
                    t1 = xnp[i, 2]
                    t2 = xnp[i, 3]

                    gpt_ipc = ipctk.point_plane_distance_gradient(p, t0, t1, t2)
                    Hpt_ipc = ipctk.point_plane_distance_hessian(p, t0, t1, t2)
                    d2_ipc = ipctk.point_plane_distance(p, t0, t1, t2)
                    # gpt_ipc = ipctk.point_triangle_distance_gradient(p, t0, t1, t2)
                    # Hpt_ipc = ipctk.point_triangle_distance_hessian(p, t0, t1, t2)
                    # d2_ipc = ipctk.point_triangle_distance(p, t0, t1, t2)
                    
                    # print(f"valid = {validnp[i]}")
                    # print(f"d2 = {d2np[i]}, d2_ipc = {d2_ipc}, diff = {d2np[i] - d2_ipc}")
                    # print(f"gpt = {pt_grad}, \nref = {gpt_ipc}, \n diff = {pt_grad - gpt_ipc}")
                    # print(f"Hpt = {Hpt}, \nref = {Hpt_ipc}, \n diff = {Hpt - Hpt_ipc}")
                

                
                    B_ = barrier_derivative_np(d2_ipc)
                    B__ = barrier_derivative2_np(d2_ipc)
                    Hpt = Hpt_ipc
                    pt_grad = gpt_ipc

                # if d2np[i] < d2hat:
                #     print(f"pt contact {i}, d^2 = {d2np[i]}")
                #     print(f"B. = {B_}")
                #     print(f"B.. = {B__}")
                #     print(f"g = {g}")
                # else: 
                #     print(f"pt contact {i}, d^2 = {d2np[i]}")



                Hipc = Hpt * B_ + np.outer(pt_grad, pt_grad) * B__ 
                pt_type = abdtk.PointTriangleDistanceType.P_T
                Hipc_ref, g_ref = abdtk.ipc_hess_pt_12x12(xnp[i], ijnp[i], pt_type, d2np[i])

                g = JTg(Jpnp[i], Jtnp[i], pt_grad)
                pt_grad *= B_
                g *= B_

                print(
                    # "hessian (psd projected) = ", Hipc,
                    # "gradient = ", pt_grad, 
                    # "ref = ", Hipc_ref, g_ref, 
                    "diff =", np.max(np.abs(Hipc - Hipc_ref)), np.max(np.abs(pt_grad - g_ref)))
                # H12 tested
                Hi, Hj, Hij = JTH12J(Hipc, Jpnp[i], Jtnp[i])

                gnp[i] = g
                Hinp[i] = Hi
                Hjnp[i] = Hj
                Hijnp[i] = Hij

    put_grad(grad, gnp, pt_list)
    put_hess(blocks, Hinp, Hjnp, Hijnp, pt_list, nij, n_bodies)
    
    return highlight



def extract_g(Q, dcdx, Jp, Jt):
    gl = np.zeros(9)
    for i in range(3):
        gl[3 * i: 3 * i + 3] = Q[4 * 3 + i]
    g12 = np.kron(dcdx.T, np.eye(3)) @ gl
    g = JTg(Jp, Jt, g12)
    return g12, g

def JTg(Jp, Jt, g12):
    Jp = Jp.reshape(1, 4)
    gp = np.kron(Jp.T, np.eye(3)) @ g12[:3]
    gt = np.kron(Jt.T, np.eye(3)) @ g12[3:]
    g = np.zeros(24)
    g[:12] = gp
    g[12:] = gt
    return g


@wp.kernel
def get_vertices(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), x: wp.array2d(dtype = wp.vec3)):
    i = wp.tid()
    p, t0, t1, t2 = fetch_pt(pt_list[i], bodies)
    x[i, 0] = p
    x[i, 1] = t0
    x[i, 2] = t1
    x[i, 3] = t2

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
    q[i, 4 * 3 + 0] = gl0 * 2.0 * l
    q[i, 4 * 3 + 1] = gl1 * 2.0 * l
    q[i, 4 * 3 + 2] = gl2 * 2.0 * l

@wp.kernel
def _dcdx(pt_list: wp.array(dtype = vec5i), bodies: wp.array(dtype = Any), ret: wp.array2d(dtype = mat34)):
    i = wp.tid()
    x0, x1, x2, x3 = fetch_pt(pt_list[i], bodies)
    dcdx_simple = dcvfdx_s(x0, x1, x2, x3)
    ret[i] = dcdx_simple
