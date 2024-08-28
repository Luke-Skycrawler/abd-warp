from const_params import *
# from ipcsk.barrier import *

from typing import Any
from psd.ee import beta_gamma_ee, C_ee, dceedx_s
from psd.hl import signed_distance, eig_Hl_tid, gl
from psd.vf import beta_gamma_pt, C_vf, dcvfdx_s
from affine_body import AffineBody, fetch_ee, fetch_pt, vg_distance, fetch_vertex, fetch_pt_xk, fetch_ee_xk, fetch_pt_x0, fetch_ee_x0
from ccd import verify_root_pt, verify_root_ee
from scipy.linalg import eigh
from ipcsk.energy import ipc_energy_ee, ipc_energy_pt, ipc_energy_vg
from ipcsk.vg_ipc import ipc_term_vg
from ipcsk.pt_ipc import ipc_term_pt
from ipcsk.ee_ipc import ipc_term_ee
class IPCContactEnergy:
    def __init__(self) -> None:
        pass

    def energy(self, inputs):
        ee_list = inputs[0]
        pt_list = inputs[1]
        vg_list = inputs[2]
        E = wp.zeros((1, ), dtype =float)
        inputs.append(E)
        Enp = 0.0
        if ee:
            wp.launch(ipc_energy_ee, dim = ee_list.shape, inputs = inputs)
        if pt:
            wp.launch(ipc_energy_pt, dim = pt_list.shape, inputs = inputs)
            # bodies = inputs[-2]
            # x = wp.zeros((pt_list.shape[0], 4), dtype = wp.vec3)
            # npt = pt_list.shape[0]
            # wp.launch(get_vertices, dim = pt_list.shape, inputs = [pt_list, bodies, x])
            # xnp = x.numpy()
            # Enp = 0.0
            # for i in range(npt):
            #     p = xnp[i, 0]
            #     t0 = xnp[i, 1]
            #     t1 = xnp[i, 2]
            #     t2 = xnp[i, 3]
            #     d2 = ipctk.point_triangle_distance(p, t0, t1, t2)
            #     Enp += barrier_np(d2)

        if vg:
            wp.launch(ipc_energy_vg, dim = vg_list.shape, inputs = inputs)
        return E.numpy()[0] + Enp

    def gradient(self, inputs):
        pass

    def hessian(self, inputs):
        pass

    def gh(self, inputs):

        pt_list, ee_list, vg_list, nij, bodies, g, blocks = inputs
        dim = vg_list.shape[0]

        highlight = ipc_term_pt(nij, pt_list, bodies, g, blocks)
        highlight = ipc_term_ee(nij, ee_list, bodies, g, blocks)
        wp.launch(ipc_term_vg, dim = (dim, ), inputs = [vg_list, bodies, g, blocks])
        return highlight

    
    




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

    def test_ee(): 
        ee_list = wp.zeros((1,), dtype = vec5i)
        arr = np.array([
            [0.4360, 0.0259, 0.5497],
            [0.4353, 0.4204, 0.3303],
            [0.2046, 0.6193, 0.2997],
            [0.2668, 0.6211, 0.5291]
        ])

        _bodies = []
        a = AffineBody()
        a.x = wp.zeros((2, ), dtype = wp.vec3)
        a.x0 = wp.zeros((2, ), dtype = wp.vec3)
        a.xk = wp.zeros((2, ), dtype = wp.vec3)
        a.x_view = wp.zeros((2, ), dtype = wp.vec3)
        a.triangles = wp.zeros((1, 3), dtype = int)
        a.edges = wp.zeros((1, 2), dtype = int)
        ax = arr[: 2]
        a.x.assign(ax)
        a.x0.assign(ax)
        a.xk.assign(ax)
        a.x_view.assign(ax)
        a.edges.assign(np.array([[0, 1]]))
        _bodies.append(a)

        b = AffineBody()
        b.x = wp.zeros((2, ), dtype = wp.vec3)
        b.x0 = wp.zeros_like(b.x)
        b.xk = wp.zeros_like(b.x)
        b.x_view = wp.zeros_like(b.x)
        tindex = np.array([0, 1, 2]).reshape(1, 3)
        b.triangles = wp.from_numpy(tindex, dtype = int, shape = (1, 3))
        b.edges = wp.zeros((1, 2), dtype = int)
        b.triangles.assign(tindex)
        b.edges.assign(np.array([[0, 1]]))

        bx = arr[2:]
        b.x.assign(bx)
        b.x0.assign(bx)
        b.xk.assign(bx)
        b.x_view.assign(bx)
        _bodies.append(b)

        bodies = wp.array(_bodies, dtype = AffineBody)

        g = wp.zeros((8,), dtype = wp.vec3)
        H = wp.zeros((4 * 16), dtype = wp.mat33)
        eenp = np.array([0, 1, 0, 0, 0])
        ee_list.assign(eenp)

        ipc_term_ee(1, ee_list, bodies, g, H)

    test_ee()