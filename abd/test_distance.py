from const_params import *
from psd.vf import beta_gamma_pt, C_vf
from psd.ee import beta_gamma_ee, C_ee
from ccd import verify_root_pt, verify_root_ee
import warp as wp
import numpy as np
from ipc import ipc_term_pt, _mask_valid
from culling import cull, BvhBuilder
from affine_body import AffineBody
import igl
seed = 114514
@wp.kernel
def init_positive(x: wp.array2d(dtype = wp.vec3)):
    i = wp.tid()
    rng = wp.rand_init(seed)
    e0 = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))
    e1 = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))

    beta = wp.randf(rng)
    gamma = wp.randf(rng) * (1.0 - beta)
    
    n = wp.cross(e0, e1)
    nd = wp.randf(rng)

    x[i, 1] = wp.vec3(0.0, 0.0, 0.0)
    x[i, 2] = e0 + x[i, 1]
    x[i, 3] = e1 + x[i, 1]
    x[i, 0] = n * nd + x[i, 1] + beta * e0 + gamma * e1

@wp.kernel
def init_negative(x: wp.array2d(dtype = wp.vec3)):
    i = wp.tid()
    rng = wp.rand_init(seed)
    e0 = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))
    e1 = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))

    _beta = wp.randf(rng)
    _gamma = wp.randf(rng) * (1.0 - _beta)

    beta = 1.0 - _gamma
    gamma = 1.0 - _beta

    n = wp.cross(e0, e1)
    nd = wp.randf(rng)

    x[i, 1] = wp.vec3(0.0, 0.0, 0.0)
    x[i, 2] = e0 + x[i, 1]
    x[i, 3] = e1 + x[i, 1]
    x[i, 0] = n * nd + x[i, 1] + beta * e0 + gamma * e1


@wp.kernel
def test_artificial_dataset(x: wp.array2d(dtype = wp.vec3), ret: wp.array(dtype = wp.bool)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]

    valid = verify_root_pt(x0, x1, x2, x3)
    ret[i] = valid

def test_triangle_inside():

    n_tests = 1000
    expect = False

    x = wp.zeros((n_tests, 4), dtype = wp.vec3) 
    ret = wp.zeros((n_tests,), dtype = wp.bool)

    ker = init_positive if expect else init_negative
    wp.launch(ker, n_tests, inputs = [x])
    wp.launch(test_artificial_dataset, n_tests, inputs = [x, ret])

    a = ret.numpy()
    assert ((a == expect).all())


def test_continous():
    # import polyscope as ps
    from simulator.mesh import RawMeshFromFile
    import fc_viewer as fcd
    # cube = RawMeshFromFile("assets/box.bgeo")
    # cube = RawMeshFromFile("assets/triangle.obj")

    # V0 = cube.V.astype(dtype = np.float32)
    # F = cube.F.astype(dtype = np.int32)
    V0 = np.eye(3)
    F = np.arange(3).reshape(1, 3)

    print(f"F.shape = {F.shape}, V.shape = {V0.shape}")
    # E = igl.edges(F)
    E = np.array([0, 1], dtype = np.int32)
    viewer = fcd.fast_cd_viewer()
    vis_cd = True
    transform = "translate"
    T0 = None

    ij_list = wp.from_numpy(np.array([[0, 1]]), dtype = wp.vec2i, shape = (1,))
    bb = BvhBuilder()

    # Vwp = wp.zeros(V0.shape[0], dtype = wp.vec3)
    # Vwp.assign(V0)
    Fwp = wp.zeros(F.shape, dtype = int)
    Fwp.assign(F)

    pt_list = wp.zeros((1,), dtype = vec5i)
    ptnp = np.array([0, 1, 0, 0, 0])
    pt_list.assign(ptnp)

    _bodies = []
    a = AffineBody()
    a.x = wp.zeros((V0.shape[0], ), dtype = wp.vec3)
    a.x0 = wp.zeros((V0.shape[0], ), dtype = wp.vec3)
    a.xk = wp.zeros((V0.shape[0], ), dtype = wp.vec3)
    a.x_view = wp.zeros((V0.shape[0], ), dtype = wp.vec3)
    a.triangles = wp.zeros(F.shape, dtype = int)
    a.edges = wp.zeros(E.shape, dtype = int)
    
    ax = V0
    a.x.assign(ax)
    a.x0.assign(ax)
    a.xk.assign(ax)
    a.x_view.assign(ax)
    a.triangles.assign(F)
    a.edges.assign(E)
    _bodies.append(a)

    b = AffineBody()
    b.x = wp.zeros((V0.shape[0], ), dtype = wp.vec3)
    b.x0 = wp.zeros_like(b.x)
    b.xk = wp.zeros_like(b.x)
    b.x_view = wp.zeros_like(b.x)
    tindex = F
    b.triangles = wp.from_numpy(tindex, dtype = int, shape = tindex.shape)
    b.edges = wp.zeros(E.shape, dtype = int)
    b.triangles.assign(tindex)

    bx = V0
    b.x.assign(bx)
    b.x0.assign(bx)
    b.xk.assign(bx)
    b.x_view.assign(bx)
    b.edges.assign(E)
    _bodies.append(b)

    bodies = wp.array(_bodies, dtype = AffineBody)

    g = wp.zeros((8,), dtype = wp.vec3)
    H = wp.zeros((4 * 16), dtype = wp.mat33)

    # ipc_term_pt(1, pt_list, bodies, g, H)
    print("viewer launched")

    def guizmo_callback(A):
        nonlocal T0
        T0 = A

    init_guizmo = True
    def callback_key_pressed(key, modifier):
        nonlocal vis_cd, transform, viewer
        if (key == ord('g') or key == ord('G')):
            if (init_guizmo):
                if (transform == "translate"):
                    transform = "rotate"
                elif (transform == "rotate"):
                    transform = "scale"
                elif (transform == "scale"):
                    transform = "translate"

                viewer.change_guizmo_op(transform)
            else:
                print("Guizmo not initialized, pass init_guizmo=True to the viewer constructor")

    def pre_draw_callback():
        A = T0
        V = V0 @ A[:3, :3].T + A[: 3, 3].reshape(1, 3)

        a = _bodies[0]
        b = _bodies[1]

        a.x.assign(V)
        b.x.assign(V0)

        bp0 = bb.build_point_bvh(a.x, dhat)
        bp1 = bb.build_point_bvh(b.x, dhat)
        

        bt1 = bb.bulid_triangle_bvh(b.x, Fwp, dhat)
        bt0 = bb.bulid_triangle_bvh(a.x, Fwp, dhat)
        pt_list = cull(ij_list, [bp0, bp1], [bt0, bt1])

        npt = pt_list.shape[0]

        valid = wp.zeros((npt, ), dtype = wp.bool)
        d2 = wp.zeros((npt, ), dtype = float)
        wp.launch(_mask_valid, dim = (npt, ), inputs = [pt_list, bodies, valid, d2])

        ptnp = pt_list.numpy()
        vnp = valid.numpy()
        dnp = d2.numpy()

        c = [0, 0]
        highlight_color = np.array([1.0, 1.0, 0.0]) * 0.8
        normal_color = np.ones(3) * 0.4
        for d2, _v, pt in zip(dnp, vnp, ptnp):
            # if _v:
            # if d2 < d2hat:
            if _v and d2 < d2hat:
                # print(f"d2 = {d2}, d2hat = {d2hat}")

                I = pt[0]                
                c[I] = 1

        for i in range(2):                
            viewer.set_color(highlight_color if c[i] == 1 else normal_color, i)

        viewer.set_mesh(V, F, 0)
        viewer.set_mesh(V0, F, 1)   

    if init_guizmo:
        if T0 is None:
            T0 = np.identity(4).astype( dtype=np.float32, order="F")
        viewer.init_guizmo(True, T0, guizmo_callback, transform)

    viewer.set_pre_draw_callback(pre_draw_callback)
    viewer.set_key_callback(callback_key_pressed)
    viewer.add_mesh(V0.astype(np.float64), F)
    for i in range(2):
        viewer.set_face_based(True, i)
        viewer.invert_normals(True, i)
    viewer.set_color(np.array([0.5, 0.5, 0.5]), 1)

    viewer.launch()    

    
if __name__ == "__main__":
    print("test start")
    test_continous()
    
    

    