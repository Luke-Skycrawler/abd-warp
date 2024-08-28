from const_params import *
from simulator.base import BaseSimulator
from affine_body import AffineBody, AffineMesh, AffineBodyStates, affine_body_states_empty
from warp.utils import array_inner
from warp.optim.linear import bicgstab
from warp.sparse import bsr_set_from_triplets, bsr_zeros
from orthogonal_energy import InertialEnergy
from ipc import IPCContactEnergy
from culling import BvhBuilder, intersection_bodies, cull, cull_vg
from simulator.fenwick import list_with_meta, compress
from typing import List, Any
# temporary
from orthogonal_energy import _init, _set_triplets
from ccd import toi_vg, toi_ee, toi_pt

class AffineBodySimulator(BaseSimulator):

    n_tmp = 5
    collision_cap = 1 << 15

    def __init__(self, config_file = "config.json"):
        super().__init__(member_type = AffineMesh, config_file = config_file) 

        n_bodies = len(self.scene.kinetic_objects)
        self.n_bodies = n_bodies
        '''
        A, p, Adot, pdot flattened together
        while mesh information is stored in AffineBody struct

        # A[0], A[1], A[2] are the degrees of freedom
        # to project a vertice, use A.T @ x + p
        '''
        self.states = affine_body_states_empty(n_bodies)

        self.states.A0.assign(self.gather("A"))
        self.states.p0.assign(self.gather("p"))
        self.states.pdot.assign(self.gather("pdot"))
        self.states.Adot.assign(self.gather("Adot"))



        self.energy = wp.zeros(2, dtype = float)
        self.alpha = wp.zeros(1, dtype = float)
        
        self.tmp_ret = wp.zeros(self.n_tmp, dtype = float)



        self.g = wp.zeros(n_bodies * 4, dtype = wp.vec3)
        self.dq = wp.zeros_like(self.g)

        self.hess = bsr_zeros(4 * n_bodies, 4 * n_bodies, wp.mat33)
        self.rows = wp.zeros(16 * n_bodies, dtype = int)
        self.cols = wp.zeros(16 * n_bodies, dtype = int)
        self.blocks = wp.zeros(16 * n_bodies, dtype = wp.mat33)

        self.pt_body_index = wp.zeros((self.collision_cap, 2), dtype = int)
        self.pt_prim_index = wp.zeros((self.collision_cap, 2), dtype = int)

        self.ee_body_index = wp.zeros((self.collision_cap, 2), dtype = int)
        self.ee_prim_index = wp.zeros((self.collision_cap, 2), dtype = int)
        
        self.g_body_index = wp.zeros(self.collision_cap, dtype = int)
        self.g_prim_index = wp.zeros(self.collision_cap, dtype = int)

        self.inertia = InertialEnergy()
        self.ipc_contact = IPCContactEnergy()
        self.bb = BvhBuilder()

        # self.affine_bodies = wp.array([ko.warp_affine_body() for ko in self.scene.kinetic_objects], dtype = AffineBody)
        self.affine_bodies = [ko.warp_affine_body(i) for i, ko in enumerate(self.scene.kinetic_objects)]
        self.warp_affine_bodies = wp.array(self.affine_bodies, dtype = AffineBody)

        self.bvh_bodies: wp.Bvh = self.bb.build_body_bvh(self.affine_bodies, dhat * 0.5)
        self.bvh_body_traj: wp.Bvh = self.bb.build_body_bvh(self.affine_bodies, dhat * 0.5)

        self.bvh_triangles: List[wp.Bvh] = []
        self.bvh_edges: List[wp.Bvh] = []
        self.bvh_points: List[wp.Bvh] = []

        self.bvh_triangle_traj: List[wp.Bvh] = []
        self.bvh_edge_traj: List[wp.Bvh] = []
        self.bvh_point_traj: List[wp.Bvh] = []

        
        for b in self.affine_bodies:
            self.bvh_triangles.append(self.bb.bulid_triangle_bvh(b.x, b.triangles, 0.0))
            self.bvh_edges.append(self.bb.build_edge_bvh(b.x, b.edges, dhat * 0.5))
            self.bvh_points.append(self.bb.build_point_bvh(b.x, dhat))

            self.bvh_triangle_traj.append(self.bb.bulid_triangle_bvh(b.x, b.triangles, 0.0))
            self.bvh_edge_traj.append(self.bb.build_edge_bvh(b.x, b.edges, dhat * 0.5))
            self.bvh_point_traj.append(self.bb.build_point_bvh(b.x, dhat))

        self.highlight = np.array([False for _ in range(n_bodies)], dtype = bool)
    
    def reset(self): 
        self.states.A0.assign(self.gather("A"))
        self.states.p0.assign(self.gather("p"))
        self.states.pdot.assign(self.gather("pdot"))
        self.states.Adot.assign(self.gather("Adot"))
        self.dq.zero_()
        for _, ab in zip(self.scene.kinetic_objects, self.affine_bodies): 
            _.assign_to(ab)
        

    @classmethod
    def simulator_args(cls):
        args_base = super().simulator_args()
        
        args_new = {
            "dt": 0.01,
            "dhat": 1e-3,
            "max_iter": 100,
            "tol": 1e-4,
            "kappa": 1e7,
            "stiffness": 1e7
        }
        
        args_base.update(args_new)
        return args_base


    def q_gets_q0(self):
        wp.copy(self.states.A, self.states.A0)
        wp.copy(self.states.p, self.states.p0)
        wp.copy(self.states.Ak, self.states.A0)
        wp.copy(self.states.pk, self.states.p0)

    def proximity_set(self):
        pass

    def compute_energy(self, alpha = 0.0):
        self.update_qk(alpha)
        self.update_mesh_vertex("xk")

        ij_list, pt_list, ee_list, vg_list = self.collision_set()

        e_c = self.ipc_contact.energy([ee_list, pt_list, vg_list, self.warp_affine_bodies])
        e_i = self.inertia.energy(self.states)
        return e_c + e_i

    def dot(self, a, b):
        return array_inner(a, b)

    def line_search(self, alpha_cap, E0):

        qTg = self.dot(self.dq, self.g)

        alpha = alpha_cap
        if alpha < 1.0:
            pass
            # print(f"debugging line search, alpha cap = {alpha_cap}, dq = {self.dq.numpy()}")
        while True:
            E1 = self.compute_energy(alpha)
            wolfe = E1 < E0 + c1 * alpha * qTg
            if wolfe or alpha < alpha_min:
                break
            alpha *= 0.5
        return alpha, E1
    
    
    def ij_list(self, bvh_bodies: wp.Bvh):
        ij_list, ij_meta = list_with_meta(wp.vec2i, 256, self.n_bodies)
        wp.launch(intersection_bodies, self.n_bodies, inputs = [bvh_bodies.id, bvh_bodies.lowers, bvh_bodies.uppers, ij_meta, ij_list])

        ij_list = compress(ij_list, ij_meta)
        return ij_list

    def trajectory_intersection_set(self):
        self.bb.body_bvh_to_traj(self.affine_bodies, self.bvh_bodies, self.bvh_body_traj)
        ij_list = self.ij_list(self.bvh_body_traj)

        for i in range(self.n_bodies):
            b = self.affine_bodies[i]
            self.bb.triangle_bvh_to_traj(b.xk, b.triangles, self.bvh_triangles[i], self.bvh_triangle_traj[i])
            self.bb.edge_bvh_to_traj(b.xk, b.edges, self.bvh_edges[i], self.bvh_edge_traj[i])
            self.bb.point_bvh_to_traj(b.xk, self.bvh_points[i], self.bvh_point_traj[i])

        ee_list = cull(ij_list, self.bvh_edge_traj)
        # pt_list = cull(ij_list, self.bvh_triangle_traj, self.bvh_point_traj)
        pt_list = cull(ij_list, self.bvh_point_traj, self.bvh_triangle_traj)

        vg_list = cull_vg(self.bvh_body_traj.lowers, self.bvh_point_traj, self.warp_affine_bodies)
        return ij_list, pt_list, ee_list, vg_list

    def collision_set(self):
        '''
        NOTE: all collision detection runs on xk(Ak, pk)
        '''

        bvh_bodies = self.bvh_bodies
        self.bb.update_body_bvh(self.affine_bodies, dhat * 0.5, bvh_bodies)
        ij_list = self.ij_list(bvh_bodies)

        for b, t, e, p in zip(self.affine_bodies, self.bvh_triangles, self.bvh_edges, self.bvh_points):
            self.bb.update_triangle_bvh(b.xk, b.triangles, 0.0, t)
            self.bb.update_edge_bvh(b.xk, b.edges, dhat * 0.5, e)
            self.bb.update_point_bvh(b.xk, dhat, p)

        ee_list = cull(ij_list, self.bvh_edges)
        # pt_list = cull(ij_list, self.bvh_triangles, self.bvh_points)
        pt_list = cull(ij_list, self.bvh_points, self.bvh_triangles)

        vg_list = cull_vg(bvh_bodies.lowers, self.bvh_points, self.warp_affine_bodies)
        return ij_list, ee_list, pt_list, vg_list
    
    def V_gets_V(self, states):
        self.update_mesh_vertex("x")

    def step(self, frame = 1):
        states = self.states
        inertia = self.inertia
        g = self.g
        rows, cols, hess = self.rows, self.cols, self.hess
        dq = self.dq
        ipc_contact = self.ipc_contact
        
        self.q_gets_q0()
        E0 = self.compute_energy(alpha = 0.0)
        # self.blocks = wp.zeros(shape = ((self.n_bodies + ij_list.shape[0] * 2) * 16, ), dtype = wp.mat33)
        it = 0 
        while True:
            self.V_gets_V(states)
            self.update_mesh_vertex("xk")
            ij_list, ee_list, pt_list, vg_list = self.collision_set()
            nij = ij_list.shape[0]
            
            self.blocks.zero_()
            self.g.zero_()
            inertia.gradient(g, states)
            inertia.hessian(self.blocks, states)

            # ipc_contact.gradient([g, states, ij_list, ee_list, pt_list])
            # ipc_contact.hessian([self.blocks, states, ij_list, ee_list, pt_list])

            self.highlight[:] = ipc_contact.gh([pt_list, ee_list, vg_list, nij, self.warp_affine_bodies, g, self.blocks])

            rows.zero_()
            cols.zero_()
            wp.launch(_set_triplets, self.n_bodies, inputs = [self.n_bodies,  nij, ij_list, rows, cols])
            bsr_set_from_triplets(hess, rows, cols, self.blocks)
            bicgstab(hess, g, dq, 1e-4)
            if vg_list.shape[0] > 0:
                # print(f"debugging: dq = {dq.numpy()}")
                # print(f"debugging: g = {g.numpy()}") 
                pass
                # print(f"H = {hess}")

            # print(bsr.blocks.numpy())
            # print(hess.values.numpy())
            # print(g.numpy())
            # print(dq.numpy())
            alpha_cap = self.ccd(states, dq)
            alpha, E0 = self.line_search(alpha_cap, E0)

            
            self.update_q(alpha)
            
            it += 1
            if alpha < 1.0: 
                print(f"iteration {it}, cap, alpha = {alpha_cap}, {alpha}, energy = {E0}")

            gTg = self.dot(g, g)
            cond = gTg < tol
            # print(f"iteration {it}, |g| = {gTg}")
            if cond or it >= max_iter:
            # if cond:
                # fixme: temp
                break

        self.update_q0qdot()
        xmin = self.update_mesh_vertex("x_view")
        # print("a dot = ", states.Adot.numpy())
        # print(f"frame {frame} finished, xmin = {xmin}")

    def ccd(self, states, dq):
        self.update_qk(1.0)
        self.update_mesh_vertex("xk")
        self.update_mesh_vertex("x")
        ij_list, ee_list, pt_list, vg_list = self.trajectory_intersection_set()

        toi = wp.ones(1, dtype = float)
        dim_vg = vg_list.shape[0]
        dim_ee = ee_list.shape[0]
        dim_pt = pt_list.shape[0]
        if vg: 
            wp.launch(toi_vg, (dim_vg, ), inputs = [self.warp_affine_bodies, vg_list, toi])
        if ee:
            wp.launch(toi_ee, (dim_ee,), inputs = [self.warp_affine_bodies, ee_list, toi])
        if pt:
            wp.launch(toi_pt, (dim_pt, ), inputs = [self.warp_affine_bodies, pt_list, toi])

        t = toi.numpy()[0] 
        t = max(0.0, min(1.0, t))
        if t < 1.0:
            t *= 0.9
        return t

    def update_mesh_vertex(self, field = "x"):
        abs = self.affine_bodies
    
        Anp = self.states.A0.numpy() if field == "x_view" else self.states.Ak.numpy() if field == "xk" else self.states.A.numpy()
        pnp = self.states.p0.numpy() if field == "x_view" else self.states.pk.numpy() if field == "xk" else self.states.p.numpy()
        xmin = None
        for i, ab in enumerate(abs):
            x0 = ab.x0.numpy()
            A = Anp[i].T
            p = pnp[i]

            x = A @ x0.T + p.reshape(3, 1)
            getattr(ab, field).assign(x.T)
            _xmin = np.min(x, axis = 1)
            if i == 0: 
                xmin = _xmin
            else:
                xmin = np.minimum(xmin, _xmin)

        return xmin

    def update_q(self, alpha):
        wp.launch(_update_q, self.n_bodies, inputs = [self.states, self.dq, alpha])

    def update_qk(self, alpha):
        wp.launch(_update_qk, self.n_bodies, inputs = [self.states, self.dq, alpha])

    def update_q0qdot(self):
        wp.launch(_update_q0qdot, self.n_bodies, inputs = [self.states])

    
    def init(self):
        # temp
        wp.launch(_init, self.n_bodies, inputs = [self.states])

@wp.kernel
def _update_q(states: AffineBodyStates, dq: wp.array(dtype = wp.vec3), alpha: float):
    i = wp.tid()
    states.p[i] = states.p[i] - dq[i * 4 + 0] * alpha
    q1 = states.A[i][0] - dq[i * 4 + 1] * alpha
    q2 = states.A[i][1] - dq[i * 4 + 2] * alpha
    q3 = states.A[i][2] - dq[i * 4 + 3] * alpha
    states.A[i] = wp.transpose(wp.mat33(q1, q2, q3))

@wp.kernel
def _update_qk(states: AffineBodyStates, dq: wp.array(dtype = wp.vec3), alpha: float):
    i = wp.tid()
    states.pk[i] = states.p[i] - dq[i * 4 + 0] * alpha
    q1 = states.A[i][0] - dq[i * 4 + 1] * alpha
    q2 = states.A[i][1] - dq[i * 4 + 2] * alpha
    q3 = states.A[i][2] - dq[i * 4 + 3] * alpha
    states.Ak[i] = wp.transpose(wp.mat33(q1, q2, q3))

@wp.kernel
def _update_q0qdot(states: AffineBodyStates):
    i = wp.tid()
    states.pdot[i] = (states.p[i] - states.p0[i]) / dt
    states.Adot[i] = (states.A[i] - states.A0[i]) / dt

    states.p0[i] = states.p[i]
    states.A0[i] = states.A[i]

if __name__ == "__main__":
    import polyscope as ps
    import threading
    import fc_viewer as fc
    wp.init()
    np.printoptions(precision = 4, suppress=True)
    sim = AffineBodySimulator(config_file = "config.json")
    sim.init()
    viewer=fc.fast_cd_viewer()
    # ps.init()


    # sim_thread = threading.Thread(target=sim.step)
    # sim_thread.start()
    print("simulation started")
    ps_meshes = []
    highlight_color = np.array([1.0, 1.0, 0.0]) * 0.8
    regular_color = np.ones(3) * 0.4

    for i, body in enumerate(sim.scene.kinetic_objects):
        # ps_meshes.append(ps.register_surface_mesh(f"body{i}", body.V, body.F))
        if i != 0: 
            viewer.add_mesh(body.V.astype(np.float64), body.F)
        viewer.set_face_based(True, i)
        viewer.invert_normals(True, i)
        viewer.set_color(regular_color, i)

    
    bodies = sim.scene.kinetic_objects
    affine_bodies = sim.affine_bodies
    frame = 0
    paused = False
    def key_callback(key, modifier):
        global paused
        if key == ord('p') or key == ord('P') or key == ord(' '):
            paused = not paused
        elif key == ord('r') or key == ord('R'):
            sim.reset()

    def predraw():
        global frame, sim, paused
        if not paused:
            sim.step(frame)
        for i, body in enumerate(affine_bodies):
            V = body.x_view.numpy()
            b = sim.scene.kinetic_objects[i]
            viewer.set_mesh(V, b.F, i)
        frame += 1
        for i, h in enumerate(sim.highlight):
            viewer.set_color(highlight_color if h else regular_color, i)
    viewer.set_pre_draw_callback(predraw)   
    viewer.set_key_callback(key_callback)
    viewer.launch()
    

    # while(True):
    #     sim.step(frame)
    #     for ab, viewer_mesh in zip(affine_bodies, ps_meshes):
    #         viewer_mesh.update_vertex_positions(ab.x_view.numpy())    
    #     frame += 1
            
    #     ps.frame_tick()
        
        
        

