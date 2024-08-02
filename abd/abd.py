from const_params import *
from simulator.base import BaseSimulator
from affine_body import AffineBody, AffineMesh, AffineBodyStates, affine_body_states_empty
from warp.utils import array_inner
from warp.optim.linear import bicgstab
from sparse import BSR, bsr_empty
from warp.sparse import bsr_set_from_triplets, bsr_zeros
from orthogonal_energy import InertialEnergy
from ipc import IPCContactEnergy

# temporary
from orthogonal_energy import _init, _set_triplets

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


        self.bsr = bsr_empty(n_bodies)

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

        # self.affine_bodies = wp.array([ko.warp_affine_body() for ko in self.scene.kinetic_objects], dtype = AffineBody)
        self.affine_bodies = [ko.warp_affine_body(i) for i, ko in enumerate(self.scene.kinetic_objects)]
        
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

    def proximity_set(self):
        pass

    def compute_energy(self, alpha):
        return 0.0

    def dot(self, a, b):
        return array_inner(a, b)

    def line_search(self, alpha_cap, E0):

        qTg = self.dot(self.dq, self.g)

        alpha = alpha_cap * 0.9 if alpha_cap < 1.0 else 1.0

        return 1.0, 0.0
        # temp
        while True:
            E1 = self.compute_energy(alpha)
            wolfe = E1 < E0 + c1 * alpha * qTg
            if wolfe:
                break
            alpha *= 0.5
        return alpha, E1

    def collision_set(self):
        return wp.zeros(0, dtype = wp.vec2i), wp.zeros(0, dtype = vec5i), wp.zeros(0, dtype = vec5i)
    
    def V_gets_V(self, states):
        pass

    def step(self, frame = 1):
        states = self.states
        inertia = self.inertia
        g = self.g
        bsr = self.bsr
        rows, cols, hess = self.rows, self.cols, self.hess
        dq = self.dq
        ipc_contact = self.ipc_contact
        
        self.q_gets_q0()
        self.V_gets_V(states)
        ij_list, ee_list, pt_list = self.collision_set()
        E0 = self.compute_energy(0.0)
        self.blocks = wp.zeros(shape = ((self.n_bodies + ij_list.shape[0] * 2) * 16), dtype = wp.mat33)
        it = 0 
        while True:
            inertia.gradient(g, states)
            inertia.hessian(self.blocks, states)

            ipc_contact.gradient([g, states, ij_list, ee_list, pt_list])
            ipc_contact.hessian([self.blocks, states, ij_list, ee_list, pt_list])

            wp.launch(_set_triplets, self.n_bodies, inputs = [self.n_bodies,  ij_list.shape[0], ij_list, rows, cols])
            bsr_set_from_triplets(hess, rows, cols, self.blocks)
            bicgstab(hess, g, dq, 1e-4)

            # print(bsr.blocks.numpy())
            # print(hess.values.numpy())
            # print(g.numpy())
            # print(dq.numpy())
            alpha_cap = self.ccd(states, dq)
            alpha, E0 = self.line_search(alpha_cap, E0)

            
            self.update_q(alpha)
            
            it += 1
            cond = self.dot(g, g) < tol
            if cond or it > 1:
                # fixme: temp
                break

        self.update_q0qdot()
        self.update_mesh_vertex()
        # print("a dot = ", states.Adot.numpy())

    def ccd(self, states, dq):
        return 1.0

    def update_mesh_vertex(self):
        abs = self.affine_bodies
        Anp = self.states.A0.numpy()
        pnp = self.states.p0.numpy()

        for i, ab in enumerate(abs):
            x0 = ab.x0.numpy()
            A = Anp[i].T
            p = pnp[i]
            x_view = A @ x0.T + p.reshape(3, 1)
            ab.x_view.assign(x_view.T)

    def update_q(self, alpha):
        wp.launch(_update_q, self.n_bodies, inputs = [self.states, self.dq, alpha])

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
def _update_q0qdot(states: AffineBodyStates):
    i = wp.tid()
    states.pdot[i] = (states.p[i] - states.p0[i]) / dt
    states.Adot[i] = (states.A[i] - states.A0[i]) / dt

    states.p0[i] = states.p[i]
    states.A0[i] = states.A[i]

if __name__ == "__main__":
    import polyscope as ps
    import threading
    wp.init()
    sim = AffineBodySimulator(config_file = "config.json")
    sim.init()
    ps.init()

    # sim_thread = threading.Thread(target=sim.step)
    # sim_thread.start()

    ps_meshes = []
    for i, body in enumerate(sim.scene.kinetic_objects):
        ps_meshes.append(ps.register_surface_mesh(f"body{i}", body.V, body.F))

    
    bodies = sim.scene.kinetic_objects
    affine_bodies = sim.affine_bodies
    frame = 0
    while(True):
        sim.step(frame)
        for ab, viewer_mesh in zip(affine_bodies, ps_meshes):
            viewer_mesh.update_vertex_positions(ab.x_view.numpy())    
        frame += 1
            
        ps.frame_tick()
        
        
        

