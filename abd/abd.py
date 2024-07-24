from const_params import *
from simulator.base import BaseSimulator
from affine_body import AffineBody, AffineMesh
from warp.utils import array_inner


class AffineBodySimulator(BaseSimulator):

    n_tmp = 5
    collision_cap = 1 << 15

    def __init__(self, config_file = "config.json"):
        super().__init__(member_type = AffineMesh, config_file = config_file) 

        n_bodies = len(self.scene.kinetic_objects)

        # A, p, Adot, pdot flattened together
        # while mesh information is stored in AffineBody struct

        self.A = wp.zeros(n_bodies, dtype = wp.mat33)
        self.p = wp.zeros(n_bodies, dtype = wp.vec3)

        self.Ak = wp.zeros_like(self.A)
        self.pk = wp.zeros_like(self.p)

        self.A0 = wp.zeros_like(self.A)
        self.p0 = wp.zeros_like(self.p)

        self.Adot = wp.zeros_like(self.A)
        self.pdot = wp.zeros_like(self.p)

        self.A0.assign(self.gather("A"))
        self.p0.assign(self.gather("p"))
        self.pdot.assign(self.gather("pdot"))
        self.Adot.assign(self.gather("Adot"))

        self.energy = wp.zeros(2, dtype = float)
        self.alpha = wp.zeros(1, dtype = float)
        
        self.tmp_ret = wp.zeros(self.n_tmp, dtype = float)



        self.dq = wp.zeros((n_bodies, 12), dtype = float)
        self.r = wp.zeros((n_bodies, 12), dtype = float)


        self.pt_body_index = wp.zeros((self.collision_cap, 2), dtype = int)
        self.pt_prim_index = wp.zeros((self.collision_cap, 2), dtype = int)

        self.ee_body_index = wp.zeros((self.collision_cap, 2), dtype = int)
        self.ee_prim_index = wp.zeros((self.collision_cap, 2), dtype = int)
        
        self.g_body_index = wp.zeros(self.collision_cap, dtype = int)
        self.g_prim_index = wp.zeros(self.collision_cap, dtype = int)

        






        


        self.rows = np.zeros()
        

        

        self.affine_bodies = wp.array([ko.warp_affine_body() for ko in self.scene.kinetic_objects], dtype = AffineBody)

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
        wp.copy(self.A, self.A0)
        wp.copy(self.p, self.p0)

    def proximity_set(self):
        pass

    def compute_energy(self):
        return 0.0

    def dot(self, a, b):
        return array_inner(a, b)

    def line_search(self, alpha_cap):

        E0 = self.compute_energy(0.0)
        qTg = self.dot(self.dq, self.r)

        alpha = alpha_cap * 0.9

        while True:
            E1 = self.compute_energy(alpha)
            wolfe = E1 < E0 + c1 * alpha * qTg


    def step(self):
        self.q_gets_q0()
        self.proximity_set()

        while True:

        

