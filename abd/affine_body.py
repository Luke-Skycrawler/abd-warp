from simulator.base import BaseSimulator
from simulator.scene import KineticMesh, Scene
from abd.const_params import *
import igl

@wp.struct
class AffineBody:
    x0: wp.array(dtype = wp.vec3)       # rest shape
    x: wp.array(dtype = wp.vec3)        # current position
    
    triangles: wp.array2d(dtype = int)  # triangle indices
    edges: wp.array2d(dtype = int)      # edge indices

class AffineMesh(KineticMesh):
    '''
    For io only. Send to AffineBody struct for CUDA simulation 
    '''
    @classmethod
    def states_default(cls):
        return {
            "A": np.identity(3, dtype = float),
            "p": np.zeros(3, dtype = float),
            "Adot": np.zeros((3,3), dtype = float),
            "pdot": np.zeros(3, dtype = float),
            "mass": 1.0,
            "I0": np.identity(3, dtype = float)
        }
    
    def __init__(self, obj_json):
        super().__init__(obj_json)
        
    def warp_affine_body(self):
        ab = AffineBody()
        ab.x0 = wp.zeros(self.V.shape[0], dtype = wp.vec3)
        ab.x = wp.zeros_like(ab.x0)

        ab.triangles = wp.zeros(self.F.shape, dtype = int)
        ab.triangles.assign(self.F)

        edges = igl.edges(self.F)
        ab.edges = wp.zeros(edges.shape, dtype = int)
        ab.edges.assign(edges)

        return ab

class AffineBodySimulator(BaseSimulator):

    def __init__(self, config_file = "config.json"):
        super().__init__(member_type = AffineMesh, config_file = config_file) 

        n_bodies = len(self.scene.kinetic_objects)

        # A, p, Adot, pdot flattened together
        # while mesh information is stored in AffineBody struct

        self.A = wp.zeros(n_bodies, dtype = wp.mat33)
        self.p = wp.zeros(n_bodies, dtype = wp.vec3)

        self.A0 = wp.zeros_like(self.A)
        self.p0 = wp.zeros_like(self.p)

        self.Adot = wp.zeros_like(self.A)
        self.pdot = wp.zeros_like(self.p)

        self.A0.assign(self.gather("A"))
        self.p0.assign(self.gather("p"))
        self.pdot.assign(self.gather("pdot"))
        self.Adot.assign(self.gather("Adot"))

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

    def step(self):
        pass

