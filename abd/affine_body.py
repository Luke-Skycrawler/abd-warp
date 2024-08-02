from simulator.base import BaseSimulator
from simulator.scene import KineticMesh, Scene
from const_params import *
import igl

@wp.struct
class AffineBodyStates:
    A: wp.array(dtype = wp.mat33)
    p: wp.array(dtype = wp.vec3)
    
    Ak: wp.array(dtype = wp.mat33)
    pk: wp.array(dtype = wp.vec3)
    
    Adot: wp.array(dtype = wp.mat33)
    pdot: wp.array(dtype = wp.vec3)

    A0: wp.array(dtype = wp.mat33)
    p0: wp.array(dtype = wp.vec3)

    mass: wp.array(dtype = float)
    I0: wp.array(dtype = float)

def affine_body_states_empty(n_bodies):
    
    states = AffineBodyStates()

    states.A = wp.zeros(n_bodies, dtype = wp.mat33)
    states.p = wp.zeros(n_bodies, dtype = wp.vec3)

    states.Ak = wp.zeros_like(states.A)
    states.pk = wp.zeros_like(states.p)

    states.A0 = wp.zeros_like(states.A)
    states.p0 = wp.zeros_like(states.p)

    states.Adot = wp.zeros_like(states.A)
    states.pdot = wp.zeros_like(states.p)

    return states

@wp.struct
class AffineBody:
    id: int
    x0: wp.array(dtype = wp.vec3)       # rest shape
    x: wp.array(dtype = wp.vec3)        # current position
    x_view: wp.array(dtype = wp.vec3)   # view position
    
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
        
    def warp_affine_body(self, id):
        ab = AffineBody()
        ab.id = id
        ab.x0 = wp.zeros(self.V.shape[0], dtype = wp.vec3)
        ab.x = wp.zeros_like(ab.x0)
        ab.x_view = wp.zeros_like(ab.x0)

        ab.triangles = wp.zeros(self.F.shape, dtype = int)
        ab.triangles.assign(self.F)

        edges = igl.edges(self.F)
        ab.edges = wp.zeros(edges.shape, dtype = int)
        ab.edges.assign(edges)

        ab.x0.assign(self.V)
        ab.x.assign(self.V)
        ab.x_view.assign(self.V)

        return ab

