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

