from simulator.scene import KineticMesh, Scene
import warp as wp
import numpy as np
import igl
from const_params import ground, vec5i

@wp.struct
class AffineBodyStates:
    A: wp.array(dtype = wp.mat33)
    p: wp.array(dtype = wp.vec3)
    
    Ak: wp.array(dtype = wp.mat33)  # used in line search
    pk: wp.array(dtype = wp.vec3)   # used in line search
    
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
class WarpMesh:
    id: int
    x0: wp.array(dtype = wp.vec3)       # rest shape
    x: wp.array(dtype = wp.vec3)        # current position
    xk: wp.array(dtype = wp.vec3)
                # line search t1 position
    x_view: wp.array(dtype = wp.vec3)   # view position
    
    triangles: wp.array2d(dtype = int)  # triangle indices
    edges: wp.array2d(dtype = int)      # edge indices

class AffineMesh(KineticMesh):
    '''
    For io only. Send to WarpMesh struct for CUDA simulation 
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
        self.edges = igl.edges(self.F)

    def assign_to(self, ab: WarpMesh):
        ab.triangles.assign(self.F)
        ab.edges.assign(self.edges)

        ab.x0.assign(self.V)

        Vt = self.V @ self.A.T + self.p.reshape(1, 3)
        ab.x.assign(Vt)
        ab.x_view.assign(Vt)
        ab.xk.assign(Vt)

    def warp_affine_body(self, id) -> WarpMesh:
        ab = WarpMesh()
        ab.id = id
        ab.x0 = wp.zeros(self.V.shape[0], dtype = wp.vec3)
        ab.x = wp.zeros_like(ab.x0)
        ab.xk = wp.zeros_like(ab.x0)
        ab.x_view = wp.zeros_like(ab.x0)

        ab.triangles = wp.zeros(self.F.shape, dtype = int)
        ab.edges = wp.zeros(self.edges.shape, dtype = int)

        self.assign_to(ab)

        return ab



@wp.func
def vg_distance(v: wp.vec3) -> float:
    return v[1] - ground

@wp.func
def fetch_pt(ijpt: vec5i, bodies: wp.array(dtype = WarpMesh)): 
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[3]
    tid = ijpt[4]

    T = bj.triangles[tid]

    p = bi.x[pid]
    t0 = bj.x[T[0]]
    t1 = bj.x[T[1]]
    t2 = bj.x[T[2]]
    return p, t0, t1, t2

@wp.func
def fetch_pt_x0(ijpt: vec5i, bodies: wp.array(dtype = WarpMesh)): 
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[3]
    tid = ijpt[4]

    T = bj.triangles[tid]

    p = bi.x0[pid]
    t0 = bj.x0[T[0]]
    t1 = bj.x0[T[1]]
    t2 = bj.x0[T[2]]
    return p, t0, t1, t2

@wp.func
def fetch_pt_xk(ijpt: vec5i, bodies: wp.array(dtype = WarpMesh)): 
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[3]
    tid = ijpt[4]

    T = bj.triangles[tid]

    p = bi.xk[pid]
    t0 = bj.xk[T[0]]
    t1 = bj.xk[T[1]]
    t2 = bj.xk[T[2]]
    return p, t0, t1, t2

@wp.func
def fetch_ee(ijee: vec5i, bodies: wp.array(dtype = WarpMesh)):
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eiid = ijee[3]
    ejid = ijee[4]

    EI = bi.edges[eiid]
    EJ = bj.edges[ejid]

    ei0 = bi.x[EI[0]]
    ei1 = bi.x[EI[1]]

    ej0 = bj.x[EJ[0]]
    ej1 = bj.x[EJ[1]]

    return ei0, ei1, ej0, ej1

@wp.func
def fetch_ee_xk(ijee: vec5i, bodies: wp.array(dtype = WarpMesh)):
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eiid = ijee[3]
    ejid = ijee[4]

    EI = bi.edges[eiid]
    EJ = bj.edges[ejid]

    ei0 = bi.xk[EI[0]]
    ei1 = bi.xk[EI[1]]

    ej0 = bj.xk[EJ[0]]
    ej1 = bj.xk[EJ[1]]

    return ei0, ei1, ej0, ej1

@wp.func
def fetch_ee_x0(ijee: vec5i, bodies: wp.array(dtype = WarpMesh)):
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eiid = ijee[3]
    ejid = ijee[4]

    EI = bi.edges[eiid]
    EJ = bj.edges[ejid]

    ei0 = bi.x0[EI[0]]
    ei1 = bi.x0[EI[1]]

    ej0 = bj.x0[EJ[0]]
    ej1 = bj.x0[EJ[1]]

    return ei0, ei1, ej0, ej1

@wp.func
def fetch_vertex(ip: wp.vec2i, bodies: wp.array(dtype = WarpMesh)):
    I = ip[0]
    pid = ip[1]

    return bodies[I].x[pid]

@wp.func
def fetch_xk(ip: wp.vec2i, bodies: wp.array(dtype = WarpMesh)):
    I = ip[0]
    pid = ip[1]

    return bodies[I].xk[pid]
