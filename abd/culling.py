import warp as wp
from affine_body import AffineBody
from typing import List
import numpy as np
from simulator.fenwick import ListMeta, insert, list_with_meta
@wp.kernel
def bvh_triangles(dialation: float, V: wp.array(dtype = wp.vec3), F: wp.array2d(dtype = int), uppers: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3)):
    tid = wp.tid()

    triangle = wp.mat33(V[F[tid, 0]], V[F[tid, 1]], V[F[tid, 2]])
    lx = wp.min(triangle[0])
    ly = wp.min(triangle[1])
    lz = wp.min(triangle[2])
    
    ux = wp.max(triangle[0])
    uy = wp.max(triangle[1])
    uz = wp.max(triangle[2])

    uppers[tid] = wp.vec3(ux, uy, uz) + wp.vec3(dialation)
    lowers[tid] = wp.vec3(lx, ly, lz) - wp.vec3(dialation)

@wp.kernel
def bvh_edges(dialation: float, V: wp.array(dtype = wp.vec3), E: wp.array2d(dtype = int), uppers: wp.array(dtype = wp.vec3), lowers = wp.array2d(dtype = wp.vec3)):
    
    eid = wp.tid()

    edge = wp.mat32f(V[E[eid, 0]], V[E[eid, 1]])
    lx = wp.min(edge[0])
    ly = wp.min(edge[1])
    lz = wp.min(edge[2])

    ux = wp.max(edge[0])
    uy = wp.max(edge[1])
    uz = wp.max(edge[2])

    uppers[eid] = wp.vec3(ux, uy, uz) + wp.vec3(dialation)
    lowers[eid] = wp.vec3(lx, ly, lz) - wp.vec3(dialation)

class BvhBuilder:
    '''
    vertices should be projected before calling any of the member functions
    '''
    def __init__(self):
        pass

    def bulid_triangle_bvh(self, V, F, dialation):

        uppers = wp.zeros(F.shape[0], dtype = wp.vec3)
        lowers = wp.zeros_like(uppers)

        wp.launch(bvh_triangles, F.shape[0], inputs = [dialation, V, F, uppers, lowers])

        bvh = wp.Bvh(lowers, uppers)
        return lowers, uppers, bvh

    def build_edge_bvh(self, V, E, dialation):
        uppers = wp.zeros(E.shape[0], dtype = wp.vec3)
        lowers = wp.zeros_like(uppers)  

        wp.launch(bvh_edges, E.shape[0], inputs = [dialation, V, E, uppers, lowers])

        bvh = wp.Bvh(lowers, uppers)
        return lowers, uppers, bvh

    def build_body_bvh(self, bodies: List[AffineBody], dialation):

        uppers = []
        lowers = []
        for b in bodies:
            vnp = b.x0.numpy()
            upper = np.max(vnp, axis = 0)
            lower = np.min(vnp, axis = 0)
            uppers.append(upper + dialation)
            lowers.append(lower - dialation)    
        
        _uppers = wp.from_numpy(np.array(uppers), dtype = wp.vec3, shape = (len(uppers)))
        _lowers = wp.from_numpy(np.array(lowers), dtype = wp.vec3, shape = (len(lowers)))

        bvh = wp.Bvh(_lowers, _uppers)
        return _lowers, _uppers, bvh


@wp.kernel
def intersection_bodies(
    bvh_bodies: wp.uint64,
    lowers: wp.array(dtype = wp.vec3),
    upper: wp.array(dtype = wp.vec3),
    meta: ListMeta,
    list: wp.array(dtype = wp.vec2i)
):
    bi = wp.tid()

    query = wp.bvh_query_aabb(bvh_bodies, lowers[bi], upper[bi])
    bj = int(0)
    while wp.bvh_query_next(query, bj):
        if bj > bi:
            insert(bi, meta, wp.vec2i(bi, bj), list)



if __name__ == "__main__":
    from abd.abd import AffineBodySimulator
    wp.init()
    sim = AffineBodySimulator(config_file = "config.json")
    sim.init()
    
    bodies = sim.scene.kinetic_objects
    affine_bodies = sim.affine_bodies

    n_bodies = len(affine_bodies)

    bb = BvhBuilder()
    body_lowers, body_uppers, bvh_bodies = bb.build_body_bvh()
    ij_list, ij_meta = list_with_meta(wp.vec2i, 4, 1, 4)
    wp.launch(intersection_bodies, n_bodies, inputs = [bvh_bodies, body_lowers, body_uppers, ij_meta, ij_list])
    
    print(ij_list.numpy())