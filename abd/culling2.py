import warp as wp
from typing import List
import numpy as np
from affine_body import AffineBody
from simulator.fenwick import ListMeta, insert_overload
insert_vec2i = insert_overload(wp.vec2i)
mat32f = wp.types.matrix(shape = (3, 2), dtype = float)
@wp.struct 
class BvhStruct: 
    '''
    Helper struct to send an array of Bvh to warp kernel
    '''
    uppers: wp.array(dtype = wp.vec3)
    lowers: wp.array(dtype = wp.vec3)
    id: wp.uint64

    
def extract_bvh_list(bvh_list: List[wp.Bvh]):
    '''
    Helper function to convert an array of Bvh to warp array
    '''
    def _extract(bvh: wp.Bvh):
        bs = BvhStruct()
        bs.uppers = bvh.uppers
        bs.lowers = bvh.lowers
        bs.id = bvh.id
        return bs
    return wp.array([_extract(bvh) for bvh in bvh_list], dtype = BvhStruct)

@wp.kernel
def bvh_triangles(dialation: float, V: wp.array(dtype = wp.vec3), F: wp.array2d(dtype = int), uppers: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3)):
    tid = wp.tid()

    v1 = V[F[tid, 0]]    
    v2 = V[F[tid, 1]]
    v3 = V[F[tid, 2]]

    triangle = wp.mat33(v1, v2, v3)
    lx = wp.min(triangle[0])
    ly = wp.min(triangle[1])
    lz = wp.min(triangle[2])
    
    ux = wp.max(triangle[0])
    uy = wp.max(triangle[1])
    uz = wp.max(triangle[2])

    uppers[tid] = wp.vec3(ux, uy, uz) + wp.vec3(dialation)
    lowers[tid] = wp.vec3(lx, ly, lz) - wp.vec3(dialation)

@wp.kernel
def bvh_edges(dialation: float, V: wp.array(dtype = wp.vec3), E: wp.array2d(dtype = int), uppers: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3)):
    
    eid = wp.tid()
    v0 = V[E[eid, 0]]
    v1 = V[E[eid, 1]]

    edge = mat32f(v0, v1)
    lx = wp.min(edge[0])
    ly = wp.min(edge[1])
    lz = wp.min(edge[2])

    ux = wp.max(edge[0])
    uy = wp.max(edge[1])
    uz = wp.max(edge[2])

    uppers[eid] = wp.vec3(ux, uy, uz) + wp.vec3(dialation)
    lowers[eid] = wp.vec3(lx, ly, lz) - wp.vec3(dialation)

@wp.kernel
def bvh_points(dialation: float, V: wp.array(dtype = wp.vec3), uppers: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3)):
    pid = wp.tid()

    l = V[pid] - wp.vec3(dialation)
    u = V[pid] + wp.vec3(dialation)

    uppers[pid] = u
    lowers[pid] = l


    
class BvhBuilder:
    '''
    vertices should be projected before calling any of the member functions
    '''
    def __init__(self):
        pass

    def build_point_bvh(self, V, dialation):
        uppers = wp.zeros(V.shape[0], dtype = wp.vec3)
        lowers = wp.zeros_like(uppers)
        wp.launch(bvh_points, V.shape[0], inputs = [dialation, V, uppers, lowers])

        bvh = wp.Bvh(lowers, uppers)
        return bvh

    def update_point_bvh(self, V, bvh: wp.Bvh, dialation):
        wp.launch(bvh_points, V.shape[0], inputs = [dialation, V, bvh.uppers, bvh.lowers])
        bvh.refit()
        return bvh

    def bulid_triangle_bvh(self, V, F, dialation):

        uppers = wp.zeros(F.shape[0], dtype = wp.vec3)
        lowers = wp.zeros_like(uppers)

        wp.launch(bvh_triangles, F.shape[0], inputs = [dialation, V, F, uppers, lowers])

        bvh = wp.Bvh(lowers, uppers)
        return bvh

    def update_triangle_bvh(self, V, F, dialation, bvh: wp.Bvh):
        wp.launch(bvh_triangles, F.shape[0], inputs = [dialation, V, F, bvh.uppers, bvh.lowers])
        bvh.refit()
        return bvh

    def build_edge_bvh(self, V, E, dialation):
        uppers = wp.zeros(E.shape[0], dtype = wp.vec3)
        lowers = wp.zeros_like(uppers)  

        wp.launch(bvh_edges, E.shape[0], inputs = [dialation, V, E, uppers, lowers])

        bvh = wp.Bvh(lowers, uppers)
        return bvh

    def update_edge_bvh(self, V, E, dialation, bvh: wp.Bvh):
        wp.launch(bvh_edges, E.shape[0], inputs = [dialation, V, E, bvh.uppers, bvh.lowers])
        bvh.refit()
        return bvh

    def build_body_bvh(self, bodies: List[AffineBody], dialation):

        uppers = []
        lowers = []
        for b in bodies:
            vnp = b.x.numpy()
            upper = np.max(vnp, axis = 0)
            lower = np.min(vnp, axis = 0)
            uppers.append(upper + dialation)
            lowers.append(lower - dialation)    
            print(upper, lower)
        
        _uppers = wp.from_numpy(np.array(uppers), dtype = wp.vec3, shape = (len(uppers)))
        _lowers = wp.from_numpy(np.array(lowers), dtype = wp.vec3, shape = (len(lowers)))

        bvh = wp.Bvh(_lowers, _uppers)
        return bvh


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
            insert_vec2i(bi, meta, wp.vec2i(bi, bj), list)