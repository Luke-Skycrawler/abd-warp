from const_params import *
from affine_body import WarpMesh
from typing import List, Any
from simulator.fenwick import ListMeta, insert_overload, list_with_meta, compress

from culling2 import *

def cull(ij_list, _bvh_list1, _bvh_list2 = None):
    if _bvh_list2 is None:
        bvh_list2 = _bvh_list1
    else:
        bvh_list2 = _bvh_list2
    # ijnp = ij_list.numpy()
    ijnp = np.array([[0, 1]])
    np1 = np.array([bvh.lowers.shape[0] for bvh in _bvh_list1])
    np2 = np.array([bvh.lowers.shape[0] for bvh in bvh_list2])

    prim_list = []
    for ii, ij in enumerate(ijnp):
        I = ij[0]
        J = ij[1]
        if _bvh_list2 is None:
            prim_list += [(I, J, ii, pi, pj) for pi in range(np1[I]) for pj in range(np2[J])]
        else :
            prim_list += [(I, J, ii, pi, pj) for pi in range(np1[I]) for pj in range(np2[J])]
            prim_list += [(J, I, ii, pj, pi) for pj in range(np1[J]) for pi in range(np2[I])]
    plnp = np.array(prim_list)
    plwp = wp.zeros((plnp.shape[0], ), dtype = vec5i)
    plwp.assign(plnp)
    return plwp

def cull_(ij_list, _bvh_list1, _bvh_list2 = None):
    bvh_list1 = extract_bvh_list(_bvh_list1)
    bvh_list2 = None
    shape = ij_list.shape[0]
    pt = 1
    if _bvh_list2 is None:
        bvh_list2 = bvh_list1
        pt = 0
    else:
        shape *= 2 # 2 for symmetric p1-t2 and p2-t1 processing
        bvh_list2 = extract_bvh_list(_bvh_list2)
        pt = 1

    prim_list, prim_meta = list_with_meta(vec5i, 256, shape)
    wp.launch(intersection_prims, (shape, ), inputs = [ij_list, bvh_list1, bvh_list2, prim_list, prim_meta, pt])

    # print(prim_meta.count.numpy())
    # print(prim_meta.count_overflow.numpy())
    
    prim_list = compress(prim_list, prim_meta)
    return prim_list

def cull_vg(lowers, _bvh_list, bodies):
    bvh_list = extract_bvh_list(_bvh_list)
    prim_list, prim_meta = list_with_meta(wp.vec2i, 256, lowers.shape[0])
    wp.launch(intersection_ground, dim = lowers.shape[0], inputs = [lowers, bvh_list, prim_list, prim_meta, dhat])

    prim_list = compress(prim_list, prim_meta)

    return prim_list

        
@wp.kernel
def intersection_ground(lowers: wp.array(dtype = wp.vec3), bvh_list: wp.array(dtype = Any), vg_list: wp.array(dtype = wp.vec2i), prim_meta: Any, dhat: float):
    i = wp.tid()
    if lowers[i][1] < 0.5 * dhat + ground:
        bound = 1e3
        u = wp.vec3(bound, ground, bound)
        l = wp.vec3(-bound, -bound, -bound)
        id = bvh_list[i].id
        query = wp.bvh_query_aabb(id, l, u)
        pid = int(0)
        while wp.bvh_query_next(query, pid):
            insert_vec2i(i, prim_meta, wp.vec2i(i, pid), vg_list)
            # if bodies[i].x[pid][1] < dhat:
            #     insert_vec2i(i, prim_meta, wp.vec2i(i, pid), vg_list)

    # i = wp.tid()
    # for pid in range(bodies[i].x.shape[0]):
    #     if bodies[i].x[pid][1] < dhat:
    #         insert_vec2i(i, prim_meta, wp.vec2i(i, pid), vg_list)


@wp.kernel
def intersection_prims(ij_list: wp.array(dtype = wp.vec2i), bvhs1: wp.array(dtype = Any), bvhs2: wp.array(dtype = Any), prim_list: wp.array(dtype = vec5i), prim_meta: Any, pt: int):
    i = wp.tid()
    ii = wp.select(pt == 1, i, i // 2)
    ij = ij_list[ii]
    I = wp.select(pt == 1 and i % 2 == 1, ij[0], ij[1])
    J = ij[0] + ij[1] - I
    
    for pi in range(bvhs1[I].lowers.shape[0]):
        id = bvhs2[J].id
        query = wp.bvh_query_aabb(id, bvhs1[I].lowers[pi], bvhs1[I].uppers[pi])
        pj = int(0)
        while wp.bvh_query_next(query, pj):
            item = vec5i(I, J, ii, pi, pj)
            insert_vec5i(i, prim_meta, item, prim_list)

insert_vec5i = insert_overload(vec5i)

if __name__ == "__main__":
    from abd import AffineBodySimulator
    wp.init()
    sim = AffineBodySimulator(config_file = "config.json")
    sim.init()
    
    bodies = sim.scene.kinetic_objects
    affine_bodies = sim.affine_bodies

    n_bodies = len(affine_bodies)

    def body_body_test():
        bb = BvhBuilder()
        bvh_bodies = bb.build_body_bvh(sim.affine_bodies, dhat * 0.5)
        ij_list, ij_meta = list_with_meta(wp.vec2i, 4, 1)
        wp.launch(intersection_bodies, n_bodies, inputs = [bvh_bodies.id, bvh_bodies.lowers, bvh_bodies.uppers, ij_meta, ij_list])
        
        print(ij_list.numpy())

    ij_list, ee_list, pt_list = sim.collision_set()
    print(ee_list.numpy())
    print(pt_list.numpy())
    