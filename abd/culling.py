from const_params import *
from affine_body import AffineBody
from typing import List
from simulator.fenwick import ListMeta, insert_overload, list_with_meta, compress

from culling2 import *



def cull(ij_list, ij_meta: ListMeta, _bvh_list1, _bvh_list2 = None):
    bvh_list1 = extract_bvh_list(_bvh_list1)
    bvh_list2 = None
    shape = ij_meta.compressed_size.numpy()[0]
    pt = 1
    if _bvh_list2 is None:
        bvh_list2 = bvh_list1
        pt = 0
    else:
        shape *= 2 # 2 for symmetric p1-t2 and p2-t1 processing
        bvh_list2 = extract_bvh_list(_bvh_list2)
        pt = 1

    prim_list, prim_meta = list_with_meta(vec5i, 32, 1, 16)
    wp.launch(intersection_prims, shape, inputs = [ij_list, bvh_list1, bvh_list2, prim_list, pt])
    compress(prim_list, prim_meta)
    return prim_list

insert_vec5i = insert_overload(vec5i)
insert_vec2i = insert_overload(wp.vec2i)

@wp.kernel
def intersection_prims(ij_list: wp.array(dtype = wp.vec2i), bvhs1: wp.array(dtype = BvhStruct), bvhs2: wp.array(dtype = BvhStruct), prim_list: wp.array(dtype = vec5i), prim_meta: ListMeta, pt: int):
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

    
if __name__ == "__main__":
    from abd import AffineBodySimulator
    wp.init()
    sim = AffineBodySimulator(config_file = "config.json")
    sim.init()
    
    bodies = sim.scene.kinetic_objects
    affine_bodies = sim.affine_bodies

    n_bodies = len(affine_bodies)

    bb = BvhBuilder()
    bvh_bodies = bb.build_body_bvh(sim.affine_bodies, dhat * 0.5)
    ij_list, ij_meta = list_with_meta(wp.vec2i, 4, 1, 4)
    wp.launch(intersection_bodies, n_bodies, inputs = [bvh_bodies.id, bvh_bodies.lowers, bvh_bodies.uppers, ij_meta, ij_list])
    
    print(ij_list.numpy())