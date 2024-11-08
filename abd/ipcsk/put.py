from const_params import *
def put_grad(g, gnp, pt_list):
    npt = pt_list.shape[0]
    gwp = wp.from_numpy(gnp.reshape(npt, 8, 3), dtype = vec3, shape = (npt, 8))
    wp.launch(_put_grad, dim = (npt, ), inputs = [gwp, pt_list, g])

def put_hess(blocks, Hinp, Hjnp, Hijnp, pt_list, nij, n_bodies):
    npt = pt_list.shape[0]
    Hjwp = to_wp(Hjnp)
    Hijwp = to_wp(Hijnp)
    Hiwp = to_wp(Hinp)
    wp.launch(_put_hess, dim = (npt, ), inputs = [blocks, Hiwp, Hjwp, Hijwp, pt_list, nij, n_bodies])

def to_wp(H):
    npt = H.shape[0]
    Hn = np.zeros((npt, 4, 4, 3, 3))
    for i in range(npt):
        for ii in range(4):
            for jj in range(4):
                Hn[i, ii, jj] = H[i, ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3]

    Hwp = wp.from_numpy(Hn, dtype = mat33, shape = (npt, 4, 4))
    return Hwp

@wp.kernel
def _put_hess(blocks: wp.array(dtype = mat33), Hi: wp.array3d(dtype = mat33), Hj: wp.array3d(dtype = mat33), Hij: wp.array3d(dtype = mat33), pt_list: wp.array(dtype = vec5i), nij: int, n_bodies: int):
    i = wp.tid()
    I = pt_list[i][0]
    J = pt_list[i][1]
    idx = pt_list[i][2]
    
    for ii in range(4):
        for jj in range(4):
            wp.atomic_add(blocks, 16 * I + ii + jj * 4, Hi[i, ii, jj])
            wp.atomic_add(blocks, 16 * J + ii + jj * 4, Hj[i, ii, jj])

            # blocks[16 * I + ii + jj * 4] += Hi[i, ii, jj]
            # blocks[16 * J + ii + jj * 4] += Hj[i, ii, jj]
            if I < J:
            # if J < I:
                # Hij should be put to upper triangle
                # blocks[16 * (n_bodies + idx) + ii + jj * 4] += Hij[i, ii, jj]
                # blocks[16 * (n_bodies + idx + nij) + jj + ii * 4] += wp.transpose(Hij[i, ii, jj])
                wp.atomic_add(blocks, 16 * (n_bodies + idx) + ii + jj * 4, Hij[i, ii, jj])
                wp.atomic_add(blocks, 16 * (n_bodies + idx + nij) + jj + ii * 4, wp.transpose(Hij[i, ii, jj]))
            else:
                # Hij.T should be put to upper triangle
                wp.atomic_add(blocks, 16 * (n_bodies + idx) + jj + ii * 4, wp.transpose(Hij[i, ii, jj]))
                wp.atomic_add(blocks, 16 * (n_bodies + idx + nij) + ii + jj * 4, Hij[i, ii, jj])
                

@wp.kernel
def _put_grad(gnp: wp.array2d(dtype = vec3), pt_list: wp.array(dtype = vec5i), g: wp.array(dtype = vec3)):
    i = wp.tid()
    I = pt_list[i][0]
    J = pt_list[i][1]

    wp.atomic_add(g, 4 * I + 0, gnp[i, 0])
    wp.atomic_add(g, 4 * I + 1, gnp[i, 1])
    wp.atomic_add(g, 4 * I + 2, gnp[i, 2])
    wp.atomic_add(g, 4 * I + 3, gnp[i, 3])

    wp.atomic_add(g, 4 * J + 0, gnp[i, 4])
    wp.atomic_add(g, 4 * J + 1, gnp[i, 5])
    wp.atomic_add(g, 4 * J + 2, gnp[i, 6])
    wp.atomic_add(g, 4 * J + 3, gnp[i, 7])
