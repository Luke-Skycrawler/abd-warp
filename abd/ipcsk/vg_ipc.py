from ipcsk.barrier import *   
from affine_body import vg_distance, WarpMesh
from typing import Any
@wp.kernel
def ipc_term_vg(vg_list: wp.array(dtype = wp.vec2i), 
                bodies: wp.array(dtype = Any), 
                g: wp.array(dtype = vec3), blocks: wp.array(dtype = mat33)):
    i = wp.tid()

    col = vg_list[i]
    b = col[0]
    vid = col[1]

    v = bodies[b].x[vid]
    vtile = bodies[b].x0[vid]

    nabla_d = vec3(scalar(0.0), scalar(1.0), scalar(0.0))
    d = vg_distance(v)
    d2bdb2 = barrier_derivative2(d * d)
    dbdd = barrier_derivative(d * d)

    os = b * 16
    for ii in range(4):
        theta_ii = wp.select(ii == 0, vtile[ii - 1], scalar(1.0))
        wp.atomic_add(g, 4 * b + ii, d * scalar(2.0) * nabla_d * theta_ii * dbdd)
        # g[4 * b + ii] += d * scalar(2.0) * nabla_d * theta_ii 
        for jj in range(4):
            theta_jj = wp.select(jj == 0, vtile[jj - 1], scalar(1.0))
            
            nabla_d_nabla_dT = wp.outer(nabla_d, nabla_d)

            dh = nabla_d_nabla_dT * (theta_ii * theta_jj * d2bdb2 * scalar(4.0) * d * d)
            # blocks[os + ii + jj * 4] += dh
            wp.atomic_add(blocks, os + ii + jj * 4, dh)
