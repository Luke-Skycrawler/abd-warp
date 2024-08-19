import warp as wp
from vf import dalpha_dx, dbeta_dx, dgamma_dx

@wp.kernel
def dcdx_delta_kernel(q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float), x: wp.array(dtype = wp.vec3), t: wp.array2d(dtype = float), ret: wp.array2d(dtype = wp.mat33), a: wp.array2d(dtype = wp.vec3)):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]

    e0 = x1 - x2
    e1 = x3 - x2

    dadx1, dadx2, dadx3 = dalpha_dx(x0, x1, x2, x3)
    dbdx0, dbdx1, dbdx2, dbdx3 = dbeta_dx(x0, x1, x2, x3)
    dgdx0, dgdx1, dgdx2, dgdx3 = dgamma_dx(x0, x1, x2, x3)

    a[0, 0] = wp.vec3(0.0)
    a[0, 1] = dadx1
    a[0, 2] = dadx2
    a[0, 3] = dadx3

    a[1, 0] = dbdx0
    a[1, 1] = dbdx1
    a[1, 2] = dbdx2
    a[1, 3] = dbdx3

    a[2, 0] = dgdx0
    a[2, 1] = dgdx1
    a[2, 2] = dgdx2
    a[2, 3] = dgdx3

    # a = wp.mat44(
    #     0.0, dadx1, dadx2, dadx3, 
    #     dbdx0, dbdx1, dbdx2, dbdx3,
    #     dgdx0, dgdx1, dgdx2, dgdx3,
    #     0.0, 0.0, 0.0, 0.0)
    
    for ii in range(5):
        t[0, ii] = wp.dot(-e0, q[ii, 1])
        t[1, ii] = wp.dot(-e0, q[ii, 2])
        t[2, ii] = wp.dot(-e1, q[ii, 2])

    sum0 = float(0.0)
    sum1 = float(0.0)
    sum2 = float(0.0)
    for ii in range(5):
        theta = lam[0, ii] / (wp.length_sq(q[ii, 0]) + wp.length_sq(q[ii, 1]) + wp.length_sq(q[ii, 2]))
        sum0 += t[0, ii] * theta * t[0, ii]

        sum1 += t[1, ii] * theta * t[1, ii]
        sum2 += t[2, ii] * theta * t[2, ii]

    for ii in range(4):
        for jj in range(4):
            h = wp.outer(a[0, ii], a[0, jj]) * sum0 + wp.outer(a[1, ii], a[1, jj]) * sum1 + wp.outer(a[2, ii], a[2, jj]) * sum2
            ret[ii, jj] = h

    

    
