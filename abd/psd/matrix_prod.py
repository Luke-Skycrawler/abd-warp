import warp as wp
@wp.kernel
def dcdx_sTq_dcdx(a: wp.array2d(dtype = float), q: wp.array2d(dtype = wp.vec3), ret: wp.array2d(dtype = wp.mat33), l: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
    for ii in range(5):
        for jj in range(4):
            l[jj, ii] = a[0, jj] * q[ii, 0] + a[1, jj] * q[ii, 1] + a[2, jj] * q[ii, 2]

    for i in range(5):
        theta = lam[0, i] / (wp.length_sq(q[i, 0]) + wp.length_sq(q[i, 1]) + wp.length_sq(q[i, 2]))
        lam[0, i] = wp.max(0.0, theta)

    for ii in range(4):
        for jj in range(4):
            h = wp.mat33(0.0)
            for kk in range(5):
                h += wp.outer(l[ii, kk], l[jj, kk] * lam[0, kk])
            ret[ii, jj] = h

    
@wp.kernel
def d2Psidx2(ret: wp.array2d(dtype = wp.mat33), d2Psi: wp.array2d(dtype = wp.mat33), dcdx_simple: wp.array2d(dtype = float), dcdx_delta: wp.array2d(dtype = wp.mat33)):
    ii, ll = wp.tid()
    h = wp.mat33(0.0)
    # H = dcdx_simple.T @ d2Psi @ dcdx_simple - dcdx_delta.T @ d2Psi @ dcdx_delta
    for jj in range(3):
        for kk in range(3):
            h += dcdx_simple[jj, ii] * d2Psi[jj, kk] * dcdx_simple[kk, ll] - wp.transpose(dcdx_delta[jj, ii]) @ d2Psi[jj, kk] @ dcdx_delta[kk, ll]
    ret[ii, ll] = h


@wp.kernel
def d2Psi_psd(d2Psi: wp.array2d(dtype = wp.mat33), q: wp.array2d(dtype = wp.vec3), lam: wp.array2d(dtype = float)):
    for i in range(5):
        if lam[0, i] < 0.0:
            theta = lam[0, i] / (wp.length_sq(q[i, 0]) + wp.length_sq(q[i, 1]) + wp.length_sq(q[i, 2]))
            for ii in range(3):
                for jj in range(3):
                    d2Psi[ii, jj] -= wp.outer(q[i, ii], q[i, jj]) * theta

