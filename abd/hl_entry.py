from psd.hl import *
from psd.matrix_prod import *
if __name__ == "__main__":
    wp.config.enable_backward = False
    wp.config.max_unroll = 0
    wp.init()
    np.set_printoptions(precision=4, suppress=True)    
    arr = np.array([
        [0.4360, 0.0259, 0.5497],
        [0.4353, 0.4204, 0.3303],
        [0.2046, 0.6193, 0.2997],
        [0.2668, 0.6211, 0.5291]
    ])

    dcdx_simple = wp.zeros((3, 4), dtype = float)
    x = wp.from_numpy(arr, dtype = wp.vec3, shape = (4))
    dcdx_delta = wp.zeros((3, 4), dtype = wp.mat33)
    ret = wp.zeros((4, 4), dtype = wp.mat33)
    d2Psi = wp.zeros((3, 3), dtype = wp.mat33)
    q = wp.zeros((9,3), dtype = wp.vec3)
    lam = wp.zeros((1, 9), dtype = float)
    l = wp.zeros((4, 5), dtype = wp.vec3)

    def to_numpy(__d2Psi):
        _d2Psi = __d2Psi.numpy()
        d2Psidc = np.zeros((3 * __d2Psi.shape[0], 3 * __d2Psi.shape[1]))
        for ii in range(__d2Psi.shape[0]):
            for jj in range(__d2Psi.shape[1]):
                d2Psidc[ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3] = _d2Psi[ii, jj]
        return d2Psidc

    def test2():
        wp.launch(test_vf, 1, inputs = [x, dcdx_delta, ret, d2Psi, dcdx_simple])
        wp.launch(verify_eig_sys_vf, 1, inputs = [x, q, lam])
        d2Psidc = to_numpy(d2Psi)
        wp.launch(d2Psi_psd, 1, inputs = [d2Psi, q, lam])
        d2Psidc1 = to_numpy(d2Psi)

        # wp.launch(dcdx_sTq_dcdx, 1, inputs = [dcdx_simple, q, ret, l, lam])

        Q = q.numpy().reshape(9, 9).T
        QTQ = Q.T @ Q
        diag_inv = np.array([(0.0 if i >= 5 else (1.0 / QTQ[i, i])) for i in range(9)])

        Lambda = np.diag(lam.numpy().reshape(9))

        diaginv = np.diag(diag_inv)
        Q_inv = diaginv @ Q.T
        # print(Q)
        # print(Q_inv @ d2Psidc @ Q)
        # print(Q_inv @ d2Psidc1 @ Q)
        # print(Q_inv @ d2Psidc2 @ Q)

        # print(Q @ Lambda @ Q_inv)
        Lambda_pos = np.diag(np.maximum(lam.numpy().reshape(9), np.zeros(9)))
        # print(Q @ Lambda_pos @ Q_inv)
        # print(d2Psidc1)
        # print(d2Psidc1 - Q @ Lambda_pos @ Q_inv)


        _dc_s = dcdx_simple.numpy().reshape(3, 4)
        dc_s = np.kron(_dc_s, np.eye(3))
        dc_d = to_numpy(dcdx_delta)

        # _ret = to_numpy(ret)
        # print(_ret)
        # print(dc_s.T @ d2Psidc1 @ dc_s)
        # print(dc_s.T @ d2Psidc1 @ dc_s - _ret)

        t = wp.array(dtype = float, shape = (3, 5))
        a = wp.array(dtype = wp.vec3, shape = (3, 4))
        wp.launch(dcdx_delta_kernel, 1, inputs = [q, lam, x, t, ret, a])

        _ret = to_numpy(ret)
        ref = dc_d.T @ d2Psidc @ dc_d
        print(dc_d.T @ d2Psidc @ dc_d)
        print(dc_d, d2Psidc)
        # print(_ret)
        # print(ref - _ret)

        # print(d2Psidc - Q @ Lambda @ Q_inv)
        
    
    # test2()

    def test():
        wp.launch(test_vf, 1, inputs = [x, dcdx_delta, ret, d2Psi, dcdx_simple])
        # wp.launch(test_ee, 1, inputs = [x, dcdx_delta, ret, d2Psi, dcdx_simple])
        print(ret.numpy())
        ret.zero_()
        wp.launch(d2Psidx2, (4, 4), inputs = [ret, d2Psi, dcdx_simple, dcdx_delta])
        print(ret.numpy())


        wp.launch(verify_eig_sys_vf, 1, inputs = [x, q, lam])
        Q = q.numpy().reshape(9, 9).T
        Lambda = np.diag(lam.numpy().reshape(9))
        # print(Q)
        # print(Lambda)
        # print(Q @ Lambda)
        # assemble H from numpy
        _dc_s = dcdx_simple.numpy().reshape(3, 4)
        _d2Psi = d2Psi.numpy()

        d2Psidc = to_numpy(d2Psi)
        dc_s = np.kron(_dc_s, np.eye(3))
        dc_d = to_numpy(dcdx_delta)

        # dc_d = np.zeros((9, 12))
        # _dc_d = dcdx_delta.numpy()
        # for ii in range(3):
        #     for jj in range(4):
        #         dc_d[ii * 3: ii * 3 + 3, jj * 3: jj * 3 + 3] = _dc_d[ii, jj]
        
        H = dc_s.T @ d2Psidc @ dc_s - dc_d.T @ d2Psidc @ dc_d
        H2 = dc_d.T @ d2Psidc @ dc_d
        print(dc_d, d2Psidc)
        print(H2)

        # print(d2Psidc @ Q - Q @ Lambda)
        d2Psidc1 = project_psd(d2Psidc, Q, Lambda)
        d2Psidc2 = project_psd(-d2Psidc, Q, -Lambda)
        QTQ = Q.T @ Q
        diag_inv = np.array([(0.0 if i >= 5 else (1.0 / QTQ[i, i])) for i in range(9)])

        diaginv = np.diag(diag_inv)
        Q_inv = diaginv @ Q.T

        # print(Q_inv @ Q)
        print(Q_inv @ d2Psidc @ Q)
        # print(Q_inv @ d2Psidc1 @ Q)
        # print(Q_inv @ d2Psidc2 @ Q)
        # print(np.diag(Q.T @ d2Psidc1 @ Q))
        # print(Q.T @ Q)
        print(Q @ Lambda @ Q_inv)
        print(d2Psidc)

        # Q = Q[:, : 5]
        # Lambda = Lambda[:5, : 5]
        # Q_inv = Q_inv[:5, :]
        # print(Q @ Lambda @ Q_inv)

        # Lambda = np.maximum(Lambda, np.zeros((5, 5)))
        # print(Q @ Lambda @ Q_inv)
        # print(Q)
        H1 = dc_s.T @ d2Psidc1 @ dc_s
        print(H1)
    test()