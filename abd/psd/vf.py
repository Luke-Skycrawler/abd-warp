import warp as wp

@wp.func
def C_vf(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    # point-triangle
    e0 = x1 - x2
    e1 = x3 - x2
    e2 = x0 - x2

    alpha = wp.dot(e0, e1) / wp.dot(e0, e0)

    e1perp = e1 - alpha * e0
    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = wp.mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = wp.vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b

    e2perp = e2 - beta_gamma[0] * e0 - beta_gamma[1] * e1
    return e0, e1perp, e2perp

@wp.func
def dcvfdx_s(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    # point-triangle
    e0 = x1 - x2
    e1 = x3 - x2
    e2 = x0 - x2

    alpha = wp.dot(e0, e1) / wp.dot(e0, e0)

    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = wp.mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = wp.vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b
    beta = beta_gamma[0]
    gamma = beta_gamma[1]

    mat34 = wp.matrix(dtype = float, shape = (3, 4))
    mat34[0, 1] = 1.0
    mat34[0, 2] = -1.0
    mat34[1, 1] = -alpha
    mat34[1, 2] = alpha - 1.0
    mat34[1, 3] = 1.0
    mat34[2, 0] = 1.0
    mat34[2, 1] = -beta
    mat34[2, 2] = beta + gamma - 1.0
    mat34[2, 3] = -gamma
    return mat34
    # return mat34(
    #     0.0, 1.0, -1.0, 0.0,
    #     0.0, -alpha, alpha - 1.0, 1.0,
    #     1.0, -beta, beta + gamma - 1.0, -gamma
    # )

@wp.func
def dalpha_dx(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    e0 = x1 - x2
    e1 = x3 - x2

    e0h = wp.normalize(e0)
    term = 1.0 / wp.length_sq(e0)
    return term * (e1 - 2.0 * e0h * wp.dot(e0h, e1)), term * (2.0 * e0h * wp.dot(e0h, e1) - e1 - e0), term * e0
    
@wp.func
def dbeta_dx(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    e0 = x1 - x2
    e1 = x3 - x2
    e2 = x0 - x2
    

    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = wp.mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = wp.vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b
    bet = beta_gamma[0]

    e1n2 = wp.length_sq(e1)
    e0n2 = wp.length_sq(e0)
    e0d1 = wp.dot(e0, e1)
    e0d2 = wp.dot(e0, e2)
    e1d2 = wp.dot(e1, e2)
    e0x12 = wp.length_sq(wp.cross(e0, e1))
    
    dbdv0 = (e1n2 * e0 - e0d1 * e1)
    dbdv1 = (2.0 * bet * (e0d1 * e1 - e1n2 * e0) - e1d2 * e1 + e1n2 * e2)
    dbdv2 = (2.0 * bet * (e1n2 * e0 + e0n2 * e1 - e0d1 * (e0 + e1)) + e1d2 * (e0 + e1) - 2.0 * e0d2 * e1 - e1n2 * (e0 + e2) + e0d1 * (e1 + e2))
    dbdv3 = 2.0 * bet * (e0d1 * e0 - e0n2 * e1) - e1d2 * e0 + 2.0 * e0d2 * e1 - e0d1 * e2

    return dbdv0 / e0x12, dbdv1 / e0x12, dbdv2 / e0x12, dbdv3 / e0x12
    
@wp.func
def dgamma_dx(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    e0 = x1 - x2
    e1 = x3 - x2
    e2 = x0 - x2
    

    e0Te0 = wp.dot(e0, e0)
    e0Te1 = wp.dot(e0, e1)
    e1Te1 = wp.dot(e1, e1)
    A = wp.mat22(e0Te0, e0Te1, e0Te1, e1Te1)
    b = wp.vec2(wp.dot(e0, e2), wp.dot(e1, e2))

    beta_gamma = wp.inverse(A) @ b
    gam = beta_gamma[1]



    e1n2 = wp.length_sq(e1)
    e0n2 = wp.length_sq(e0)
    e0d1 = wp.dot(e0, e1)
    e0d2 = wp.dot(e0, e2)
    e1d2 = wp.dot(e1, e2)
    e0x12 = wp.length_sq(wp.cross(e0, e1))
    
    dgdv0 = -e0d1 * e0 + e0n2 * e1
    dgdv1 = 2.0 * gam * (e0d1 * e1 - e1n2 * e0) + 2.0 * e1d2 * e0 - e0d2 * e1 - e0d1 * e2
    dgdv2 = 2.0 * gam * (e1n2 * e0 - e0d1 * (e0 + e1) + e0n2 * e1) - 2.0 * e1d2 * e0 + e0d2 * (e0 + e1) + e0d1 * (e0 + e2) - e0n2 * (e1 + e2)
    dgdv3 = 2.0 * gam * (e0d1 * e0 - e0n2 * e1) - e0d2 * e0 + e0n2 * e2

    return dgdv0 / e0x12, dgdv1 / e0x12, dgdv2/ e0x12, dgdv3 / e0x12

@wp.func
def dcdx_delta_vf(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, ret: wp.array2d(dtype = wp.mat33)):

    e0 = x1 - x2
    e1 = x3 - x2

    dadx1, dadx2, dadx3 = dalpha_dx(x0, x1, x2, x3)
    dbdx0, dbdx1, dbdx2, dbdx3 = dbeta_dx(x0, x1, x2, x3)
    dgdx0, dgdx1, dgdx2, dgdx3 = dgamma_dx(x0, x1, x2, x3)

    z3 = wp.mat33(0.0)
    ret[0, 0] = z3
    ret[0, 1] = z3
    ret[0, 2] = z3
    ret[0, 3] = z3

    ret[1, 0] = z3
    ret[1, 1] = - wp.outer(e0, dadx1)
    ret[1, 2] = - wp.outer(e0, dadx2)
    ret[1, 3] = - wp.outer(e0, dadx3)

    ret[2, 0] = -wp.outer(e0, dbdx0) - wp.outer(e1, dgdx0)
    ret[2, 1] = -wp.outer(e0, dbdx1) - wp.outer(e1, dgdx1)
    ret[2, 2] = -wp.outer(e0, dbdx2) - wp.outer(e1, dgdx2)
    ret[2, 3] = -wp.outer(e0, dbdx3) - wp.outer(e1, dgdx3)
    
    return