import warp as wp
import numpy as np

x_error = wp.constant(1e-10)

@wp.func
def evaluate(coef: wp.vec4, x: float):
    return coef[0] + x * (coef[1] + x * (coef[2] + x * coef[3]))

@wp.func
def multi_sign(a: float, b: float) -> float:
    ret = a
    if b < 0.0:
        ret = -a
    return ret

@wp.func
def is_different_sign(y0: float, yr: float):
    return y0 * yr < 0.0

@wp.func
def deflate(coef: wp.vec4, root: float):
    defpoly = wp.vec3(0.0)
    defpoly[2] = coef[3]
    for i in range(2, 0, -1):
        defpoly[i - 1] = coef[i] + root * defpoly[i]
    return defpoly

@wp.func
def find_closed(coef: wp.vec4, deriv: wp.vec4, x0: float, x1: float, y0: float, y1: float):
    xr = (x0 + x1) / 2.0
    yr = evaluate(coef, xr)
    xb0 = float(x0)
    xb1 = float(x1)
    if x1 - x0 <= x_error * 2.0:
        pass
    else:
        while True:
            side = is_different_sign(y0, yr)
            if side:
                xb1 = xr
            else:
                xb0 = xr

            dy = evaluate(deriv, xr)
            dx = yr / dy
            xn = xr - dx

            if xn > xb0 and xn < xb1:
                stepsize = wp.abs(xr - xn)
                xr = xn
                if stepsize > x_error:
                    yr = evaluate(coef, xr)
                else:
                    break
            else:
                xr = (xb0 + xb1) / 2.0
                yr = evaluate(coef, xr)
                if xb0 == xr or xb1 == xr or xb1 - xb0 <= 2.0 * x_error:
                    break

    return xr
@wp.func
def quadratic_roots(coef: wp.vec3, x0: float, x1: float):

    roots = wp.vec2(0.0)
    c = coef[0]
    b = coef[1]
    a = coef[2]
    delta = b * b - 4.0 * a * c
    ret = int(0)
    if delta > 0.0:
        d = wp.sqrt(delta)
        q = -(b + multi_sign(d, b)) * 0.5
        rv0 = q / a
        rv1 = c / q
        xa = wp.min(rv0, rv1)
        xb = wp.max(rv0, rv1)

        if xa >= x0 and xa <= x1:
            roots[ret] = xa
            ret += 1
        if xb >= x0 and xb <= x1:
            roots[ret] = xb
            ret += 1
    elif delta < 0.0:
        ret = 0
    else:
        r0 = -0.5 * b / a
        roots[0] = r0
        if r0 >= x0 and r0 <= x1:
            ret = 1
    return ret, roots

@wp.func
def cubic_roots(coef: wp.vec4, x0: float, x1: float):
    roots = wp.vec3(0.0, 0.0, 0.0)  
    y0 = evaluate(coef, x0)
    y1 = evaluate(coef, x1)

    # Coefficients of derivative
    a = coef[3] * 3.0
    b_2 = coef[2]  # b / 2
    c = coef[1]
    deriv = wp.vec4(c, 2.0 * b_2, a, 0.0)
    delta_4 = b_2 * b_2 - a * c
    ret = int(0)
    while True:
        if delta_4 > 0.0:
            d_2 = wp.sqrt(delta_4)
            q = -(b_2 + multi_sign(d_2, b_2))
            rv0 = q / a
            rv1 = c / q
            xa = wp.min(rv0, rv1)
            xb = wp.max(rv0, rv1)

            if is_different_sign(y0, y1):
                if xa >= x1 or xb <= x0 or (xa <= x0 and xb >= x1):
                    roots[0] = find_closed(coef, deriv, x0, x1, y0, y1)
                    ret = 1
                    break
            else:
                if (xa >= x1 or xb <= x0 or (xa <= x0 and xb >= x1)):
                    ret = 0
                    break
            if xa > x0:
                ya = evaluate(coef, xa)
                if is_different_sign(y0, ya):
                    roots[0] = find_closed(coef, deriv, x0, xa, y0, ya)

                    if is_different_sign(ya, y1) or (xb < x1 and is_different_sign(ya, evaluate(coef, xb))):
                        defpoly = deflate(coef, roots[0])
                        _ret, qroots = quadratic_roots(defpoly, xa, x1)
                        roots[1] = qroots[0]
                        roots[2] = qroots[1]
                        ret = _ret + 1
                        break
                    else:
                        ret = 1
                        break

                if xb < x1:
                    yb = evaluate(coef, xb)

                    if is_different_sign(ya, yb):
                        roots[0] = find_closed(coef, deriv, xa, xb, ya, yb)
                        if is_different_sign(yb, y1):
                            defpoly = deflate(coef, roots[0])
                            _ret, qroots = quadratic_roots(defpoly, xb, x1)
                            roots[1] = qroots[0]
                            roots[2] = qroots[1]
                            ret = _ret + 1
                            break
                        else :
                            ret = 1
                            break
                    if is_different_sign(yb, y1):
                        roots[0] = find_closed(coef, deriv, xb, x1, yb, y1)
                        ret = 1
                        break
                elif is_different_sign(ya, y1):
                    roots[0] = find_closed(coef, deriv, xa, x1, ya, y1)
                    ret = 1
                    break

            else:
                yb = evaluate(coef, xb)
                if is_different_sign(y0, yb):
                    roots[0] = find_closed(coef, deriv, x0, xb, y0, yb)
                    if is_different_sign(yb, y1):
                        defpoly = deflate(coef, roots[0])
                        _ret, qroots = quadratic_roots(defpoly, xb, x1)
                        roots[1] = qroots[0]
                        roots[2] = qroots[1]
                        ret = _ret + 1
                        break
                    else :
                        ret = 1
                        break
                if is_different_sign(yb, y1):
                    roots[0] = find_closed(coef, deriv, xb, x1, yb, y1)
                    ret = 1
                    break
        else:
            if is_different_sign(y0, y1):
                roots[0] = find_closed(coef, deriv, x0, x1, y0, y1)
                ret = 1
                break
            else:
                ret = 0
                break
        break
    return ret, roots

@wp.kernel
def test(coeffs:wp.array(dtype = wp.vec4), _roots: wp.array(dtype = wp.vec3), _ret: wp.array(dtype = int), rand_init: int):
    i = wp.tid()
    state = wp.rand_init(100 * i)
    if rand_init:
        coeffs[i] = wp.vec4(wp.randf(state), wp.randf(state), wp.randf(state), wp.randf(state))
    ret, roots = cubic_roots(coeffs[i], -5.0, 5.0)

    _ret[i] = ret
    _roots[i] = roots

@wp.kernel
def test_quadratic(coeffs:wp.array(dtype = wp.vec3), _roots: wp.array(dtype = wp.vec2), _ret: wp.array(dtype = int), rand_init: int):
    i = wp.tid()
    if rand_init:
        state = wp.rand_init(100 * i)
        coeffs[i] = wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))
    ret, roots = quadratic_roots(coeffs[i], -5.0, 5.0)

    _ret[i] = ret
    _roots[i] = roots

@wp.kernel
def verify_roots(coeffs: wp.array(dtype = wp.vec4), roots: wp.array(dtype = wp.vec3), n_roots: wp.array(dtype = int), err: wp.array2d(dtype = float)):
    i = wp.tid()
    coef = coeffs[i]
    for j in range(n_roots[i]):
        err[i, j] = wp.abs(evaluate(coef, roots[i][j]))


def construct_3_root_polynomial(n_test, bound):
    def rand():
        abc = np.random.rand(3) * bound
        a = np.min(abc)
        c = np.max(abc)
        b = np.sum(abc) - a - c

        abc = np.array([a, b, c])
        return abc

    def gen_poly(abc):
        a, b, c = abc
        a0 = np.random.rand() * bound
        assert a0 != 0.0
        # coeffs = np.array([1.0, -(a + b + c), a * c + b * c + a * b, -a * b * c]) * a0
        coeffs = np.array([- a* b * c, a * b + a * c + b * c, -a - b - c, 1.0]) * a0
        return coeffs

    roots = np.array([rand() for _ in range(n_test)])
    test_coeffs = np.array([gen_poly(roots[i]) for i in range(n_test)])
    
    return test_coeffs, roots

def construct_2_root_polynomial(n_test, bound):
    def rand():
        ab = np.random.rand(2) * bound
        a = np.min(ab)
        b = np.max(ab)

        ab = np.array([a, b])
        return ab

    def gen_poly(ab):
        a, b = ab
        a0 = np.random.rand() * bound
        assert a0 != 0.0
        # coeffs = np.array([1.0, -(a + b), a * b]) * a0
        coeffs = np.array([a * b, -(a + b), 1.0]) * a0
        return coeffs
    

    roots = np.array([rand() for _ in range(n_test)])
    test_coeffs = np.array([gen_poly(roots[i]) for i in range(n_test)])
    
    return test_coeffs, roots

@wp.kernel
def test_deflate(coef: wp.array(dtype = wp.vec4), roots: wp.array(dtype = wp.vec3), ret: wp.array(dtype = int), qroots: wp.array(dtype = wp.vec2)):
    i = wp.tid()


    defpoly = deflate(coef[i], roots[i][0])
    _ret, _qroots = quadratic_roots(defpoly, -5.0, 5.0)
    qroots[i] = _qroots
    ret[i] = _ret

        
        
    
    
if __name__ == "__main__":
    wp.init()
    n_test = 9
    roots = wp.zeros((n_test, ), dtype = wp.vec3)
    ret = wp.zeros(n_test, dtype = int)
    coeffs = wp.zeros(n_test, dtype = wp.vec4)
    err = wp.zeros((n_test, 3), dtype = float)


    # constructed polynomials
    # 3 roots
    bound = 5.0
    qtest_coeffs, refqroots = construct_2_root_polynomial(n_test, bound)

    test_coeffs, refroots = construct_3_root_polynomial(n_test, bound)
    coeffs.assign(test_coeffs)

    wp.launch(test, n_test, inputs = [coeffs ,roots, ret, 0])

    qroots = wp.zeros((n_test, ), dtype = wp.vec2)
    qcoeffs = wp.zeros(n_test, dtype = wp.vec3)
    qcoeffs.assign(qtest_coeffs)
    
    # test quadratic, passed
    # wp.launch(test_quadratic, n_test, inputs = [qcoeffs , qroots, ret, 0])
    # 0 to disable random init test

    wp.launch(verify_roots, n_test, inputs = [coeffs, roots, ret, err])
    
    # test deflate, passed
    # roots.assign(refroots)
    # wp.launch(test_deflate, n_test, inputs = [coeffs, roots, ret, qroots])
    
    if n_test < 10:
        # print(coeffs.numpy())
        print(roots.numpy())
        print(ret.numpy())
        # print(err.numpy())
        # print(qroots.numpy())
        print(refroots)
