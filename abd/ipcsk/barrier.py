from const_params import *
        
@wp.func
def barrier(d: float) -> float:
    ret = 0.0

    if d < d2hat:
        dbydhat = d / d2hat
        ret = kappa * - wp.pow((dbydhat - 1.0), 2.0) * wp.log(dbydhat)
    return ret

def barrier_np(d: float) -> float:
    ret = 0.0

    if d < d2hat:
        dbydhat = d / d2hat
        ret = kappa * - (dbydhat - 1.0) ** 2.0 * np.log(dbydhat)
    return ret

@wp.func
def barrier_derivative(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = kappa * (d2hat - d) * (2.0 * wp.log(d / d2hat) + (d - d2hat) / d) / (d2hat * d2hat)

    return ret

def barrier_derivative_np(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = kappa * (d2hat - d) * (2.0 * np.log(d / d2hat) + (d - d2hat) / d) / (d2hat * d2hat)

    return ret

@wp.func
def barrier_derivative2(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = -kappa * (2.0 * wp.log(d / d2hat) + (d - d2hat) / d + (d - d2hat) * (2.0 / d + d2hat / (d * d))) / (d2hat * d2hat)
    return ret

def barrier_derivative2_np(d: float) -> float:
    ret = 0.0
    if d < d2hat:
        ret = -kappa * (2.0 * np.log(d / d2hat) + (d - d2hat) / d + (d - d2hat) * (2.0 / d + d2hat / (d * d))) / (d2hat * d2hat)
    return ret

