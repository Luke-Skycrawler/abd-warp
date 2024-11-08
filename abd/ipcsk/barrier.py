from const_params import *
        
@wp.func
def barrier(d: scalar) -> scalar:
    ret = scalar(0.0)

    if d < d2hat:
        dbydhat = d / d2hat
        ret = kappa * - wp.pow((dbydhat - scalar(scalar(1.0))), scalar(scalar(2.0))) * wp.log(dbydhat)
    return ret

def barrier_np(d):
    ret = 0.0

    if d < d2hat:
        dbydhat = d / d2hat
        ret = kappa * - (dbydhat - scalar(1.0)) ** scalar(2.0) * np.log(dbydhat)
    return ret

@wp.func
def barrier_derivative(d: scalar) -> scalar:
    ret = scalar(0.0)
    if d < d2hat:
        ret = kappa * (d2hat - d) * (scalar(2.0) * wp.log(d / d2hat) + (d - d2hat) / d) / (d2hat * d2hat)

    return ret

def barrier_derivative_np(d):
    ret = 0.0
    _d2hat = float(d2hat)
    if d < _d2hat:
        ret = float(kappa) * (_d2hat - d) * (2.0 * np.log(d / _d2hat) + (d - _d2hat) / d) / (_d2hat * _d2hat)

    return ret

@wp.func
def barrier_derivative2(d: scalar) -> scalar:
    ret = scalar(0.0)
    if d < d2hat:
        ret = -kappa * (scalar(2.0) * wp.log(d / d2hat) + (d - d2hat) / d + (d - d2hat) * (scalar(2.0) / d + d2hat / (d * d))) / (d2hat * d2hat)
    return ret

def barrier_derivative2_np(d: scalar) -> scalar:
    ret = 0.0
    _d2hat = float(d2hat)
    if d < _d2hat:
        ret = -kappa * (2.0 * np.log(d / _d2hat) + (d - _d2hat) / d + (d - _d2hat) * (2.0 / d + _d2hat / (d * d))) / (_d2hat * _d2hat)
    return ret

