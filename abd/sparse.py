import warp as wp
from const_params import mat33
@wp.struct
class BSR:
    stride: wp.array(dtype = int)
    offset: wp.array(dtype = int)
    blocks: wp.array(dtype = mat33)

    
def bsr_empty(n = 1, nnz = 1):
    bsr = BSR()
    bsr.stride = wp.zeros(n, dtype = int)
    bsr.offset = wp.zeros(n, dtype = int)
    bsr.blocks = wp.zeros(nnz  * 16, dtype = mat33)
    return bsr