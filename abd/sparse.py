import warp as wp

@wp.struct
class BSR:
    stride: wp.array(dtype = int)
    offset: wp.array(dtype = int)
    blocks: wp.array(dtype = wp.mat33)

    
def bsr_empty(n = 1, nnz = 1):
    bsr = BSR()
    bsr.stride = wp.zeros(n, dtype = int)
    bsr.offset = wp.zeros(n, dtype = int)
    bsr.blocks = wp.zeros(nnz  * 16, dtype = wp.mat33)
    return bsr