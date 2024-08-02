import warp as wp
import numpy as np 

@wp.func
def digit(i: int, d: int) -> int:
    if d > 0:
        i = i >> d
    return i & 1

@wp.kernel
def build_tree(a: wp.array1d(dtype = int), ptree: wp.array1d(dtype = int), digits: int):
    i = wp.tid()

    for d in range(digits + 1):
        if digit(i, d) != 1:
            iadd = ((( i >> d) + 1) << d) - 1
            wp.atomic_add(ptree, iadd, a[i])
        
@wp.kernel
def prefix_sum(ptree: wp.array1d(dtype = int), sum: wp.array1d(dtype = int), digits: int):
    i = wp.tid()
    sum[i] = 0
    for d in range(digits + 1):
        if digit(i + 1, d) == 1:
            iadd = (((i + 1) >> d) << d) - 1
            sum[i] = sum[i] + ptree[iadd]

@wp.kernel
def prefix_slow(a: wp.array1d(dtype = int), ref: wp.array1d(dtype = int)):
    i = wp.tid()
    for j in range(i + 1):
        ref[i] = ref[i] + a[j]

class FenwickTree:
    def __init__(self, digits):
        self.digits = digits
        self.N = N = 1 << digits
        self.a = wp.zeros((N), dtype = int)
        self.p = wp.zeros((N), dtype = int)
        self.sum = wp.zeros((N), dtype = int)
        
    def prefix_fast(self, arr, n = -1):
        wp.copy(self.a, arr)
        self.p.zero_()
        if n == -1:
            n = self.N
        else:
            n = min(n, self.N)
        wp.launch(build_tree, dim = n, inputs = [self.a, self.p, self.digits])
        wp.launch(prefix_sum, dim = n, inputs = [self.p, self.sum, self.digits])

    
@wp.kernel
def init(a: wp.array1d(dtype = int)):
    i = wp.tid()
    # a[i] = wp.randi(wp.uint32(i)) % 2
    a[i] = i % 2

class FenwickTest(FenwickTree):
    def __init__(self, digits):
        super().__init__(digits)
        self.ref = wp.zeros((self.N), dtype = int)
    
    def init(self):
        wp.launch(init, dim = self.N, inputs = [self.a])

    def test(self):
        self.init()
        self.p.zero_()
        wp.launch(build_tree, dim = self.N, inputs = [self.a, self.p, self.digits])
        wp.launch(prefix_sum, dim = self.N, inputs = [self.p, self.sum, self.digits])
        if self.digits <= 16:
            wp.launch(prefix_slow, dim = (self.N), inputs = [self.a, self.ref])

        ref = self.ref.numpy()
        sum = self.sum.numpy()
        print(f"max error = {np.max(self.ref.numpy() - self.sum.numpy())}")
        print("ref = ", ref, " sum = ", sum)

        # print(f"prefix sum {self.sum.numpy()}")
        # print(f"p {self.p.numpy()}")
        # print(f"ref {self.ref.numpy()}")

@wp.kernel
def init(a: wp.array1d(dtype = int), deleted: wp.array1d(dtype = int)):
    i = wp.tid()
    a[i] = i + 1
    seed = wp.rand_init(i)
    deleted[i] = wp.randi(seed, 0, 2)

@wp.kernel
def _compress(prefix_deleted: wp.array1d(dtype = int), a: wp.array1d(dtype = int), b: wp.array1d(dtype = int), deleted: wp.array1d(dtype = int)):
    i = wp.tid()
    if not deleted[i]:
        b[i - prefix_deleted[i]] = a[i]


import warp as wp 
from typing import Any
@wp.struct
class ListMeta:
    count: wp.array(dtype = int)
    volume_per_thread: int
    volume_overflow: int
    count_overflow: wp.array(dtype = int)

@wp.func
def insert(tid: int, meta: ListMeta, item: Any, a: wp.array(dtype = Any)):
    overflow_offset = meta.volume_per_thread * a.shape[0]
    if meta.count[tid] < meta.volume_per_thread:
        a[meta.count[tid]] = item
        meta.count[tid] += 1
    else:
        id = wp.atomic_add(meta.count_overflow, 0, 1)
        a[overflow_offset + id] = item

class WarpList:
    def __init__(self, type, thread_count):
        digits = np.ceil(np.log2(thread_count)).astype(int)
        self.ft = FenwickTree(digits)
        self.N = self.ft.N
        self.a = wp.zeros((self.N), dtype = type)
        self.b = wp.zeros((self.N), dtype = type)    


        self.meta = ListMeta()
        self.meta.count = wp.zeros((self.N), dtype = int)
        self.meta.count 
        return 
        

class Deleter:
    def __init__(self, digits):
        self.ft = FenwickTree(digits)
        self.N = self.ft.N
        self.a = wp.zeros((self.N), dtype = int)
        self.b = wp.zeros((self.N), dtype = int)    
        self.deleted = wp.zeros((self.N), dtype = int)

    def test(self):
        wp.launch(init, dim = self.N, inputs = [self.a, self.deleted])
        self.ft.prefix_fast(self.deleted)
        wp.launch(_compress, dim = self.N, inputs = [self.ft.sum, self.a, self.b, self.deleted])
        n_elements = self.ft.N - self.ft.sum.numpy()[self.N - 1]
        if self.N < 100:
            print(f"prefix = {self.ft.sum.numpy()}")
            print(f"a init = {self.a.numpy()}")
            print(f"delete = {self.deleted.numpy()}")
            print(f"b = {self.b.numpy()}")
            print(f"elements left = {n_elements}")
        else: 
            select = self.a.numpy()[self.deleted.numpy() == 0] 
            if np.all(select == self.b.numpy()[:len(select)]) and len(select) == n_elements:
                print("check passed")
            else: 
                print("fail")


if __name__ == "__main__":
    wp.init()
    # ft = FenwickTest(24)
    # ft.test()
    # print("Done!")

    deleter = Deleter(24)
    deleter.test()