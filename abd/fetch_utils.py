from const_params import *
from affine_body import AffineBody


@wp.func
def vg_distance(v: wp.vec3) -> float:
    return v[1]

@wp.func
def fetch_pt(ijpt: vec5i, bodies: wp.array(dtype = AffineBody)): 
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[3]
    tid = ijpt[4]

    T = bj.triangles[tid]

    p = bi.x[pid]
    t0 = bj.x[T[0]]
    t1 = bj.x[T[1]]
    t2 = bj.x[T[2]]
    return p, t0, t1, t2

@wp.func
def fetch_pt_xk(ijpt: vec5i, bodies: wp.array(dtype = AffineBody)): 
    I = ijpt[0]
    J = ijpt[1]

    bi = bodies[I]
    bj = bodies[J]
    pid = ijpt[3]
    tid = ijpt[4]

    T = bj.triangles[tid]

    p = bi.xk[pid]
    t0 = bj.xk[T[0]]
    t1 = bj.xk[T[1]]
    t2 = bj.xk[T[2]]
    return p, t0, t1, t2

@wp.func
def fetch_ee(ijee: vec5i, bodies: wp.array(dtype = AffineBody)):
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eiid = ijee[3]
    ejid = ijee[4]

    EI = bi.edges[eiid]
    EJ = bj.edges[ejid]

    ei0 = bi.x[EI[0]]
    ei1 = bi.x[EI[1]]

    ej0 = bj.x[EJ[0]]
    ej1 = bj.x[EJ[1]]

    return ei0, ei1, ej0, ej1

@wp.func
def fetch_ee_xk(ijee: vec5i, bodies: wp.array(dtype = AffineBody)):
    I = ijee[0]
    J = ijee[1]

    bi = bodies[I]
    bj = bodies[J]
    eiid = ijee[3]
    ejid = ijee[4]

    EI = bi.edges[eiid]
    EJ = bj.edges[ejid]

    ei0 = bi.xk[EI[0]]
    ei1 = bi.xk[EI[1]]

    ej0 = bj.xk[EJ[0]]
    ej1 = bj.xk[EJ[1]]

    return ei0, ei1, ej0, ej1

@wp.func
def fetch_vertex(ip: wp.vec2i, bodies: wp.array(dtype = AffineBody)):
    I = ip[0]
    pid = ip[1]

    return bodies[I].x[pid]

@wp.func
def fetch_xk(ip: wp.vec2i, bodies: wp.array(dtype = AffineBody)):
    I = ip[0]
    pid = ip[1]

    return bodies[I].xk[pid]