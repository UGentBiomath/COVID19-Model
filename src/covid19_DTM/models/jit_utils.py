import numpy as np
from numba import jit

@jit(nopython=True)
def jit_matmul_2D_1D(A, b):
    """ (n,m) x (m,) --> (n,)"""
    n = A.shape[0]
    m = A.shape[1]
    out = np.zeros(n, np.float64)
    for i in range(n):
        for k in range(m):
            out[i] += A[i, k] * b[k]
    return out

@jit(nopython=True)
def jit_matmul_2D_2D(A, B):
    """A simple jitted implementation of 2Dx2D matrix multiplication
    """
    n = A.shape[0]
    f = A.shape[1]
    m = B.shape[1]
    out = np.zeros((n,m), np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(f):
                out[i, j] += A[i, k] * B[k, j]
    return out

@jit(nopython=True)
def jit_matmul_3D_2D(A, B):
    """(n,k,m) x (n,m) --> for n: (k,m) x (m,) --> (n,k) """
    out = np.zeros(B.shape, np.float64)
    for idx in range(A.shape[0]):
        A_acc = A[idx,:,:]
        b = B[idx,:]
        n = A_acc.shape[0]
        f = A_acc.shape[1]
        for i in range(n):
                for k in range(f):
                    out[idx, i] += A_acc[i, k] * b[k]
    return out

@jit(nopython=True)
def jit_matmul_klm_m(A,b):
    out = np.zeros((A.shape[:-1]), np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                out[i,j] += A[i,j,k]*b[k]
    return out

@jit(nopython=True)
def jit_matmul_klmn_n(A,b):
    out = np.zeros((A.shape[:-1]), np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    out[i,j,k] += A[i,j,k,l]*b[l]
    return out

@jit(nopython=True)
def matmul_q_2D(A,B):
    """ A simple jitted implementation to multiply a 2D matrix of size (n,m) with a 3D matrix (m,k,q)
        Implemented as q times the matrix multiplication (n,m) x (m,k)
        Output of size (n,k,q)
    """
    out = np.zeros((A.shape[0],B.shape[1],B.shape[2]), np.float64)
    for q in range(B.shape[2]):
        b = B[:,:,q]
        n = A.shape[0]
        f = A.shape[1]
        m = b.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(f):
                    out[i, j, q] += A[i, k] * b[k, j]
    return out

@jit(nopython=True)
def redistribute_infections(S, I, G):
    """ 
    A jitted function to redistribute the number of drawn infections happening on the visited patch across the residency patches

    Inputs
    ------

    S : np.ndarray (11,10,4)
        Susceptibles

    I : np.ndarray (11,10,4)
        Number of infections happening on the visited patch

    G : np.ndarray (11,11)
        Origin-destination matrix
    
    Outputs
    -------

    I_star : np.ndarray (11,10,4)
        Number of infections happening on visited patch, redistributed across residencies
    """

    # Compute ratio susceptibles on home over visited patch
    div = S / matmul_q_2D(np.transpose(G), S)
    # Compute fraction of susceptibles in spatial patch i coming from spatial patch j
    out = np.zeros(S.shape)
    # age
    for j in range(S.shape[1]):
        # vaccine
        for k in range(S.shape[2]):
            # Construct matrix
            mat = G * div[:,j,k][:, np.newaxis]
            # Matrix multiply
            res = mat @ I[:,j,k]
            # Assing to output
            out[:,j,k] = res
    return out
