import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef align(float[:,:] sig1,float[:,:] sig2):
    cdef int d = sig1.shape[1]
    cdef int len1 = sig1.shape[0]
    cdef int len2 = sig2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] npL = np.empty((len1,len2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] npP = np.empty((len1,len2), dtype=np.float32)
    
    cdef float[:,:] L = npL
    cdef float[:,:] P = npP
    
    cdef float cost,tmp
    cdef int j,k,i
    for j in range(0,len1):
        for k in range(0,len2):
            cost = 0
            for i in range(d):
                tmp = sig1[j,i] - sig2[k,i]
                cost += tmp * tmp
            cost = sqrt(cost)
            
            if j == 0 and k == 0:
                L[j,k] = cost
                P[j,k] = 3
            elif k == 0:
                L[j,k] = cost + L[j-1,k]
                P[j,k] = 2
            elif j == 0:
                L[j,k] = cost + L[j,k-1]
                P[j,k] = 1
            else: # j, k > 0
                if L[j-1,k] < L[j,k-1] and L[j-1,k] < L[j-1,k-1]: # insertion (up)
                    P[j,k] = 1
                    L[j,k] = cost + L[j-1,k]
                elif L[j,k-1] < L[j-1,k-1]: # deletion (left)
                    P[j,k] = 2
                    L[j,k] = cost + L[j,k-1]
                else: # match (up left)
                    P[j,k] = 3
                    L[j,k] = cost + L[j-1,k-1]
    
    return npL,npP

def traceback(float[:,:] sig1,float[:,:] sig2, float[:,:] L):
    sig12 = np.zeros(sig2.shape) # align 1 onto 2
    cdef int j = sig1.shape[0]-1
    cdef int k = sig2.shape[0]-1
    A = []
    C = []
    cdef float cost,tmp
    while True:
        if j == 0 and k == 0:
            A.append((0,0))
            C.append(L[0,0])
            break # got back to the beginning
        
        cost = 0
        for i in range(sig1.shape[1]):
            tmp = sig1[j,i] - sig2[k,i]
            cost += tmp * tmp
        cost = sqrt(cost)
        
        if j>0 and k>0 and L[j,k] == L[j-1,k-1] + cost: # progress
            A.append((j,k))
            C.append(L[j,k])
            j -= 1
            k -= 1
        elif k>0 and L[j,k] == L[j,k-1] + cost: # stay sig2
            A.append((j,k))
            C.append(L[j,k])
            k -= 1
        elif j>0 and L[j,k] == L[j-1,k] + cost: # stay sig1
            A.append((j,k))
            C.append(L[j,k])
            j -= 1
        else: 
            assert False
    
    return list(reversed(A)),list(reversed(C))
