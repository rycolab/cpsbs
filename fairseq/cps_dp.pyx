import numpy as np

cimport cython
from cython.parallel import prange

cimport numpy as np
from libc.math cimport exp, log1p


cdef extern from "math.h":
    float INFINITY

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log1pexp(DTYPE_t x) nogil:
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).
    -log1pexp(-x) is log(sigmoid(x))
    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37.:
        return exp(x)
    elif -37. <= x <= 18.:
        return log1p(exp(x))
    elif 18. < x <= 33.3:
        return x + exp(-x)
    else:
        return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log_add(DTYPE_t x, DTYPE_t y) nogil:
    """
    Addition of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
    if x == -INFINITY:
        return y
    elif y == -INFINITY:
        return x
    else:
        if y <= x:
            return x + log1pexp(y-x)
        else:
            return y + log1pexp(x-y)


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_normalization(np.ndarray[DTYPE_t, ndim=1] logp_sliced, int k):
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs

    subset_sum_product_probs = np.full((k + 1, n + 1), -INFINITY, dtype=np.float64)
    subset_sum_product_probs[0, :] = 0.
    cdef float intermediate_res
    cdef int r
    cdef int i

    for r in range(1, k + 1):
        for i in prange(1, n + 1, nogil=True):
            intermediate_res = subset_sum_product_probs[r - 1, i - 1] + logp_sliced[i - 1]
            subset_sum_product_probs[r, i] = log_add(subset_sum_product_probs[r, i - 1], intermediate_res)
    return subset_sum_product_probs


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_log_inclusion_probs(np.ndarray[DTYPE_t, ndim=1] logp_sliced, np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs, int k):
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=1] dp = np.full(n, -np.inf, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs

    cdef np.ndarray[DTYPE_t, ndim=2] remaining_subsetsum_product_probs = np.full((k + 2, n + 2), -np.inf, dtype=np.float64)
    remaining_subsetsum_product_probs[k, :] = 0.

    cdef int r
    cdef int i
    for r in range(k, 0, -1):
        for i in prange(n, 0, -1, nogil=True):
            dp[i-1] = log_add(dp[i-1], subset_sum_product_probs[r - 1, i - 1] + remaining_subsetsum_product_probs[r, i + 1])
            remaining_subsetsum_product_probs[r, i] = log_add(remaining_subsetsum_product_probs[r + 1, i + 1] + logp_sliced[i-1], remaining_subsetsum_product_probs[r, i + 1])

    log_inclusion_probs = logp_sliced + dp - subset_sum_product_probs[k, n]
    return log_inclusion_probs


@cython.boundscheck(False)
@cython.wraparound(False)
def sample(np.ndarray[DTYPE_t, ndim=1] logp, int k, int bsz):
    cdef long n = len(logp)
    k = min(n, k)

    cdef list samples_idx = []
    cdef np.ndarray[DTYPE_t, ndim=1] thresholds = np.log(np.random.uniform(size= n))

    cdef long i
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs
    cdef DTYPE_t thresh
    cdef int to_pick_number
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs

    to_pick_number = k
    subset_sum_product_probs = calc_normalization(logp, k)
    log_inclusion_probs = calc_log_inclusion_probs(logp, subset_sum_product_probs, k)
    for i in range(n, 0, -1):
        thresh = logp[i - 1] + subset_sum_product_probs[to_pick_number - 1, i - 1] - subset_sum_product_probs[to_pick_number, i]
        if thresholds[i - 1] < thresh:
            samples_idx.append(i - 1)
            to_pick_number -= 1
            if to_pick_number == 0:
                break
    return np.asarray(samples_idx), log_inclusion_probs