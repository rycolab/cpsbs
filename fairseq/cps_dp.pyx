cimport cython
import numpy as np

cimport numpy as np

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def log1pexp(DTYPE_t x):
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).
    -log1pexp(-x) is log(sigmoid(x))
    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37:
        return np.exp(x)
    elif -37 <= x <= 18:
        return np.log1p(np.exp(x))
    elif 18 < x <= 33.3:
        return x + np.exp(-x)
    else:
        return x


@cython.boundscheck(False)
@cython.wraparound(False)
def log_add(DTYPE_t x, DTYPE_t y):
    """
    Addition of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
    cdef DTYPE_t d
    cdef DTYPE_t r

    if x == -np.inf:
        return y
    elif y == -np.inf:
        return x
    else:
        if y <= x:
            d = y-x
            r = x
        else:
            d = x-y
            r = y
        return r + log1pexp(d)


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_normalization(np.ndarray[DTYPE_t, ndim=1] logp, int k):
    cdef int n = len(logp)
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs

    subset_sum_product_probs = np.full((k + 1, n + 1), -np.inf, dtype=np.float64)
    subset_sum_product_probs[0, :] = 0.
    cdef DTYPE_t intermediate_res

    cdef int r
    cdef int i

    for r in range(1, k + 1):
        for i in range(1, n + 1):
            intermediate_res = subset_sum_product_probs[r - 1, i - 1] + logp[i - 1]
            subset_sum_product_probs[r, i] = log_add(intermediate_res, subset_sum_product_probs[r, i - 1])
    return subset_sum_product_probs


@cython.boundscheck(False)
@cython.wraparound(False)
def sample(np.ndarray[DTYPE_t, ndim=2] logp, int k, int bsz):
    cdef int n = logp.shape[1]
    k = min(n, k)

    cdef np.ndarray[np.int_t, ndim=2] samples_idx
    samples_idx = np.zeros((bsz, k), dtype=np.int)

    cdef np.ndarray[DTYPE_t, ndim=2] thresholds = np.log(np.random.uniform(size=(bsz, n)))

    cdef int j
    cdef int i
    cdef int to_pick_number
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs
    cdef DTYPE_t thresh

    for j in range(bsz):
        to_pick_number = k
        subset_sum_product_probs = calc_normalization(logp[j, :], k)
        for i in range(n, 0, -1):
            thresh = logp[j, i - 1] + subset_sum_product_probs[to_pick_number - 1, i - 1] - subset_sum_product_probs[to_pick_number, i]
            if thresholds[j, i - 1] < thresh:
                samples_idx[j, k - to_pick_number - 1] = (i - 1)
                to_pick_number -= 1
                if to_pick_number == 0:
                    break
    return samples_idx