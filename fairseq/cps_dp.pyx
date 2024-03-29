import numpy as np
from libc.math cimport NAN

cimport cython
from cython.parallel import prange

cimport numpy as np
from libc.math cimport exp, log1p, log, expm1, abs


cdef extern from "math.h":
    float INFINITY

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_int_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log1mexp(DTYPE_t x):
    """
    Numerically stable implementation of log(1-exp(x))
    Note: function is finite for x < 0.
    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    cdef DTYPE_t a
    if x >= 0:
        return NAN
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return log(-expm1(-a))
        else:
            return log1p(-exp(-a))

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
            return x + log1pexp(y - x)
        else:
            return y + log1pexp(x - y)

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_normalization(np.ndarray[DTYPE_t, ndim=1] logp_sliced, int k):
    """
    This function calculates the normalization factor in CPS which is
    sum of product of weights of all sets that have size k
    @param logp_sliced: weights of candidates in log space
    @param k: sample size
    @return: dp matrix containing all normalization factors
    """
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs

    subset_sum_product_probs = np.full((k + 1, n + 1), -np.inf, dtype=np.float64)
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
def calc_log_inclusion_probs(np.ndarray[DTYPE_t, ndim=1] logp_sliced,
                             np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs, int k):
    """
    This function calculates the inclusion probability for CPS design
    operates in log space
    @param logp_sliced: weights of candidates which can be probabilities or odds
    @param subset_sum_product_probs: normalization factors
    @param k:sample size
    @return: log inclusion probabilities
    """
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=1] dp = np.full(n, -np.inf, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs

    cdef np.ndarray[DTYPE_t, ndim=2] remaining_subsetsum_product_probs = np.full((k + 2, n + 2), -np.inf,
                                                                                 dtype=np.float64)
    remaining_subsetsum_product_probs[k, :] = 0.

    cdef int r
    cdef int i
    for r in range(k, 0, -1):
        for i in prange(n, 0, -1, nogil=True):
            dp[i - 1] = log_add(dp[i - 1],
                                subset_sum_product_probs[r - 1, i - 1] + remaining_subsetsum_product_probs[r, i + 1])
            remaining_subsetsum_product_probs[r, i] = log_add(
                remaining_subsetsum_product_probs[r + 1, i + 1] + logp_sliced[i - 1],
                remaining_subsetsum_product_probs[r, i + 1])

    log_inclusion_probs = logp_sliced + dp - subset_sum_product_probs[k, n]
    return log_inclusion_probs

@cython.boundscheck(False)
@cython.wraparound(False)
def sample(np.ndarray[DTYPE_t, ndim=1] logp, np.ndarray[DTYPE_int_t, ndim=1] selected_inds, int k):
    """
    This function picks a sample of size k from candidates
    @param logp: log probability of candidates
    @param selected_inds: selected candidates after nucleus filtering
    @param k: sample size
    @return: selected candidates indices and their inclusion probabilities
    """
    cdef long n = len(logp)
    k = min(n, k)

    cdef list samples_idx = []
    cdef list selected_incs = []
    cdef np.ndarray[DTYPE_t, ndim=1] thresholds = np.log(np.random.uniform(size=n))

    cdef np.ndarray[DTYPE_t, ndim=1] log_weights # using odds approximation as weights
    cdef np.ndarray[DTYPE_t, ndim=1] log_prob_filtered
    log_prob_filtered = logp.copy()
    log_prob_filtered[log_prob_filtered > 0.99] = 0.99 # clipping in order to prevent NAN generation
    log_weights = log_prob_filtered - np.array(list(map(log1mexp, log_prob_filtered)))

    cdef long i
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs
    cdef DTYPE_t thresh
    cdef int to_pick_number
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs

    to_pick_number = k
    subset_sum_product_probs = calc_normalization(log_weights, k)
    log_inclusion_probs = calc_log_inclusion_probs(log_weights, subset_sum_product_probs, k)
    for i in range(n, 0, -1):
        thresh = log_weights[i - 1] + subset_sum_product_probs[to_pick_number - 1, i - 1] - subset_sum_product_probs[
            to_pick_number, i]
        if thresholds[i - 1] < thresh:
            samples_idx.append(selected_inds[i - 1])
            selected_incs.append(log_inclusion_probs[i - 1])
            to_pick_number -= 1
            if to_pick_number == 0:
                break
    return np.asarray(samples_idx), np.asarray(selected_incs)
