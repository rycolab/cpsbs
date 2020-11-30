import numpy as np

cimport numpy as np


def log1pexp(np.float_t x):
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


def log_add(np.float_t x, np.float_t y):
    """
    Addition of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
    cdef np.float_t d
    cdef np.float_t r

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


def calc_normalization(np.ndarray logp, int k):
    cdef int n = len(logp)

    cdef np.ndarray subset_sum_product_probs = np.full((k + 1, n + 1), -np.inf)
    subset_sum_product_probs[0, :] = 0.
    cdef np.float_t intermediate_res

    for r in range(1, k + 1):
        for i in range(1, n + 1):
            intermediate_res = subset_sum_product_probs[r - 1, i - 1] + logp[i - 1]
            subset_sum_product_probs[r, i] = log_add(intermediate_res, subset_sum_product_probs[r, i - 1])
    return subset_sum_product_probs