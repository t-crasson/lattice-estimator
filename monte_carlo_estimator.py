import numpy as np
#from estimator import *
from typing import Optional


def sample_sparse_binary_indices(N: int, h: int, batch_size: int = 1) -> np.ndarray:
    """Sample the `h` non-zero indices (sorted) of a sparse binary key in R_N.

    :param N: ring dimension.
    :param h: Hamming weight.
    :param batch_size: number of keys to sample.
    :returns: sorted index vectors of length `h`, with values in [0, N).
    """
    return np.stack(
        [np.sort(np.random.choice(N, size=h, replace=False)) for _ in range(batch_size)]
    )


def diff_of_indices(idx: np.ndarray) -> np.ndarray:
    """Compute `diff(s)` as defined in Section 4 of the paper.

    d[0]    = -idx[0]
    d[i]    = idx[i-1] - idx[i]   for 1 <= i <= h-1
    d[h]    = -idx[h-1]

    Note: we return |d[i]| for 1 <= i <= h-1, as these are the only values
    constrained by B during rejection sampling.

    :param idx: sorted non-zero indices.
    :returns: vector of internal |d[i]| values, of length h-1.
    """
    # |idx[i] - idx[i-1]| for i = 1..h-1
    return np.abs(np.diff(idx))


def estimate_delta_montecarlo(
    N: int,
    h: int,
    B: int,
    n_samples: int = 1000000,
    seed: Optional[int] = 0,
) -> float:
    """Estimate δ = -log2(Pr[a random sparse key passes rejection sampling]).

    Reproduces the procedure from Appendix C: for each key sampled uniformly
    among sparse binary keys of R_N with Hamming weight `h`, the key is accepted
    if and only if max_i |d[i]| < B. The acceptance probability is equal to
    2^{-δ}.

    :param N: ring dimension.
    :param h: Hamming weight.
    :param B: upper bound on |d[i]|.
    :param n_samples: number of keys to simulate (≥ 10⁴ recommended, 10⁵ used in the paper).
    :param seed: random seed for reproducibility.
    :returns: δ ∈ [0, +∞). Returns +∞ if no key is accepted.
    """
    np.random.seed(seed)
    indexes = sample_sparse_binary_indices(N, h, batch_size=n_samples)
    diffs = diff_of_indices(indexes)
    p = np.mean(np.max(diffs, axis=1) < B)
    return -np.log2(p)



