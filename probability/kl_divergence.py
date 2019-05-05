######### KL Divergence Implementation #########
import numpy as np
from scipy import stats

def KL(P,Q):
    """
    Implements the KL divergence in between P and Q,
    Please note that P and Q should be probability distributions, i.e. add up to 1
    """
    epsilon = 1e-9 # to avoid numerical undef/over-flows
    P = np.asarray(P, dtype=np.float)
    Q = np.asarray(Q, dtype=np.float)
    
    kl = sum(P * np.log( (P+epsilon) / (Q+epsilon)))
    return kl


if __name__ == "__main__":
    p = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.64]
    q= [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.928]
    assert ((KL(p,q) - stats.entropy(p,q)) < 1e-10)
    print("Test passed!")
