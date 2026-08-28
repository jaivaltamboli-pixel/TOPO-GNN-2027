import numpy as np
from scipy.stats import norm

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + (z**2) / n
    centre = (p_hat + (z**2) / (2 * n)) / denom
    spread = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)
