import numpy as np

# calculate mean and covariance of an array
def stats(x):
    μ = x.mean(0)
    Σ = np.cov(x, rowvar=False)
    return (μ, Σ)