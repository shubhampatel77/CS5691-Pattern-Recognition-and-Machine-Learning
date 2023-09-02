def stats(x):
    μ = np.array([np.mean(x[:,0]), np.mean(x[:,1])])
    μ = np.reshape(μ, (len(μ),1)).T
    Σ = np.cov(x, rowvar=False)
    return (μ, Σ)
