def gaussian(x, μ, Σ):
    d = len(μ[0])
    exponent = np.exp(-0.5*(x-μ)@np.linalg.inv(Σ)@(x-μ).T )
    det = 1/(np.linalg.det(Σ))**0.5
    gdf = 1/(2*np.pi)**(d/2)*det*exponent
    return float(gdf[0])
