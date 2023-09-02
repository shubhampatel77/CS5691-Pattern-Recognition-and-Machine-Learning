def devclass(x):
    c1 = x[:100, :]; c2 = x[100:200, :]; c3 = x[200:300, :]
    return (c1, c2, c3)
