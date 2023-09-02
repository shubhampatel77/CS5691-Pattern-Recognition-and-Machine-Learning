def trainclass(x):
    c1 = x[:350, :]; c2 = x[350:700, :]; c3 = x[700:1050, :]
    return (c1, c2, c3)
