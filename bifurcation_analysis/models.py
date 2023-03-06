def fitzhugh_nagumo(x, t, a, b, I, eps):
    f1 = a*x[0]*(1-x[0])*(x[0]-b)-x[1]+I
    f2 = eps*(x[0]-x[1])
    return [f1, f2]

def fitzhugh_nagumo_nullclines(x, t, a, b, I, eps):
    f2_nullcline = a*x[0]*(1-x[0])*(x[0]-b)+I
    f1_nullcline = x[1]
    return [f1_nullcline, f2_nullcline]

def saddlenodelike(x,t,p):
    dxdt = - x[1] + p * x[0] + x[0]*x[1]*x[1]
    dydt = x[0] + p * x[1] - x[0]*x[0]
    return [dxdt, dydt]

def saddlenode_bifurcation(x, t, p):
    dx = x**2 - p
    return dx

def supercritical_pitchfork_bifurcation(x, t, p):
    dx = p*x - x**3
    return dx