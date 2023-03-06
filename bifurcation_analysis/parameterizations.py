import numpy as np
from sympy import MatrixSymbol, Matrix, symbols
from sympy.utilities import lambdify
from sympy.abc import x, p

'''def local_parameterization(sym_f, sym_x, sym_p, z_curr, z_prev, step_length=1):
    k = np.argmax(np.abs(z_curr))
    eta = z_curr[k]+step_length*(z_curr[k]-z_curr[k-1])
    # create z
    sym_z = MatrixSymbol('z', len(sym_x)+1, 1)
    sym_z = Matrix(sym_z)
    for i in range(len(sym_x)):
        sym_z[i]=sym_x[i]
    sym_z[len(sym_z)] = p
    # create h
    sym_h = list(sym_f)
    sym_h.append(z[k]-eta)
    sym_h = Matrix(sym_h)
    sym_h_jac = sym_h.jacobian(z)
    # convert to lamda functions
    h = lambdify(sym_h)
    h_jac = lambdify(sym_h_jac)
    return h, h_jac'''

def local_parameterization(f, z_curr, z_prev=None, step_length=1):
    k = np.argmax(np.abs(z_curr))
    if z_prev is None:
        z_prev = z_curr
    eta = z_curr[k]+step_length*(z_curr[k]-z_prev[k])
    # create z
    z = MatrixSymbol('z', len(z_curr), 1)
    z = Matrix(z)
    # for i in range(len(z)-1):
    #     z[i]=x[i] if len(z)>2 else x
    # z[len(z)-1] = p
    # create h
    if len(z_curr)>2:
        sym_f = Matrix(f(z[:-1], z[-1]))
        sym_h = list(sym_f)
    else:
        sym_f = f(z[0], z[1])
        sym_h = [sym_f]
    sym_h.append(z[k]-eta)
    sym_h = Matrix(sym_h)
    sym_h_jac = sym_h.jacobian(z)
    # convert to lamda functions
    h_lambda = lambdify([[z[i] for i in range(len(z))]], sym_h, 'numpy')
    h = lambda z: np.squeeze(h_lambda(z))
    h_jac_lambda = lambdify([[z[i] for i in range(len(z))]], sym_h_jac, 'numpy')
    h_jac = lambda z: h_jac_lambda(z)
    return h, h_jac