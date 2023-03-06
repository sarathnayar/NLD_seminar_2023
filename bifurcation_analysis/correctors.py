import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def newton_corrector(f, x0, args=(), jac=None, options={'maxfev': 10000}):
    sol = root(f, x0, args, jac=jac, options=options)
    return sol
#TODO: write newton's method

