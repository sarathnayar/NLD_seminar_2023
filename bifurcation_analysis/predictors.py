import numpy as np

def secant_predictor(z_curr:np.ndarray, z_prev:np.ndarray=None, step_length:int=1):
    if z_prev is None:
        z_prev = z_curr # trivial predictor
    return z_curr + step_length*(z_curr-z_prev)

def trivial_predictor(z_curr:np.ndarray):
    return z_curr