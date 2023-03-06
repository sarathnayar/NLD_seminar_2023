import matplotlib.pyplot as plt
import numpy as np

def phase_plane_2d(f, args, x_limits, y_limits, samples=25, f_nullclines=None, fps=None, axis=None):
    if axis is None:
        fig, axis = plt.subplots()
    
    x = np.linspace(x_limits[0], x_limits[1], samples)
    y = np.linspace(y_limits[0], y_limits[1], samples)
    X,Y = np.meshgrid(x, y)
    
    DX,DY = f([X,Y], **args)
    nrm = np.sqrt(DX**2 + DY**2)
    
    axis.quiver(X, Y, DX/nrm, DY/nrm, np.log(nrm))
    
    if fps is not None:
        axis.scatter(*fps.T, c='r', s=25)
    
    if f_nullclines is not None:
        # get x nullcline
        x_nullcline, y_nullcline = f_nullclines([x, y], **args)
        axis.plot(x_nullcline, y)
        axis.plot(x, y_nullcline)
    
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_xlabel('x[0]')
    axis.set_ylabel('x[1]')
    axis.set_title(args)
    return axis
        
#test
from models import fitzhugh_nagumo, fitzhugh_nagumo_nullclines
phase_plane_2d(fitzhugh_nagumo, 
               {'t': 0, 'a': 2, 'b': 0.2, 'I': 0, 'eps': 0.01}, 
               x_limits=[-1, 2], y_limits=[-0.5, 1], 
               f_nullclines=fitzhugh_nagumo_nullclines)

def phase_plane_1d(f, args, x_limits, samples=25, fps=None, axis=None):
    if axis is None:
        fig, axis = plt.subplots()
    
    x = np.linspace(x_limits[0], x_limits[1], samples)
    
    dx = f(x, **args)
    nrm = np.abs(dx)
    axis.plot(x, dx)
    axis.quiver(x, np.zeros_like(x), dx/nrm, np.zeros_like(x), np.log(nrm))
    axis.set_xlabel('x')
    axis.set_ylabel('dx')
    axis.set_title(args)
    if fps is not None:
        for fp in fps:
            axis.scatter(fp, 0, c='r', s=25)
            
# test
from models import saddlenode_bifurcation
phase_plane_1d(saddlenode_bifurcation, 
               {'t': 0, 'p': 1.0}, 
               x_limits=[-3, 3])