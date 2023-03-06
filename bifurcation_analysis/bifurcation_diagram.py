import numpy as np
from scipy.optimize import root
from scipy.linalg import eig
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plot_utils import phase_plane_1d, phase_plane_2d
from models import fitzhugh_nagumo, saddlenode_bifurcation, saddlenodelike, supercritical_pitchfork_bifurcation
from sympy.utilities import lambdify
from sympy import MatrixSymbol, Matrix
from sympy import diff
from time import perf_counter

def create_bifurcation_curve(f, f_jac, x_0s, p_0s):
    df_fixed_points = pd.DataFrame()
    
    # Iterate through phase space. No need for vectorization, the root finder does the hard work
    for p in p_0s:
        fps = []
        # Iterate through our sample mesh
        for x_0 in x_0s:
            
            # Use current sample point and search for a solution
            sol = root(f, x_0, jac = f_jac, args=(0,p))
            
            if sol.success:
                
                # Calculate jacobian at root and calculate stability
                jac = f_jac(sol.x, 0, p)
                if len(sol.x) == 1:
                    eigval = np.sign(jac)
                else:
                    eigval, _ = eig(jac)
                stable = all(np.real(eigval) < 0)
                
                fp = list(np.around(sol.x, 6))
                if fp not in fps:
                    fps.append(fp)
                else:
                    continue
                
                fixed_point = {f'x_{i}': fp[i] for i in range(len(fp))}
                eigs = {f'lambda_{i}': np.real(eigval)[i] for i in range(len(eigval))}
                stability = {'stability': 'stable' if stable else 'unstable'}
                
                df_fixed_points = df_fixed_points.append(
                    {**{'p': p}, **fixed_point, **eigs, **stability}, ignore_index=True)
                
                
    return df_fixed_points

#%% test on saddlenode bifurcation model
from sympy.abc import x, t, p
f = saddlenode_bifurcation(x=x, t=0, p=p)
f_jac = diff(f, x)
f = lambdify((x, t, p), f, 'numpy')
f_jac = lambdify((x, t, p), f_jac, "numpy")

x_0s = [0]#np.linspace(-1, 1, 4)
p_0s = np.linspace(-1, 1, 101)
t_start = perf_counter()
df_fixed_points = create_bifurcation_curve(f, f_jac, x_0s, p_0s)
t_end = perf_counter()
print(t_end-t_start)

# plot bifurcation curbe
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for p in p_0s:
    ax[0].cla()
    ax[1].cla()
    sns.scatterplot(data=df_fixed_points, x='p', y='x_0', hue='stability', 
                    ax=ax[0])
    ax[0].set_xlim(p_0s.min(), p_0s.max())
    ax[0].set_ylim(x_0s.min(), x_0s.max())
    ax[0].axvline(p)
    p_fps = df_fixed_points[df_fixed_points.p==p]['x_0'].values
    phase_plane_1d(f, {'t':0, 'p':np.around(p, 3)}, x_limits=[x_0s.min(), x_0s.max()], fps=p_fps, axis=ax[1])
    plt.pause(0.1)
    plt.show()
    
#%% test n pitchfork modl
from sympy.abc import x, t, p
f = supercritical_pitchfork_bifurcation(x=x, t=0, p=p)
f_jac = diff(f, x)
f = lambdify((x, t, p), f, 'numpy')
f_jac = lambdify((x, t, p), f_jac, "numpy")

x_0s = np.linspace(-1, 1, 4)
p_0s = np.linspace(-1, 1, 101)
t_start = perf_counter()
df_fixed_points = create_bifurcation_curve(f, f_jac, x_0s, p_0s)
t_end = perf_counter()
print(t_end-t_start)

# plot bifurcation curbe
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for p in p_0s:
    ax[0].cla()
    ax[1].cla()
    sns.scatterplot(data=df_fixed_points, x='p', y='x_0', hue='stability', 
                    ax=ax[0])
    ax[0].set_xlim(p_0s.min(), p_0s.max())
    ax[0].set_ylim(x_0s.min(), x_0s.max())
    ax[0].axvline(p)
    p_fps = df_fixed_points[df_fixed_points.p==p]['x_0'].values
    phase_plane_1d(f, {'t':0, 'p':np.around(p, 3)}, x_limits=[x_0s.min(), x_0s.max()], fps=p_fps, axis=ax[1])
    plt.pause(0.1)
    plt.show()
    
#%% test n saddlenodelike modl 2D
from sympy.abc import x, t, p
x = Matrix(MatrixSymbol('x', 2, 1))
f = Matrix(saddlenodelike(x=x, t=0, p=p))
f_jac = f.jacobian(x)
f = saddlenodelike
f_jac = lambdify((x, t, p), f_jac, "numpy")

# Generate a mesh of initial conditions in phase space
x = np.linspace(-8, 8, 4)
y = np.linspace(-8, 8, 4)
X,Y = np.meshgrid(x, y)
x_0s = np.array(
    [[x_curr, y_curr] for x_curr, y_curr in np.nditer([X,Y])])
p_0s = np.linspace(-3,3,100)
df_fixed_points = create_bifurcation_curve(f, f_jac, x_0s, p_0s)

# plot bifurcation curbe
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].remove()
ax[0]=fig.add_subplot(1, 2, 1,projection='3d')
for p in p_0s:
    ax[0].cla()
    ax[1].cla()
    fps = df_fixed_points[['p', 'x_0', 'x_1']].values
    stable = df_fixed_points['stability'].values=='stable'
    ax[0].scatter(xs=fps[:, 1], ys=fps[:, 2], zs=fps[:, 0], c=stable)
    ax[0].set_xlim(x_0s[:, 0].min(), x_0s[:, 0].max())
    ax[0].set_ylim(x_0s[:, 1].min(), x_0s[:, 1].max())
    ax[0].plot_surface(X, Y, np.ones_like(X)*p, alpha=0.2)
    ax[0].set_xlabel('x[0]')
    ax[0].set_ylabel('x[1]')
    ax[0].set_zlabel('p')
    p_fps = df_fixed_points[df_fixed_points.p==p][['x_0', 'x_1']].values
    phase_plane_2d(f, {'t':0, 'p':np.around(p, 3)}, 
                    x_limits=[x_0s[:, 0].min(), x_0s[:, 0].max()], 
                    y_limits=[x_0s[:, 1].min(), x_0s[:, 1].max()], 
                    fps=p_fps, axis=ax[1])
    plt.pause(0.1)
    plt.show()

#%% test on fitzhugh nagumo model
from sympy.abc import x, t, p
x = Matrix(MatrixSymbol('x', 2, 1))
f = Matrix(fitzhugh_nagumo(x=x, t=0, a=p, b=0.2, I=0, eps=0.01))
f_jac = f.jacobian(x)
f_lambda = lambdify((x, t, p), f, 'numpy')
f = lambda x, t, p: np.squeeze(f_lambda(x, t, p))
f_jac = lambdify((x, t, p), f_jac, "numpy")

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X,Y = np.meshgrid(x, y)
x_0s = np.array(
    [[x_curr, y_curr] for x_curr, y_curr in np.nditer([X,Y])])
p_0s = np.linspace(1, 10, 100)
df_fixed_points = create_bifurcation_curve(f, f_jac, x_0s, p_0s)

#plot
fig, ax = plt.subplots()
sns.scatterplot(data=df_fixed_points, x='p', y='x_1', hue='stability', 
                ax=ax)
ax.set_title('fitzhugh nagumo model')

# plot bifurcation curbe
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].remove()
ax[0]=fig.add_subplot(1, 2, 1,projection='3d')
for p in p_0s:
    ax[0].cla()
    ax[1].cla()
    fps = df_fixed_points[['p', 'x_0', 'x_1']].values
    stable = df_fixed_points['stability'].values=='stable'
    ax[0].scatter(xs=fps[:, 1], ys=fps[:, 2], zs=fps[:, 0], c=stable)
    ax[0].set_xlim(x_0s[:, 0].min(), x_0s[:, 0].max())
    ax[0].set_ylim(x_0s[:, 1].min(), x_0s[:, 1].max())
    ax[0].plot_surface(X, Y, np.ones_like(X)*p, alpha=0.2)
    ax[0].set_xlabel('x[0]')
    ax[0].set_ylabel('x[1]')
    ax[0].set_zlabel('p')
    p_fps = df_fixed_points[df_fixed_points.p==p][['x_0', 'x_1']].values
    phase_plane_2d(f, {'t':0, 'p':np.around(p, 3)}, 
                    x_limits=[x_0s[:, 0].min(), x_0s[:, 0].max()], 
                    y_limits=[x_0s[:, 1].min(), x_0s[:, 1].max()], 
                    fps=p_fps, axis=ax[1])
    plt.pause(0.1)
    plt.show()