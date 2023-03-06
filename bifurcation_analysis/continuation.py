from predictors import secant_predictor
from correctors import newton_corrector
from models import saddlenode_bifurcation, supercritical_pitchfork_bifurcation
from parameterizations import local_parameterization
from sympy.abc import x, t, p
from sympy import diff, lambdify
import pandas as pd

def compute_stability(f_jac, x_star, p):
    jac = f_jac(x_star, p)
    if len(x_star) == 1:
        eigvals = np.sign(jac)
    else:
        eigvals, _ = eig(jac)
    stable = all(np.real(eigvals) < 0)
    return eigvals, stable

def branch_tracing(f, f_jac, x_0, p_0, del_z_0, p_limits, predictor_step_length=1, 
                   parameterization_steplength=1):
    df_fixed_points = pd.DataFrame()
    z_preds = []
    zs = []
    # calculate one solution on the branch
    sol = root(f, x_0, jac = f_jac, args=(p_0))
    eigvals, stable = compute_stability(f_jac, sol.x, p_0)
    df_fixed_points = df_fixed_points.append(
        {**{'p': p_0}, 
         **{f'x_{i}': sol.x[i] for i in range(len(sol.x))}, 
         **{f'lambda_{i}': eigvals[i] for i in range(len(eigvals))}, 
         **{'stability': 'stable' if stable else 'unstable'}}, 
        ignore_index=True)
    # compute z
    z_curr = np.ones(len(sol.x)+1)
    z_curr[:len(sol.x)] = sol.x
    z_curr[-1] = p_0
    z_prev = z_curr + del_z_0
    #print('z_curr', z_curr, z_prev)
    zs.append(z_curr)
    p_curr = p_0
    while p_curr > p_limits[0] and p_curr < p_limits[1]:
        # predictor
        z_pred = secant_predictor(z_curr, z_prev, step_length=predictor_step_length)
        #print('z_pred', z_pred)
        # parameterization
        h, h_jac = local_parameterization(f, z_curr, z_prev, step_length=parameterization_steplength)
        # corrector
        sol = newton_corrector(h, z_pred, jac=h_jac)
        z_prev = z_curr
        z_curr = sol.x
            
        if sol.success:
            eigvals, stable = compute_stability(f_jac, sol.x[:-1], sol.x[-1])
            z_preds.append(z_pred)
            
            zs.append(z_curr)
            p_curr = z_curr[-1]
            df_fixed_points = df_fixed_points.append(
                {**{'p': p_curr}, 
                 **{f'x_{i}': sol.x[i] for i in range(len(sol.x)-1)}, 
                 **{f'lambda_{i}': eigvals[i] for i in range(len(eigvals))}, 
                 **{'stability': 'stable' if stable else 'unstable'}}, 
                ignore_index=True)
            #print(z_curr)
        
    return df_fixed_points, zs, z_preds
        
#%% test saddlenode_bifurcation
from sympy.abc import x, t, p
f = saddlenode_bifurcation(x=x, t=0, p=p)
f_jac = diff(f, x)
f = lambdify((x, p), f, 'numpy')
f_jac = lambdify((x, p), f_jac, "numpy")

x_0 = 0
p_0 = 1
del_z_0 = [0.1, 0.1]
p_limits = [-1.01, 1.01]
t_start = perf_counter()
df_fixed_points, zs, z_preds = branch_tracing(f, f_jac, x_0, p_0, del_z_0, p_limits)
t_end = perf_counter()
print(t_end-t_start)
#plot
fig, ax = plt.subplots()
sns.scatterplot(data=df_fixed_points, x='p', y='x_0', hue='stability', 
                ax=ax)
ax.set_title('saddlenode bifurcation')
#zs_zpreds = np.array([[z, z_pred] for z, z_pred in zip(zs, z_preds)]).reshape(-1, 2, order='C')
#zs_zpreds = zs_zpreds[:, [1, 0]]
#ax.scatter(*np.array(z_preds)[:, [1, 0]].T, marker='x', c='k', s=10)


#%% test pitchfork_bifurcation
from sympy.abc import x, t, p
f = supercritical_pitchfork_bifurcation(x=x, t=0, p=p)
f_jac = diff(f, x)
f = lambdify((x, p), f, 'numpy')
f_jac = lambdify((x, p), f_jac, "numpy")

x_0 = 0
p_0 = 1
del_z_0 = [0.02, 0.02]
p_limits = [-1.01, 1.01]
t_start = perf_counter()
df_fixed_points, zs, z_preds = branch_tracing(f, f_jac, x_0, p_0, del_z_0, p_limits)
t_end = perf_counter()
print(t_end-t_start)
#plot
fig, ax = plt.subplots()
sns.scatterplot(data=df_fixed_points, x='p', y='x_0', hue='stability', 
                ax=ax)
ax.set_title('supercritical pitchfork bifurcation')

#%% fitzhugh-nagumo model
from sympy.abc import x, t, p
x = Matrix(MatrixSymbol('x', 2, 1))
f = Matrix(fitzhugh_nagumo(x=x, t=0, a=p, b=0.2, I=0, eps=0.01))
f_jac = f.jacobian(x)
f_lambda = lambdify((x, p), f, 'numpy')
f = lambda x, p: np.squeeze(f_lambda(x, p))
f_jac = lambdify((x, p), f_jac, "numpy")

x_0 = [0.8, 0.8]
p_0 = 10
del_z_0 = [0.1, 0.1, 0.1]
p_limits = [0.99, 10.01]
t_start = perf_counter()
df_fixed_points, zs, z_preds = branch_tracing(f, f_jac, x_0, p_0, del_z_0, p_limits)
t_end = perf_counter()
print(t_end-t_start)
#plot
fig, ax = plt.subplots()
sns.scatterplot(data=df_fixed_points, x='p', y='x_0', hue='stability', 
                ax=ax)
ax.set_title('fitzhugh nagumo model')
# ax.remove()
# ax=fig.add_subplot(1, 1, 1,projection='3d')
# ax.scatter(xs=df_fixed_points['x_0'], ys=df_fixed_points['x_1'], zs=df_fixed_points['p'], c=df_fixed_points.stability=='stable')
#ax.set_xlabel('x[0]')
ax.set_ylabel('x[0]')
ax.set_xlabel('p')
zs_zpreds = np.array([[z, z_pred] for z, z_pred in zip(zs, z_preds)]).reshape(-1, 3, order='C')
zs_zpreds = zs_zpreds[:, [2, 0]]
ax.plot(*np.array(zs_zpreds).T, c='k')