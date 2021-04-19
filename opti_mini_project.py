## PART 0: Your Details





## PART A: Importing the necessary libraries
import numpy as np
from scipy import optimize
# =============================================================================
# import compare as cp
# =============================================================================
import time
import plot_ing as pt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from functools import partial
from itertools import zip_longest
from Project import HV
from compare import GPS


from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from autograd import value_and_grad
from scipy.optimize import minimize
from collections import defaultdict


# =============================================================================
# from opti_project_shibani import HV
# =============================================================================
## PART B: Writing the functions
def func(x):
    '''
    =======================================================
    This is the function we wish to differentiate
    =======================================================
    '''
    ## BEGIN YOUR CODE HERE ##
# =============================================================================
#     fx=(1.5+x[1]*x[0]-x[0])**2 + (2.25+(x[1]**2)*x[0]-x[0])**2+(2.625+(x[1]**3)*x[0]-x[0])**2
# =============================================================================
    fx=100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    ##8*x[0]**2+4*x[0]*x[1]+5*x[1]**2

    ## END YOUR CODE HERE ##
    return fx

## PART C: Main Code -- plotiing


##initial guess and max iter            
x0=np.array([2.5,2.5])
x_init=np.vstack(x0)
 
         
##Evaluating    
     



# =============================================================================
# delFC=cp.compare(func,x0,algo="Powell",initial=time.time())
# =============================================================================
# =============================================================================
# pt.plot_ing(delFC)
# =============================================================================




f =lambda x, y: 100*(y-x**2)**2 + (1-x)**2
minima=np.array([1,1])
func(minima)
minima_=minima.reshape(-1,1)
minima_
f(*minima_)


xmin,xmax,xstep=-4.5,4.5,0.2
ymin,ymax,ystep=-4.5,4.5,0.2


x, y=np.meshgrid(np.arange(xmin,xmax+xstep,xstep),np.arange(ymin,ymax+ystep,ystep))
x2=np.array([x,y])
z=func(x2)
# =============================================================================
# path=np.array(delFC[0]).T
# =============================================================================
# =============================================================================
# 
# fig=plt.figure(figsize=(8,5))
# ax=plt.axes(projection='3d',elev=50,azim=-50)
# ax.plot_surface(x,y,z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none',alpha=0.8,cmap=plt.cm.jet)
# ax.plot(minima_[0],minima_[1],func(minima_),'r*',  markersize=50)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$z$')
# plt.grid()
# ax.set_xlim((xmin,xmax))
# ax.set_ylim((ymin,ymax))
# plt.show()
# ## End of Code
# 
# 
# from autograd import elementwise_grad, value_and_grad
# dz_dx=elementwise_grad(f,argnum=0)(x,y)
# dz_dy=elementwise_grad(f,argnum=1)(x,y)
# fig, ax=plt.subplots(figsize=(10,6)) 
# 
# ax.contour(x,y,z, levels=np.logspace(0,5,35), norm=LogNorm(),cmap=plt.cm.jet)
# ax.quiver(x,y,x-dz_dx,y-dz_dy,alpha=0.5)
# ax.plot(minima_[0],minima_[1],'r*',markersize=50)      
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')        
# ax.set_xlim((xmin,xmax))
# ax.set_ylim((ymin,ymax))
# plt.grid()
# plt.show()        
#       
# 
# fig, ax = plt.subplots(figsize=(10, 6))
# 
# ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm())
# ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1],scale_units='xy', angles='xy', scale=1, color='k') # 
# ax.plot(minima_[0],minima_[1], 'r*', markersize=50)
# 
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# plt.grid()
# ax.set_xlim((xmin, xmax))
# ax.set_ylim((ymin, ymax)) 
# 
# 
# 
# 
# fig = plt.figure(figsize=(8, 5))
# ax = plt.axes(projection='3d', elev=50, azim=-50)
# 
# ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
# ax.quiver(path[0,:-1], path[1,:-1], func(path[0,:-1]), 
#           path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], func((path[0,1:]-path[0,:-1])), color='k') 
#          
# ax.plot(minima_[0],minima_[1],func(minima_), 'r*', markersize=20)
# 
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$z$')
# 
# ax.set_xlim((xmin, xmax))
# ax.set_ylim((ymin, ymax))   
# =============================================================================
#PARAmeters 
alfa=1.1
epsilon=1e-6
delta=2
max_iter=20000 

c=HV(func,x_init,max_iter,epsilon,delta,alfa)
print(c[0])
# =============================================================================
# def make_minimize_cb(path=[]):
#     
#     def minimize_cb(xk):
#         # note that we make a deep copy of xk
#         path.append(np.copy(xk))
# 
#     return minimize_cb
# =============================================================================

# =============================================================================
# func = value_and_grad(lambda args: f(*args))
# 
# 
# path_ = [x0]
# res = minimize(func, x0=x0, method='Newton-CG',
#                jac=True, tol=1e-20, callback=make_minimize_cb(path_))
# 
# =============================================================================

v= np.array(c[0]).T
v1=v.shape[2]
path=v.reshape(2,v1)

# =============================================================================
# for i in range(v1):
#     path
# print(path)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.plot(*minima_, 'r*', markersize=50)

line, = ax.plot([], [], 'b', label='Newton-CG', lw=2)
point, = ax.plot([], [], 'bo')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

ax.legend(loc='upper left')











def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    line.set_data(path[::,:i])
    point.set_data(path[::,i-1:i])
    return line, point


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=path.shape[1], interval=60, 
                               repeat_delay=5, blit=True)

anim.save("ani.mp4")


# =============================================================================
# func = value_and_grad(lambda args: f(*args))
# 
# =============================================================================





# =============================================================================
# class TrajectoryAnimation(animation.FuncAnimation):
#     
#     def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, 
#                  interval=60, repeat_delay=5, blit=True, **kwargs):
# 
#         if fig is None:
#             if ax is None:
#                 fig, ax = plt.subplots()
#             else:
#                 fig = ax.get_figure()
#         else:
#             if ax is None:
#                 ax = fig.gca()
# 
#         self.fig = fig
#         self.ax = ax
#         
#         self.paths = paths
# 
#         if frames is None:
#             frames = max(path.shape[1] for path in paths)
#   
#         self.lines = [ax.plot([], [], label=label, lw=2)[0] 
#                       for _, label in zip_longest(paths, labels)]
#         self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
#                        for line in self.lines]
# 
#         super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
#                                                   frames=frames, interval=interval, blit=blit,
#                                                   repeat_delay=repeat_delay, **kwargs)
# 
#     def init_anim(self):
#         for line, point in zip(self.lines, self.points):
#             line.set_data([], [])
#             point.set_data([], [])
#         return self.lines + self.points
# 
#     def animate(self, i):
#         for line, point, path in zip(self.lines, self.points, self.paths):
#             line.set_data(*path[::,:i])
#             point.set_data(*path[::,i-1:i])
#         return self.lines + self.points
# 
# 
# 
# 
# 
# 
# 
# methods = [
#     "CG",
# #   "BFGS",
#     "Newton-CG",
#     "L-BFGS-B",
#     "TNC",
#     "SLSQP",
#     "GPS",
# #   "dogleg",
# #   "trust-ncg"
# ]
# 
# 
# # =============================================================================
# # minimize_ = partial(minimize, fun=func, x0=x0, jac=True, bounds=[(xmin, xmax), (ymin, ymax)], tol=1e-20)
# # =============================================================================
# 
# 
# 
#     
#     
#     
#     
# paths_ = defaultdict(list)
# for method in methods:
#     
#     delF=cp.compare(func,x0,algo=method,initial=time.time())     
#     paths_[method]=np.array(delF[0])    
#     paths = [np.array(paths_[method]).T    
#     
#     
#     
# 
# # =============================================================================
# # zpaths = [f(*path) for path in paths]
# # =============================================================================
# 
# fig, ax = plt.subplots(figsize=(10, 6))
# 
# ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
# ax.plot(*minima_, 'r*', markersize=50)
# 
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# 
# ax.set_xlim((xmin, xmax))
# ax.set_ylim((ymin, ymax))
# 
# anim = TrajectoryAnimation(*paths, labels=methods, ax=ax)
# 
# ax.legend(loc='upper left')
# 
# 
# 
# 
# anim.save("ani2.mp4")
# 
# =============================================================================
