## PART 0: Your Details





## PART A: Importing the necessary libraries
import numpy as np
from scipy import optimize
import compare as cp
import time
import plot_ing as pt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from functools import partial
from itertools import zip_longest
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from autograd import value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from benchmark_func import F1,F2
import pylab


def comparision_animation(ackley):



    ##initial guess and minima           
    x0=np.array([3.2,3.5])
    
    minima=np.array([1,1])
    ackley(minima)
    minima_=minima.reshape(-1,1)
    print(minima_)
    ackley(minima_)
    
    
    ##mesh generation for contour plot
# =============================================================================
#     xmin,xmax,xstep=-5,5,0.5
#     ymin,ymax,ystep=-4,21,0.5
# =============================================================================
    xmin,xmax,xstep=-4.5,4.5,0.2
    ymin,ymax,ystep=-4.5,4.5,0.2
    
    x, y=np.meshgrid(np.arange(xmin,xmax+xstep,xstep),np.arange(ymin,ymax+ystep,ystep))
    x2=np.array([x,y])
    z=ackley(x2)
    
    
    
    
# =============================================================================
#     fig = plt.figure(figsize=(8, 5),dpi=500)
#     ax = plt.axes(projection='3d', elev=50, azim=-50)
#     
#     ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, 
#                     edgecolor='none', alpha=.8, cmap=plt.cm.jet)
#     ax.plot(*minima_, f(*minima_), 'r*', markersize=10)
#     
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$')
#     ax.set_zlabel('$z$')
#     
#     ax.set_xlim((xmin, xmax))
#     ax.set_ylim((ymin, ymax))
#     
#     plt.show()
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    # =============================================================================
    # func = value_and_grad(lambda args: f(*args))
    # =============================================================================
    
    
    
    
    
    
    class TrajectoryAnimation(animation.FuncAnimation):
        
        def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, 
                     interval=200, repeat_delay=15, blit=False, **kwargs):
    
            if fig is None:
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()
            else:
                if ax is None:
                    ax = fig.gca()
    
            self.fig = fig
            self.ax = ax
            
            self.paths = paths
    
            if frames is None:
                frames = max(path.shape[1] for path in paths)
      
            self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                          for _, label in zip_longest(paths, labels)]
            self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                           for line in self.lines]
    
            super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                      frames=frames, interval=interval, blit=blit,
                                                      repeat_delay=repeat_delay, **kwargs)
    
        def init_anim(self):
            for line, point in zip(self.lines, self.points):
                line.set_data([], [])
                point.set_data([], [])
            return self.lines + self.points
    
        def animate(self, i):
            for line, point, path in zip(self.lines, self.points, self.paths):
                line.set_data(*path[::,:i])
                point.set_data(*path[::,i-1:i])
            return self.lines + self.points
    
    
    
    
    
    
    methods = [
        
        "BFGS",
        
        "Powell",
        
# =============================================================================
#         "Nelder-Mead",
#         "DE",
#         "basinhopping",
#         "dual_annealing",
#         "HV",
# =============================================================================
        "HV_modified",
        "SDG",
# =============================================================================
#        

# =============================================================================
        
        
    ]
    
    # =============================================================================
    # methods = [
    #     "CG",
    # #   "BFGS",
    #     "Newton-CG",
    # # =============================================================================
    # #     "L-BFGS-B",
    # # =============================================================================
    #     "TNC",
    #     "SLSQP",
    # #   "dogleg",
    # #   "trust-ncg"
    # ]
    # =============================================================================
    
    # =============================================================================
    # def make_minimize_cb(path=[]):
    #     
    #     def minimize_cb(xk):
    #         # note that we make a deep copy of xk
    #         path.append(np.copy(xk))
    # 
    #     return minimize_cb
    # minimize_ = partial(minimize, fun=func, x0=x0, jac=True, bounds=[(xmin, xmax), (ymin, ymax)], tol=1e-20)
    # =============================================================================
    
    
    
        
        
        
        
    paths_ = defaultdict(list)
    for method in methods:
        
        delF=cp.compare(ackley,x0,algo=method,initial=time.time())  
        paths_[method]=np.array(delF[0]).T
        print(paths_[method].shape)
        paths = [paths_[method] for method in methods]        
        
        
    zpaths = [ackley(path) for path in paths]     
        
    
    origin='lower'
    levels = np.array([1,4,6,8,10,12])
# =============================================================================
#     levels=np.logspace(0,5,7 )
# =============================================================================
    fig, ax = plt.subplots(figsize=(10, 6),constrained_layout=True)
    p=ax.contour(x, y, z, levels=levels, colors='black', linestyles='dashed', linewidths=1, origin=origin)
    plt.clabel(p, inline=1,fmt='%1.1f' , fontsize=15 )
    p=ax.contourf(x, y, z, alpha=0.85,cmap='RdGy')  
    ax.plot(*minima_, 'r*', markersize=20)
# =============================================================================
#     ax.clabel( inline=1, fontsize=12)
# =============================================================================
    labelsize_=15
    ax.tick_params(axis='both', which='major', labelsize=labelsize_) 
    ax.set_xlabel(r'$x$', fontdict=None, labelpad=2, fontsize=25)
    ax.set_ylabel(r'$y$', fontdict=None, labelpad=-16, fontsize=25)
    
# =============================================================================
#     ax.set_xticks([-15,-10,-5,0,5]) 
#     ax.set_xticklabels(["-15","-10","-5","0","5"], fontsize=labelsize_)
#     ax.set_yticks([-15,-10,-5,0,5]) 
#     ax.set_yticklabels(["-15","-10","-5","0","5"], fontsize=labelsize_)
# =============================================================================
# =============================================================================
#     ax.set_xlim((xmin, xmax))
#     ax.set_ylim((ymin, ymax))
#     
# =============================================================================
    anim = TrajectoryAnimation(*paths, labels=methods, ax=ax)
    ax.legend(loc='upper left')
    anim.save("ani2.mp4",dpi=300)
    
    return


comparision_animation(F1)