## PART 0: Your Details





## PART A: Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from compare import HV_modified,HV
from matplotlib import animation
from IPython.display import HTML


def func(x):
    
# =============================================================================
#     fx=(1.5+x[1]*x[0]-x[0])**2 + (2.25+(x[1]**2)*x[0]-x[0])**2+(2.625+(x[1]**3)*x[0]-x[0])**2
# =============================================================================
    fx=100*(x[1]-x[0]**2)**2 + (1-x[0])**2
# =============================================================================
#     8*x[0]**2+4*x[0]*x[1]+5*x[1]**2
# =============================================================================

    return fx





## PART C: Main Code -- Animation over countour plot





f =lambda x, y: 100*(y-x**2)**2 + (1-x)**2
minima=np.array([1,1])
func(minima)
minima_=minima.reshape(-1,1)
minima_
f(*minima_)

##grid formation to make countour plot
xmin,xmax,xstep=-4.5,4.5,0.2
ymin,ymax,ystep=-4.5,4.5,0.2


x, y=np.meshgrid(np.arange(xmin,xmax+xstep,xstep),np.arange(ymin,ymax+ystep,ystep))
x2=np.array([x,y])
z=func(x2)


##initial guess and max iter            
x0=np.array([1.5,1.5])
x_init=np.vstack(x0)
 

#PARAmeters 
alfa=2
epsilon=1e-6
delta=1
max_iter=20000 

res=HV(x_init,alfa,epsilon,delta,max_iter)

v= np.array(res[0]).T
print(v.shape)

##to reduce shape of x_vec from(1,2,iterarion) to (2,iteration)

n=v.shape[2]
path_=v.reshape(2,n)
print(path_.shape)


##to animate "(total iteration)/d" iteration only to make animation faster for HV by 
d=50    
j=n//d
    
k1=np.array(np.arange(0,j*d,d))
k2=np.hstack((k1,n-1))
path=[] 
for i,res in enumerate(k2):
    path.append(path_[:,res])

path=np.array(path).T
print(path)



fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.plot(*minima_, 'r*', markersize=50)

line, = ax.plot([], [], 'b', label='Hooke Jeeve Modified Algorithm', lw=2)
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