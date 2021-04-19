##comparison part

import time
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import algorithms.HV as hv
import algorithms.HVM as hvm
import algorithms.SDG as sdg






def compare(f,x0,algo,initial):

    '''
    
    =======================================================
           
    methods to which we want to compare our GPS method

    
    ##Levenberg-Marquardt    (leastsq least_squares )
    Powell
    Nelder-Mead
    Powell
    BFGS
    basinhopping
    dual_annealing
    DE                       (differential evolution)

    =======================================================
    
    '''    
    
    
    paths_=[x0]
    f_val =[f(x0)]
    
    initial_time=initial
    
    c=x0
    
    bounds_x1 = (-5,5)
    bounds_x2 = (-5,5)
    bounds=[bounds_x1,bounds_x2]
    
    #initial guess 
    x_init=np.vstack(x0)
    
    
    #HV PARAmeters 
    delta=1
    alfa=2
    epsilon=1e-6    
    max_iter=20000 
    
    #SDG parameter
    tolerance=1e-6
    
    
    if algo=="HV_modified":
       
       
       #Evaluating
       result=hvm.HV_modified(f,x_init,alfa,epsilon,delta,max_iter)
       v= np.array(result[0])
       
        
       ##to reduce shape of x_vec from(1,2,iterarion) to (2,iteration)
        
       n=v.shape[0]
       v1=v.reshape(n,2)
       
        
        
       ##to animate "(total iteration)/d" iteration only to make animation faster for HV by 
       if n>=500 and n<1000: 
          d=3
       elif n>=1000 and n<1500:
          d=5
       elif n>=1500 and n<3000:
          d=10
       elif n>=3000 and n<6000:
          d=20
       elif n>=6000 and n<12000:
          d=60
       elif n>=12000:
          d=120
       else:
          d=1 
          
       j=n//d
            
       k1=np.array(np.arange(0,j*d,d))
       k2=np.hstack((k1,n-1))
       paths_=[] 
       
       for i,res in enumerate(k2):
           paths_.append(v1[res,:])

       
       
    
    elif algo=="HV":
        
       
       
       #Evaluating
       result=hv.HV(f,x_init,alfa,epsilon,delta,max_iter)  
      
       v= np.array(result[0])
       
        
       ##to reduce shape of x_vec from(1,2,iterarion) to (2,iteration)
        
       n=v.shape[0]
       v1=v.reshape(n,2)
       
        
        
       ##to animate "(total iteration)/d" iteration only to make animation faster for HV by 
       if n>=500 and n<1000: 
          d=3
       elif n>=1000 and n<1500:
          d=5
       elif n>=1500 and n<3000:
          d=10
       elif n>=3000 and n<6000:
          d=20
       elif n>=6000 and n<12000:
          d=60
       elif n>=12000:
          d=120
       else:
          d=1 
       j=n//d   
       j=n//d
            
       k1=np.array(np.arange(0,j*d,d))
       k2=np.hstack((k1,n-1))
       paths_=[] 
       
       for i,res in enumerate(k2):
           paths_.append(v1[res,:])

    
    elif algo=="SDG":
       
       
       
       #Evaluating
       result=sdg.SDG(f, x_init, max_iter, tolerance)
       
       v= np.array(result[0])
       
        
       ##to reduce shape of x_vec from(1,2,iterarion) to (2,iteration)
        
       n=v.shape[0]
       v1=v.reshape(n,2)
       
        
        
       ##to animate "(total iteration)/d" iteration only to make animation faster for HV by 
       if n>=500 and n<1000: 
          d=3
       elif n>=1000 and n<1500:
          d=5
       elif n>=1500 and n<3000:
          d=10
       elif n>=3000 and n<6000:
          d=20
       elif n>=6000 and n<12000:
          d=60
       elif n>=12000:
          d=120
       else:
          d=1 
       j=n//d
            
       k1=np.array(np.arange(0,j*d,d))
       k2=np.hstack((k1,n-1))
       paths_=[] 
       
       for i,res in enumerate(k2):
           paths_.append(v1[res,:])

        
    elif algo=="Levenberg-Marquardt":
         
       def store(xk):
        
           paths_.append(np.array(xk))
           f_val.append(np.array(f(xk)))   

       k=optimize.least_squares(f,x0, method='lm')
    
    elif algo=="Powell":
         
       def store(xk):
        
           paths_.append(np.array(xk))
           f_val.append(np.array(f(xk)))   

       k=optimize.minimize(f, x0, method="Powell", callback=store )
       
    elif algo=="Nelder-Mead":
         
       def store(xk):
        
           paths_.append(np.array(xk))
           f_val.append(np.array(f(xk)))   

       k=optimize.minimize(f, x0, method="Nelder-Mead", callback=store ) 
    
    elif algo=="BFGS":
         
       def store(xk):
        
           paths_.append(np.array(xk))
           f_val.append(np.array(f(xk)))  

       k=optimize.minimize(f, x0, method="BFGS", callback=store )        
       
    elif algo=="basinhopping":  
         
         def store(xk,f1,convergence=c):
        
             paths_.append(np.array(xk))
             f_val.append(np.array(f(xk)))  
         minimizer_kwargs = { "method": "Powell", "bounds":bounds }
         k=optimize.basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, callback=store)
    
    elif algo=="dual_annealing":
         
         def store(xk,f1,convergence=c):
        
             paths_.append(np.array(xk))
             f_val.append(np.array(f(xk)))     
    
         k=optimize.dual_annealing(f, bounds,callback=store )
    
    elif algo=="DE":
         def store(xk,convergence=c):
        
             paths_.append(np.array(xk))
             f_val.append(np.array(f(xk)))
         k=optimize.differential_evolution(f, bounds, callback=store, maxiter=300)


    final_time=time.time()-initial_time
    
    
    return paths_, f_val, final_time


##Evaluting


'''
 algo=   
=======================================================
          
Levenberg-Marquardt    (leastsq least_squares )
Powell
Nelder-Mead
BFGS
basinhopping
dual_annealing
DE                       (differential evolution)

=======================================================
    
'''    
methods = [
    "DE",
    "BFGS",
    "Nelder-Mead",
    "Powell",
    "basinhopping",
    "dual_annealing",
    "GPS",
    "HV",
    "HV_modified"
    
]

# =============================================================================
# x0=np.array([2.5,2.5])
# x_init=np.vstack(x0)
# #PARAmeters 
# alfa=1.1
# epsilon=1e-6
# max_iter=20000
# delta=2
# delF=GPS(x_init, alfa, epsilon, delta,max_iter)
# =============================================================================
# =============================================================================
# x0=np.array([1.5,1.5])
# paths_ = defaultdict(list)
# 
# for method in methods:
#     
#     delF=compare(func,x0,algo=method,initial=time.time())     
#     paths_[method]=np.array(delF[0])    
#     print(paths_[method])
#     paths = [np.array(paths_[method]).T for method in methods]     
#     
# zpaths = [func(path) for path in paths]
# =============================================================================

# =============================================================================
# delF=compare(func,x0,algo="GPS",initial=time.time()) 
# =============================================================================



# =============================================================================
# path=np.array(delF[0]).T
# print(path.shape)
# v= np.array(delF[0]).T
# v1=v.shape[2]
# path=v.reshape(2,v1)
# print(path)
# =============================================================================

  
    
        
