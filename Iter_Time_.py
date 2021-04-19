

import compare as cp
import time 
import numpy as np
from benchmark_func import F1,F2








methods = [
    
    "BFGS",
    
    "Nelder-Mead",
    "Powell",
    "DE",
    "basinhopping",
    "dual_annealing",
    "HV",
    "HV_modified",
    "SDG"
    
]

x0=np.array([4.5,4.5])

for method in methods:
    
    delF=cp.compare(F2,x0,algo=method,initial=time.time()) 
    final_time=delF[2]
    iter=np.array(delF[0]).shape[0]
    print('Required Iteration and time are',iter,'and',final_time,'seconds for',method,'Algorithm')