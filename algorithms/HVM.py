
import numpy as np



def HV_modified(func,x_init,alpha,epsilon,delta,max_iter):
   
    
    k=0
    it =0
    xk=x_init
    x_vec_save=[]
    f_vec_save=[]
    x_vec_save.append(x_init)             #To create array of all x at different iteration
    f_vec_save.append(func(x_init))
    n=x_init.shape[0]
    e=np.eye(n)
    while delta>epsilon and k<max_iter:
        for i in range(n):
            xi = xk
            he=delta*(np.vstack(e[:,i]))
            x_p=xk+he
            x_n=xk-he
            f=func(xk)
            f1=func(x_p)
            f2=func(x_n)
            if f1<f:
                xk_1=x_p
                x_converged = xk_1
            elif f2<f:
                xk_1=x_n
                x_converged = xk_1
            else:
                xk_1=xk
                x_converged = xk_1
            xk=xk_1
          
        if (xk_1).all==(xi).all:
            delta=delta/alpha
        else:
            
            sk = xk_1-xi
            for i in range (max_iter):
                
                xk_2 = xk_1+sk
                f_new = func(xk_2)
                f_old = func(xk_1)
                if f_new<f_old:
                    x_converged=xk_2
                    xk_1 = xk_2
                else:
                    x_converged = xk_1
                    break
                i+=1
            
        k=k+1
        it=k
        xk = x_converged
        x_vec_save.append(x_converged)
        f_save = func(x_converged)
        f_vec_save.append(f_save)
        
    return(x_vec_save,f_vec_save,it,x_converged)

