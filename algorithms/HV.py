import numpy as np


def HV(func,x_init,alpha,epsilon,delta,max_iter):
    k=0
    xk=x_init
    x_vec_save1=[]
    f_vec_save1=[]
    x_vec_save1.append(x_init)             #To create array of all x at different iteration
    f_vec_save1.append(func(x_init))
    n=x_init.shape[0]
    e=np.eye(n)
    while delta>epsilon and k<max_iter:
        for i in range(n):
            he=delta*(np.vstack(e[:,i]))
            x_p=xk+he
            x_n=xk-he
            f=func(xk)
            f1=func(x_p)
            f2=func(x_n)
            if f1<f:
                xk_1=x_p
            elif f2<f:
                xk_1=x_n
        if xk_1.all==xk.all:
            delta=delta/alpha
            for i in range(n):
                he=delta*(np.vstack(e[:,i]))
                x_p=xk+he
                x_n=xk-he
                f=func(xk)
                f1=func(x_p)
                f2=func(x_n)
                if f1<f:
                    xk_1=x_p
                elif f2<f:
                    xk_1=x_n
        xk_2=2*xk_1-xk
        if func(xk_2)<func(xk):
            xk=xk_2
        else:
            xk=xk_1
        x_converged=xk
        x_vec_save1.append(x_converged)
        F=func(x_converged)
        f_vec_save1.append(F)
        k=k+1
        it=k
    x_vec_save=x_vec_save1
    f_vec_save=f_vec_save1
    return x_vec_save,f_vec_save,it,x_converged

