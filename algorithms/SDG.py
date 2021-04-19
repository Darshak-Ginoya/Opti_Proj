import numpy as np


def grad(func, x_star):
    

    h=0.001
    n=x_star.shape[0]
    e=np.eye(n)
    delF=np.zeros((n,1))

    for index in range(n):
        he=h*np.vstack(e[:, index])
        x_p_he = x_star + he
        x_m_he = x_star - he
        delF[index,0] = (func(x_p_he)-func(x_m_he))/(2*h)

    ## END YOUR CODE HERE ##
    return delF


def SDG(func, x_init, max_iter, tolerance):
    
   
    x_vec_save=[]
    f_vec_save=[]
    it=np.zeros(max_iter)
    C = np.eye(2)
    x_converged=[]
          

    for k in range(max_iter):
        
        G=grad(func, x_init)
        p = -G
        
        alfa=5
        rho=0.8
        c=0.1
        x_init1=x_init+alfa*p
        i=0
   
        while (func(x_init1)>(func(x_init)+c*alfa*(p.T.dot(G)))) and (i<100):
              
              alfa=rho*alfa
              x_init1=x_init+alfa*p
              i+=1
              
        x_vec_save.append(x_init)
        f_vec_save.append(func(x_init))      
        it[k]=k+1
        
        x_init=x_init1
        
        if  np.linalg.norm(G)**2<tolerance:
            x_converged=x_init
            it=it[0:k+1]
            break
        
    return  x_vec_save, f_vec_save, x_converged


