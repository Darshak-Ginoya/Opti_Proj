
import numpy as np
import matplotlib.pyplot as plt





def plot_ing(df):



    x_vec1=np.array(df[0])
    f_vec1=np.array(df[1])


    N=np.vstack(np.arange(0,x_vec1.shape[0],1))
    plt.figure(figsize=(6,4),dpi=300)
    plt.plot(N,x_vec1[:,0],'--r',label='X1')
    plt.plot(N,x_vec1[:,1],label='X2')
    plt.xlabel('Iteration')
    plt.ylabel('X1 & X2 ')
    plt.title('X1 & X2 plotted against iteration for rosenberg func')  
    plt.grid()
    plt.legend(bbox_to_anchor=(0.75,0.75),ncol=2)
    plt.show()


    plt.figure(figsize=(6,4),dpi=300)
    plt.plot(N,f_vec1)
    plt.xlabel('iteration')
    plt.ylabel('Function Value')
    plt.title(' Function Value plotted for each iteration')  
    plt.grid()
    plt.show()

