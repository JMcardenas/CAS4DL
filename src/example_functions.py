import math
import numpy as np

#------------------------------------------------------------------------------#
# Define function 
#------------------------------------------------------------------------------#
def func_test(x,d,example):
    
    if example == 1:
        # isotropic exponential 
        f_x = np.exp(-np.sum(x,axis=1)/d)
    elif example == 2:
        # anisotropic split-product
        f_x = np.array([])
        k   = math.ceil(d/2)
        for j in range(len(x)):
            t3 = np.array([])
            t4 = np.array([])
            for i in range(d):
                t1 = np.cos(16*x[j,i]/(2**(i+1)))
                t2 = 1 - x[j,i]/(4**(i+1)) 
                t3 = np.append(t3, [t1])
                t4 = np.append(t4, [t2])
            t5 = ( np.prod(t3[k:d]) ) / np.prod(t4[0:k])
            f_x = np.append(f_x, [t5])
    elif example == 3:
        # Genz peak-product
        f_x = np.array([])
        for j in range(len(x)):
            t2  = np.array([])
            for i in range(d):
                t0 = x[j,i] +  ((-1)**(i+2))/(i+2) 
                t1 = (d/4) / ( (d/4) + t0**2 )
                t2 = np.append(t2, [t1])
            t3  = np.prod(t2)
            f_x = np.append(f_x, [t3]) 
    elif example == 4:
        # absolute reciprocal 
        f_x = np.array([])
        for j in range(len(x)):
            t1  = 1/(np.sum(np.sqrt(np.abs(x[j,:]))))
            f_x = np.append(f_x, [t1]) 
    elif example == 5:
        # anisotropic MLFA : function (4.3) 
        f_x = np.array([])
        k   = math.ceil(d/2)
        for j in range(len(x)):
            t3 = np.array([])
            t4 = np.array([])
            for i in range(d):
                t1 = 1 + (4**(i+1))*(x[j,i])**2 
                t2 = 100 + 5*x[j,i]
                t3 = np.append(t3, [t1])
                t4 = np.append(t4, [t2])
            t5 = (  np.prod(t3[0:k])  / np.prod(t4[k:d]) )**(1/d)
            f_x = np.append(f_x, [t5])        

    # Convert f_x to column vector
    f_x = np.transpose(np.asmatrix(f_x))

    return f_x
