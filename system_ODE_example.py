# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 21:37:08 2022
@author: Nakarin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N_mat = 2; # Dimension of matrix
m = np.zeros((N_mat, N_mat));

def H(t): # Define matrix for system of ODE
    m[0,0]=1/9;
    m[0,1]=0;

    m[1,0]=0
    m[1,1]=np.sin(t)
    return m   
     

def dydt(t,y0): # system of ODE dy/dt=A*y0
    A=H(t)
    return A@y0

# Define initial value
y0 = np.zeros(N_mat)
y0[0] = -1
y0[1] = 1
y0=np.transpose(y0)    

# Define time
tf =10;
t_span = (0, tf) 

C_i = solve_ivp(dydt, t_span, y0, vectorized=True, method='BDF',max_step=10000,min_step=5000)
t=C_i.t # Time
C_i=C_i.y # y(t)

# Solution
plt.plot(t,C_i[0],'r')
plt.plot(t,C_i[1],'g')

# # Exact solution (Calculate from Wolfram Mathematica)
plt.plot(t,-np.exp(t/9),'r--',linewidth=2)
plt.plot(t,np.exp(1-np.cos(t)),'g--',linewidth=2)