# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:49:07 2022

@author: jomor
"""

import numpy as np
from flat_interface_env import *

### parameters: kon=1.0, kp=1.0, koff=1.0, sigma=5.0, N=3, n_0=[1000,0], T_decision=60
def fi_env_main(kon=1.0, kp=1.0, koff=1.0, sigma=5.0, N=3, n_0=[1000,0], T_decision=60):
    p_plus = 0.5
    p_h = 0.5
    
    
    ### number of samples to approximate gamma ###
    kmax = 1
    ### array for recoding activation k_max activation outcomes ###
    X = np.zeros(kmax)
    
    for k in range(kmax):
        
        ### determine if agonist positive and how many agonists ###
        s = np.random.binomial(1, p_plus, 1)[0]
        if s == 1:
            n_ag = np.random.randint(100, size=1)[0]
            #print(n_ag)
        else:
            n_ag = 0
            
        ### draw host populaitons ###
        n_h = np.random.binomial(20000, p_h, 1)[0]
        
        ### Simulate the Model
        FPT_array = np.zeros((1,10000))        
    
        FI = FlatInterfaceFP(kon=kon, kp=kp, koff=koff, sigma=5.0, N=N, n_0=[n_h,n_ag], T_decision=T_decision)       
        FI.reset()
        while FI.done==False:
            FI.tau_step()
            FI.update_state()
        
        if n_ag > 0 and FI.activation == 1:
            X[k] = 1 
        elif n_ag > 0 and FI.activation == 0:
            X[k] = 0
        elif n_ag == 0 and FI.activation == 1:
            X[k] = 0
        elif n_ag == 0 and FI.activation == 0:
            X[k] = 1
        
    reward = np.sum(X)/kmax
    
    next_state = [kon, kp, koff, N, T_decision, reward]
    
    return next_state, reward
        
# next_state, gamma_reward = fi_env_main(kon=1.0, kp=1.0, koff=1.0, sigma=5.0, N=3, n_0=[1000,0], T_decision=60)
# print(gamma)    
    