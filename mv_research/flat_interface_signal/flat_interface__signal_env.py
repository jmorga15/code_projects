# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:33:55 2022

@author: jomor
"""
import numpy as np
import random
import h5py

### This is the flat interface environment with signal activetion
### the signal is accumulated and must pass a threshold
### this model does not account for discrete populations of TCRs
### this model does not account for second order binding


class FlatInterfaceFP():
    def __init__(self, kon=1.0, kp=1.0, koff=1.0, kact = 1.0, sig_threshold=0.9, sigma=5.0, N=3, n_0=[1000,0], T_decision=60 ):
        
        # kp parameters #
        self.kon = kon
        self.kp = kp
        self.koff = koff
        self.kact = kact
        self.sig_threshold = sig_threshold
        
        # sigma-fold change in dissociation rate
        self.sigma = sigma
        
        # probabilities used in update function
        self.alpha_ag = self.kp/(self.kp+self.koff)
        self.alpha_host = self.kp/(self.kp+self.koff*self.sigma)
        
        # kp steps
        self.N = N
        
        # antigen population
        self.n_host=n_0[0]
        self.n_ag = n_0[1]
        
        ### total signal by time t
        self.signal = 0
        
        # decision time (contact time)
        self.T_decision = T_decision       
        
        # antigen state_vector (state, is_agonist)
        # state is the tcr/antigen bound state {0->dissociated, 1->C0 (first bound state), 2->C1, 3->C2, ..., N+1->CN}
        self.antigen_state = np.zeros((self.n_host+self.n_ag,2),dtype=np.int64)
        for i in range(self.n_host+self.n_ag):
            if i < self.n_host:
                self.antigen_state[i,1] = 0
            else:
                self.antigen_state[i,1] = 1
        
        ### done flag ###
        self.done = False
        
    def reset(self):
        
        ### done flag ###
        self.done = False
        
        # start time
        self.t = 0
        
        # antigen state_vector
        # state is the tcr/antigen bound state {0->dissociated, 1->C0 (first bound state), 2->C1, 3->C2, ..., N+1->CN}
        # self.host_state = np.zeros(self.n_host)
        # self.ag_state = np.zeros(self.n_ag)
        #self.antigen_state[:,0] = 1
        
        for i in range(self.n_host+self.n_ag):
            if i < self.n_host:
                self.antigen_state[i,1] = 0
            else:
                self.antigen_state[i,1] = 1
        
    def tau_step(self):
        
        self.tau_step_ = 0.001
        self.t = self.t + self.tau_step_
        
    def update_state(self):
        
        rng = np.random.default_rng()
        
        ### Event Propensities ###
        
        # binding #
        #if np.sum(self.antigen_state[:,0]==0)>0:
        transition_pop = self.antigen_state[:,0]==0
        kon_prop = np.sum(transition_pop)*self.kon
        
        bindings = rng.poisson(kon_prop*self.tau_step_,1)
        
        bindings = np.min([bindings[0],np.sum(transition_pop)])
        
        # the change vector
        # indexes of antigen that can transition
        transition_indexes = np.where(self.antigen_state[:,0]==0)
        # locations of indexes that bind
        binding_transitions = random.sample(transition_indexes[0].tolist(), bindings)
        
        
        # escap Bi host
        # the number of transitions that occur starting from a bound state #
        #if np.sum(self.antigen_state[:,0]>0)>0:
        transition_pop = (self.antigen_state[:,0]>0) * (self.antigen_state[:,1]==0)
        bound_escape_prop = np.sum(transition_pop)*(self.kp + self.koff*self.sigma)
        
        
        bound_escapes = rng.poisson(bound_escape_prop*self.tau_step_,1)
        
        #print(bound_escapes)
        
        bound_escapes = np.min([bound_escapes[0],np.sum(transition_pop)])
        
        #print(bound_escapes)
        
        #print(self.antigen_state)
        # indexes of antigen that can transition
        transition_indexes = np.where(self.antigen_state[:,0]>0)
        escape_transitions = random.sample(transition_indexes[0].tolist(), bound_escapes)
        
        # a vector to determine if the escapes were dissociations or phospho
        v_escape_to = np.random.binomial(1, self.alpha_host, bound_escapes)
        
        #print(v_escape_to)
     
        
        
        
        
        
      
        
        # update binding #
        if bindings > 0:
            # make adjustments (this should go at the end after debugging)
            self.antigen_state[binding_transitions,0] = 1
        
      
        # update state for escapes
        if bound_escapes > 0:
            for i in range(len(escape_transitions)):
                index = escape_transitions[i]
                if v_escape_to[i] == 0:
                    self.antigen_state[index,0] = 0
                else:
                    self.antigen_state[index,0] += 1
                    

        # signal accumulation
        activated_receptors = np.sum(self.antigen_state[:,0]==4) + np.sum(self.antigen_state[:,1]==4)
        if activated_receptors > 0:
            self.signal = self.signal + self.tau_step_*(activated_receptors)*(self.kact * (1.0-self.signal))
            print(self.signal)
            
            if self.signal>self.sig_threshold:
                self.done = True
            
        
        
        
        
    def step(self,action):
        ### here, take action and update environment ###
        ### action is an array with model parameters ###
        # kp parameters #
        self.kon = action[0]
        self.kp = action[1]
        self.koff = action[2]
        
        
        # probabilities used in update function
        self.alpha_ag = self.kp/(self.kp+self.koff)
        self.alpha_host = self.kp/(self.kp+self.koff*self.sigma)
        
        # kp steps
        self.N = action[3]
        
        
        # decision time (contact time)
        self.T_decision = action[4]
        
        
        ### need to return reward, new_observation, done ###
#        observation = 
        
        
        
        
FPT_array = np.zeros((1,100))        

for ind in range(1):
    FI = FlatInterfaceFP()       
    FI.reset()
    while FI.done==False:
        FI.tau_step()
        FI.update_state()
    FPT_array[0,ind] = FI.t
    
# data_file = h5py.File('FPT_array_tau_leap.h5', 'a')
# data_file.create_dataset('default_params_n_1000_1_koff_5', data=FPT_array)
# data_file.close()    

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    