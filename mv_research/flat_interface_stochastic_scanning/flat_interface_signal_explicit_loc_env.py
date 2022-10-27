# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:33:55 2022

@author: jomor
"""
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from MV_class import *


### This is the flat interface environment with signal activetion
### the signal is accumulated and must pass a threshold
### this model does not account for discrete populations of TCRs
### this model does not account for second order binding


class FlatInterfaceFP():
    def __init__(self,ka = 1.0, kd = 1.0, ku=1.0, ks = 1.0, max_mv = 50, kon=1.0, kp=1.0, koff=1.0, kact = 1.0, sig_threshold=0.9, tau_step = 0.01, sigma=5.0, N=3, n_0=[0,1], T_decision=60, signal_activated = False, fp_activated=True, vog_points = False, record_AF = False, record_CF = False):

        ### initialize the MV states ###
        self.MV_state = MV_scan(ka = ka, kd = kd, ks = ks, ku = ku, max_mv = max_mv, vog_points = vog_points, record_AF = record_AF, record_CF = record_CF)


        # kp parameters #
        self.kon = kon
        self.kp = kp
        self.koff = koff
        self.kact = kact
        self.sig_threshold = sig_threshold

        # sigma-fold change in dissociation rate
        self.sigma = sigma
        
        ### activation type ###
        self.signal_activated = signal_activated
        self.fp_activated = fp_activated

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
        
        # tau step
        self.tau_step_ = tau_step

        # antigen state_vector (state, is_agonist, x_location, y_location)
        # state is the tcr/antigen bound state {0->dissociated, 1->C0 (first bound state), 2->C1, 3->C2, ..., N+1->CN}
        self.antigen_state = np.ones((self.n_host+self.n_ag,4),dtype=np.float64)*(-1.0)
        for i in range(self.n_host+self.n_ag):
            if i < self.n_host:
                self.antigen_state[i,1] = 0
            else:
                self.antigen_state[i,1] = 1

        # antigen locations #
        self.antigen_locations()

        ### done flag ###
        self.done = False

    def antigen_locations(self):


        n = self.n_host + self.n_ag

        ### uniform in disk ###
        R = 1
        u = np.random.uniform(0,1,n)
        v = np.random.uniform(0,1,n)

        radius = R*np.sqrt(v)
        theta = 2*np.pi*u


        coordinates = np.zeros((n, 2))
        coordinates[:,0] = radius * np.cos(theta)
        coordinates[:,1] = radius * np.sin(theta)
        #print('coordinates',coordinates)
        self.antigen_state[:,2:] = (coordinates[:,:])
        #print('antigen state', self.antigen_state)
        #print('MV state', self.MV_state.mv_state)

    def plot_disk(self):

        # just plots the antigen locations #

        circle2 = plt.Circle((0, 0), 1, color='b', fill=False)


        fig = plt.figure(dpi=400)

        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax = fig.add_subplot(1, 1, 1)

        ax.add_patch(circle2)
        color = np.sqrt((self.antigen_state[:,2:]**2).sum(axis = 1))/np.sqrt(2.0)
        rgb = plt.get_cmap('jet')(color)
        ax.scatter(self.antigen_state[:,2], self.antigen_state[:,3], color = rgb)
        plt.show()

    def reset(self):

        ### done flag ###
        self.done = False

        # start time
        self.t = 0

        # antigen state_vector
        # state is the tcr/antigen bound state {-1->not accessible,0->accessible but dissociated, 1->C0 (first bound state), 2->C1, 3->C2, ..., N+1->CN}
        # self.host_state = np.zeros(self.n_host)
        # self.ag_state = np.zeros(self.n_ag)
        #self.antigen_state[:,0] = 1

        for i in range(self.n_host+self.n_ag):
            if i < self.n_host:
                self.antigen_state[i,1] = 0
            else:
                self.antigen_state[i,1] = 1

    def tau_step(self):

        
        self.t = self.t + self.tau_step_
        if self.t > self.T_decision:
            self.done = True
            
    def deStabilization_pop(self):
        
        ### a vector containing 1 if can destabilize and 0 if cannot ###
        ### 3 columns, 0 column: can_destab, 1 column: x, 2 column: y
        ### transition pop has same length of mv_state
        transition_pop_potential = np.zeros((len(self.MV_state.mv_state[:,0]),3))
        transition_pop_potential[:,1:3] = self.MV_state.mv_state[:,1:3]
        
        ### get the indices of stabilized MV ###
        stabilized_pop = (self.MV_state.mv_state[:,0]==2)
        
        ### turn the stabilized mv into potential destabilizers ###
        loc_stab_mv = np.where(stabilized_pop)
        transition_pop_potential[loc_stab_mv[0],0] = 1
        
        ### get state vectors for each mv ###
        mv_stab_state = self.MV_state.mv_state[loc_stab_mv[0],:]
        
        
        ### get activated TCR Indexes ###
        activated_TCR_pop = np.where(self.antigen_state[:,0] == self.N+1)
        ### get x and y coordinates: antgien_state[_,2]=x, antigen_state[_,3]=y
        x,y = self.antigen_state[activated_TCR_pop[0],2], self.antigen_state[activated_TCR_pop[0],3]
        
        ### get distance measure check for each mv ###
        for mvi in range(len(transition_pop_potential)):
            ### check to see if it is stabilized ###
            if transition_pop_potential[mvi,0] == 1:
                mvx = transition_pop_potential[mvi,1]
                mvy = transition_pop_potential[mvi,2]
                ### check the radial distances from the center of the MV contact to the activated TCR/antigen complex ###
                radial_distance = np.sqrt( (x - mvx)**2 + (y - mvy)**2 )
                
                # check if any MV cover antigen  #
                if np.any(radial_distance < self.MV_state.rMV):
                    transition_pop_potential[mvi,0] = 0
                    
        transition_pop = transition_pop_potential[:,0] 
        transition_indexes = np.where(transition_pop_potential[:,0] == 1)           
        #print(x.shape,y.shape)
        ### stabilized 
        
        return transition_pop, transition_indexes
    
    def stabilization_pop(self):
        
        ### a vector containing 1 if can stabilize and 0 if cannot ###
        ### 3 columns, 0 column: can_stab, 1 column: x, 2 column: y
        ### transition pop has same length of mv_state
        transition_pop_potential = np.zeros((len(self.MV_state.mv_state[:,0]),3))
        transition_pop_potential[:,1:3] = self.MV_state.mv_state[:,1:3]
        
        
        ### get the indices of unstabilized MV ###
        unstabilized_pop = (self.MV_state.mv_state[:,0]==1)
        
        
        
        ### turn the stabilized mv into potential destabilizers ###
        loc_stab_mv = np.where(unstabilized_pop)
        transition_pop_potential[loc_stab_mv[0],0] = 1
        
        ### get state vectors for each mv ###
        mv_stab_state = self.MV_state.mv_state[loc_stab_mv[0],:]
        
        
        ### get activated TCR Indexes ###
        activated_TCR_pop = np.where(self.antigen_state[:,0] == self.N+1)
        
        
        x,y = self.antigen_state[activated_TCR_pop[0],2], self.antigen_state[activated_TCR_pop[0],3]
        #print(x,y)
        
        
        ### get distance measure check for each mv ###
        for mvi in range(len(transition_pop_potential)):
            ### check to see if it is stabilized ###
            if transition_pop_potential[mvi,0] == 1:
                mvx = transition_pop_potential[mvi,1]
                mvy = transition_pop_potential[mvi,2]
                ### check the radial distances from the center of the MV contact to the activated TCR/antigen complex ###
                radial_distance = np.sqrt( (x - mvx)**2 + (y - mvy)**2 )
                
                # check if any MV cover antigen  #
                if np.any(radial_distance < self.MV_state.rMV):
                    transition_pop_potential[mvi,0] = 1
                else:
                    transition_pop_potential[mvi,0] = 0
                    
        transition_pop = transition_pop_potential[:,0] 
        transition_indexes = np.where(transition_pop_potential[:,0] == 1)           
        
        ### stabilized 
        
        return transition_pop, transition_indexes

    def update_state(self):

        rng = np.random.default_rng()

        ### Event Propensities ###
        ### this is a mv model, so antigen can only bind when covered by MV contact

        # first update the antigen that are covered by MV

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
        bound_escape_prop_host = np.sum(transition_pop)*(self.kp + self.koff*self.sigma)


        bound_escapes_host = rng.poisson(bound_escape_prop_host*self.tau_step_,1)

        bound_escapes_host = np.min([bound_escapes_host[0],np.sum(transition_pop)])

        # indexes of antigen that can transition
        transition_indexes = np.where(self.antigen_state[:,0]>0)
        escape_transitions_host = random.sample(transition_indexes[0].tolist(), bound_escapes_host)

        # a vector to determine if the escapes were dissociations or phospho
        v_escape_to_host = np.random.binomial(1, self.alpha_host, bound_escapes_host)
        
        
        
        # escap Bi Agonist
        # the number of transitions that occur starting from a bound state #
        transition_pop = (self.antigen_state[:,0]>0) * (self.antigen_state[:,1]==1)
        bound_escape_prop_agonist = np.sum(transition_pop)*(self.kp + self.koff)


        bound_escapes_agonist = rng.poisson(bound_escape_prop_agonist*self.tau_step_,1)

        bound_escapes_agonist = np.min([bound_escapes_agonist[0],np.sum(transition_pop)])

        # indexes of antigen that can transition
        transition_indexes = np.where(self.antigen_state[:,0]>0)
        escape_transitions_agonist = random.sample(transition_indexes[0].tolist(), bound_escapes_agonist)

        # a vector to determine if the escapes were dissociations or phospho
        v_escape_to_agonist = np.random.binomial(1, self.alpha_ag, bound_escapes_agonist)



        ### now get the MV capable of being added (state 0) ###
        transition_pop = (self.MV_state.mv_state[:,0]==0)
        MV_addition_prop = np.sum(transition_pop)*(self.MV_state.ka)

        ### number of MV to attempt to add ###
        MV_additions = rng.poisson(MV_addition_prop*self.tau_step_,1)

        ### cant have more additions than MV capable of being added ###
        MV_additions = np.min([MV_additions[0],np.sum(transition_pop)])



        ### next get the MV capable of being removed (state 1) ###
        transition_pop = (self.MV_state.mv_state[:,0]==1)
        MV_removal_prop = np.sum(transition_pop)*(self.MV_state.kd)

        ### number of MV to attempt to add ###
        MV_removals = rng.poisson(MV_removal_prop*self.tau_step_,1)

        ### cant have more additions than MV capable of being added ###
        MV_removals = np.min([MV_removals[0],np.sum(transition_pop)])



        ### next get the MV capable of being destabilized (state 2) ###
        # transition_pop = all MV that are stabilized and don't house activated TCR
        
        transition_pop, transition_indexes_destab = self.deStabilization_pop()
        #transition_pop = (self.MV_state.mv_state[:,0]==2)
        MV_destabilize_prop = np.sum(transition_pop)*(self.MV_state.ku)

        ### number of MV to attempt to add ###
        MV_destabilize = rng.poisson(MV_destabilize_prop*self.tau_step_,1)

        ### cant have more destabilizations than MV capable of being destabilized ###
        MV_destabilize = np.min([MV_destabilize[0],np.sum(transition_pop)])



        ### next get the MV capable of being stabilized (state 2) ###
        transition_pop, transition_indexes_stab = self.stabilization_pop()
        #transition_pop = (self.MV_state.mv_state[:,0]==1)
        MV_stabilize_prop = np.sum(transition_pop)*(self.MV_state.ks)

        ### number of MV to attempt to add ###
        MV_stabilize = rng.poisson(MV_stabilize_prop*self.tau_step_,1)

        ### cant have more additions than MV capable of being added ###
        MV_stabilize = np.min([MV_stabilize[0],np.sum(transition_pop)])


        ### NOTE: the order of operations to an update
        ### 1) update the signal, if tau is a time step, then we know that using
        ###     tau-leaping, that previously activated receptors were activated for tau time
        ### 2) update the receptor positions; some of these updates may be redundant since
        ###     mv removals will negate any antigen transitions
        ### 3) add or remove contacts; if mv are added, then we need to cmopute what antigen, if any,
        ###     have been newly covered (-1->0). if a mv is removed, then what antigen were forcibly dissociated
        ###     back to the -1 state (0->-1)
        
        # signal accumulation
        activated_receptors = np.sum(self.antigen_state[:,0]==self.N+1)
        if activated_receptors > 0:
            #print('Activated Receptors',activated_receptors)
            self.signal = self.signal + self.tau_step_*(activated_receptors)*(self.kact * (1.0-self.signal))
            #print(self.signal)

            ### signal activation done flag ###
            if self.signal_activated==True:
                if self.signal>self.sig_threshold:
                    self.done = True
                
            ### first passage activation done flag ###
            if self.fp_activated==True:
                self.done = True

        # update binding #
        if bindings > 0:
            # make adjustments (this should go at the end after debugging)
            self.antigen_state[binding_transitions,0] = 1


        # update state for host escapes
        if bound_escapes_host > 0:
            for i in range(len(escape_transitions_host)):
                index = escape_transitions_host[i]
                if v_escape_to_host[i] == 0:
                    self.antigen_state[index,0] = 0
                else:
                    self.antigen_state[index,0] += 1
                    
        # update state for agonist escapes
        if bound_escapes_agonist > 0:
            for i in range(len(escape_transitions_agonist)):
                index = escape_transitions_agonist[i]
                if v_escape_to_agonist[i] == 0:
                    self.antigen_state[index,0] = 0
                else:
                    self.antigen_state[index,0] += 1

        ### call the MV destabilization method ###
        self.MV_state.destabilize_MV(MV_destabilize,transition_indexes_destab)

        ### call the MV stabilization method ###
        #print('MV_removals',MV_removals)
        self.MV_state.stabilize_MV(MV_stabilize, transition_indexes_stab)

        ### call the MV removal method ###
        #print('MV_removals',MV_removals)
        self.MV_state.remove_MV(MV_removals)



        ### call the mv add method ###
        #print('MV_additions',MV_additions)
        self.MV_state.add_MV(MV_additions)

        if self.MV_state.vog_points == True:
            self.MV_state.vogel_points_capture()

        if self.MV_state.record_AF == True:
            self.MV_state.get_instantaneous_area_fraction( self.t)

        if self.MV_state.record_CF == True:
            self.MV_state.get_cumulative_area_fraction( self.t)

        ### mv are added, now check if any antigen are newly covered ###
        self.antigen_state = self.MV_state.newly_covered_antigen(self.antigen_state)

        #print('new mv_state',self.MV_state.mv_state)
        #print('new antigen_state',self.antigen_state)

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


    def plot_model_current_state(self):

        # plots antigen and MV #

        circle2 = plt.Circle((0, 0), 1, color='b', fill=False)


        fig = plt.figure(dpi=400)

        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax = fig.add_subplot(1, 1, 1)

        ax.add_patch(circle2)
        color = np.sqrt((self.antigen_state[:,2:]**2).sum(axis = 1))/np.sqrt(2.0)
        rgb = plt.get_cmap('jet')(color)
        #ax.scatter(self.antigen_state[:,2], self.antigen_state[:,3], color = rgb)

        for mv in self.MV_state.mv_state:
            if mv[0]==1:
                circle2 = plt.Circle((mv[1], mv[2]), self.MV_state.rMV, color='b', fill=False)

                ax.add_patch(circle2)

            if mv[0]==2:
                circle2 = plt.Circle((mv[1], mv[2]), self.MV_state.rMV, color='r', fill=False)

                ax.add_patch(circle2)

        if self.MV_state.vog_points==True:
            ax.scatter(self.MV_state.vog_coordinates[:,1], self.MV_state.vog_coordinates[:,2], color='C0', s=2)
            #print(len(self.MV_state.vog_coordinates[:,1]))




        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])


        plt.show()


    def plot_AF(self):
        fig = plt.figure(dpi=400)

        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax = fig.add_subplot(1, 1, 1)

        if self.MV_state.record_AF == True:
            ax.plot(self.MV_state.t_AF[:,0], self.MV_state.t_AF[:,1], color='C0',linewidth=5)

        ax.set_ylim([0, 1])

        plt.show()



# FI = FlatInterfaceFP()
# FI.reset()
# #FI.plot_disk()
# FI.tau_step()
# FI.update_state()

# FI.tau_step()
# FI.update_state()
# # FI.tau_step()
# # FI.update_state()

# FI.plot_model_current_state()
#data_file = h5py.File('MV_instantaneous_area_coverage.h5', 'a')

tau_step = 0.01
T_decision = 60
num_step = T_decision/tau_step
FPT_array = np.zeros((5,100))
CF_array = np.zeros((5,int(num_step)+2,2))
#ks_list = [0.01,0.1, 1.0, 5.0, 10]
nAG_list = [10, 100, 1000, 5000, 10000]
#for ind in range(len(ks_list)):
for ind in range(len(nAG_list)):
    for episode in range(1):
        #ks = ks_list[ind]
        ks = 10.0
        nAG = nAG_list[ind]
        FI = FlatInterfaceFP(ks = ks, ku = 0.01, ka = 1.0, kd=0.5, koff=1.0, kact = 0.01, max_mv = 150, T_decision=60, n_0=[1,nAG], signal_activated = True,  sig_threshold=1.9, fp_activated=False, vog_points = True, record_AF = False, record_CF = True)
        FI.reset()
        while FI.done==False:
            FI.tau_step()
            FI.update_state()
        FPT_array[ind,episode] = FI.t
        FI.plot_model_current_state()
        #print('mv state',FI.MV_state.t_AF.shape)
        print([ind,episode, FI.t])
    CF_array[ind] = FI.MV_state.t_CF

data_file = h5py.File('MV_CAF_varyAG.h5', 'a')
#data_file.create_dataset('FPActivation(kd_index,trial)', data=FPT_array)
#data_file.create_dataset('CAF_MV150', data=CF_array)
data_file.create_dataset('CAF_MV150', data=CF_array)
data_file.close()
FI.plot_model_current_state()
FI.plot_AF()

print('FI' , FI.t)
# print('antigen state',FI.antigen_state)
#print('mv state',FI.MV_state.mv_state)
#FI.plot_model_current_state()
#print('AF', FI.MV_state.t_AF[:,1])
# data_file = h5py.File('FPT_array_tau_leap.h5', 'a')
# data_file.create_dataset('default_params_n_1000_1_koff_5', data=FPT_array)
# data_file.close()
