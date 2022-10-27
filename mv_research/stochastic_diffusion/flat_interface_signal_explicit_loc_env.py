# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:33:55 2022

@author: jomor
"""
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from mv_diff_class import *


### This is the MV diffusion environment with signal activetion
### the signal is accumulated and must pass a threshold
### this model does not account for discrete populations of TCRs
### this model does not account for second order binding


class FlatInterfaceFP():
    def __init__(self, L = 1, K = 1000, ka = 1.0, kd = 1.0, rMV = 50, max_mv = 500, kon=1.0, kp=1.0, koff=1.0, kact = 1.0, sig_threshold=0.9, sigma=5.0, N=3, n_0=[1000,1], tau_step_=0.0001, T_decision=60, signal_activated = False, fp_activated=True, vog_points = False, record_AF = False, record_CF = False):

        ### initialize the MV states ###
        self.MV_state = MV_diffusion( rMV=rMV, L=1, K=1000, D=0.001, n_molecules = max_mv, x0 = [500,500], tau_step_ = tau_step_, ka = ka, kd = kd, random_walk_test = False, record_af = record_AF)
        self.MV_state.MV_locations_init(max_mv)

        # kp parameters #
        self.kon = kon
        self.kp = kp
        self.koff = koff
        self.kact = kact
        self.sig_threshold = sig_threshold

        ### activation type ###
        self.signal_activated = signal_activated
        self.fp_activated = fp_activated

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

        # tau_step #
        self.tau_step_ = tau_step_
        # decision time (contact time)
        self.T_decision = T_decision

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
        # R = 1
        # u = np.random.uniform(0,1,n)
        # v = np.random.uniform(0,1,n)

        # radius = R*np.sqrt(v)
        # theta = 2*np.pi*u

        ### uniform in box ###
        x_vec = np.random.randint(1000, size=n)
        y_vec = np.random.randint(1000, size=n)

        coordinates = np.zeros((n, 2))
        coordinates[:,0] = x_vec
        coordinates[:,1] = y_vec
        #print('coordinates',coordinates)
        self.antigen_state[:,2:] = (coordinates[:,:])
        #print('antigen state', self.antigen_state)
        #print('MV state', self.MV_state.molecule_state)

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
        #print(self.t)
        if self.t > self.T_decision:
            self.done = True
            self.activation = 0

    def update_state(self):

        rng = np.random.default_rng()

        ### Event Propensities ###
        ### this is a mv model, so antigen can only bind when covered by MV contact

        # first update the antigen that are covered by MV

        # binding #
        #if np.sum(self.antigen_state[:,0]==0)>0:
        #print(self.antigen_state)
        transition_pop = (self.antigen_state[:,0]==0)
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

        bound_escapes = np.min([bound_escapes[0],np.sum(transition_pop)])

        # indexes of antigen that can transition
        transition_indexes = np.where(self.antigen_state[:,0]>0)
        escape_transitions = random.sample(transition_indexes[0].tolist(), bound_escapes)

        # a vector to determine if the escapes were dissociations or phospho
        v_escape_to = np.random.binomial(1, self.alpha_host, bound_escapes)


        self.MV_state.diffusive_transfer_step()
        #print(np.sum(self.MV_state.molecule_state[6,:]==1))
        # ### now get the MV capable of being added (state 0) ###
        # transition_pop = (self.MV_state.mv_state[:,0]==0)
        # MV_addition_prop = np.sum(transition_pop)*(self.MV_state.ka)

        # ### number of MV to attempt to add ###
        # MV_additions = rng.poisson(MV_addition_prop*self.tau_step_,1)

        # ### cant have more additions than MV capable of being added ###
        # MV_additions = np.min([MV_additions[0],np.sum(transition_pop)])


        # ### next get the MV capable of being removed (state 1) ###
        # transition_pop = (self.MV_state.mv_state[:,0]==1)
        # MV_removal_prop = np.sum(transition_pop)*(self.MV_state.kd)

        # ### number of MV to attempt to add ###
        # MV_removals = rng.poisson(MV_removal_prop*self.tau_step_,1)

        # ### cant have more additions than MV capable of being added ###
        # MV_removals = np.min([MV_removals[0],np.sum(transition_pop)])




        ### NOTE: the order of operations to an update
        ### 1) update the signal, if tau is a time step, then we know that using
        ###     tau-leaping, that previously activated receptors were activated for tau time
        ### 2) update the receptor positions; some of these updates may be redundant since
        ###     mv removals will negate any antigen transitions
        ### 3) add or remove contacts; if mv are added, then we need to cmopute what antigen, if any,
        ###     have been newly covered (-1->0). if a mv is removed, then what antigen were forcibly dissociated
        ###     back to the -1 state (0->-1)

        # signal accumulation
        activated_receptors = np.sum(self.antigen_state[:,0]==4) + np.sum(self.antigen_state[:,1]==4)
        if activated_receptors > 0:
            self.signal = self.signal + self.tau_step_*(activated_receptors)*(self.kact * (1.0-self.signal))
            print(self.signal)

            ### signal activation done flag ###
            if self.signal_activated==True:
                if self.signal>self.sig_threshold:
                    self.done = True
                    self.activation = 1


            ### first passage activation done flag ###
            elif self.fp_activated==True:
                self.done = True

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



        ### call the MV removal method ###
        #print('MV_removals',MV_removals)
        #self.MV_state.remove_MV(MV_removals)

        ### mv are added, now check if any antigen are newly covered ###
        #self.antigen_state = self.MV_state.newly_covered_antigen(self.antigen_state)




        ### call the mv add method ###
        #print('MV_additions',MV_additions)
        #self.MV_state.add_MV(MV_additions)

        # if self.MV_state.vog_points == True:
        #     self.MV_state.vogel_points_capture()

        # if self.MV_state.record_AF == True:
        #     self.MV_state.get_instantaneous_area_fraction( self.t)

        # if self.MV_state.record_CF == True:
        #     self.MV_state.get_cumulative_area_fraction( self.t)

        ### mv are added, now check if any antigen are newly covered ###
        self.newly_covered_antigen(self.MV_state.molecule_state)

        #print('new mv_state',self.MV_state.mv_state)
        #print('new antigen_state',self.antigen_state)

    def newly_covered_antigen(self, mv_state):
        ### this method finds the antigen that are covered and returns the new antigen state
        ### get the indexes of active MV contacts ###
        #.molecule_state[6,:]==0

        active_mv_indexes = np.where(mv_state[6,:] == 1)[0]
        active_coordinates = mv_state[0:2, active_mv_indexes].transpose()

        ### index is the current antigen coordinate index we are checking ###
        index = 0
        for x,y in self.antigen_state[:,2:]:
            #print(x,y)

            ### check the radial distances from the centers of MV contacts to the antigen in question
            radial_distance = np.sqrt( (x - active_coordinates[:,0])**2 + (y - active_coordinates[:,1])**2 )

            # check if any MV cover antigen  #
            #print( 'radial distance',np.any(radial_distance < 2*self.rMV) )
            if np.any(radial_distance < self.MV_state.rMV):

                if self.antigen_state[index,0]==-1:
                    # update the antigen state (-1 -> 0) #

                    # check that the antigen state was -1
                    #assert antigen_state[index,0] == -1, "Invalid flag, why is this not -1?"

                    self.antigen_state[index,0] = 0

            elif self.antigen_state[index,0]==0:

                self.antigen_state[index,0] = -1





            index += 1

        #print(np.sum(self.antigen_state[:,0]>=0))

        #return antigen_state

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

        for mv in self.MV_state.molecule_state.transpose():
            #print(mv)
            if mv[6]==1:
                circle2 = plt.Circle((mv[0], mv[1]), self.MV_state.rMV, color='b', fill=False)

                ax.add_patch(circle2)

        # if self.MV_state.vog_points==True:
        #     ax.scatter(self.MV_state.vog_coordinates[:,1], self.MV_state.vog_coordinates[:,2], color='C0', s=2)
        #     print(len(self.MV_state.vog_coordinates[:,1]))




        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])


        plt.show()


    def plot_AF(self):
        fig = plt.figure(dpi=400)

        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax = fig.add_subplot(1, 1, 1)

        # if self.MV_state.record_AF == True:
        #     ax.plot(self.MV_state.t_AF[:,0], self.MV_state.t_AF[:,1], color='C0',linewidth=5)

        ax.set_ylim([0, 1])

        plt.show()




### Input Parameters
# L = 1, K = 1000, ka = 1.0, kd = 1.0, max_mv = 500, kon=1.0, kp=1.0,
# koff=1.0, kact = 1.0, sig_threshold=0.9, sigma=5.0, N=3, n_0=[1000,1],
# tau_step_=0.0001, T_decision=60, signal_activated = False, fp_activated=True,
# vog_points = False, record_AF = False, record_CF = False
FPT_array = np.zeros((1,5))
pPLUS_list = [0.001,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# constant parameters

kd = 1/7
ka = kd*2.0
max_mv = 200
kon = 1.0
kp = 1.0
koff = 1.0
kact = 1.0
sig_threshold = 0.99
sigma = 10.0
N=10
tau_step = 0.01
T_decision = 60



X = np.zeros((5,100))

#kd = 1/7.0
for pPLUS in range(len(pPLUS_list)):

    for ind in range(100):

        ### determine if agonist positive and how many agonists ###
        p_plus = pPLUS_list[pPLUS]
        p_h = 0.5
        s = np.random.binomial(1, p_plus, 1)[0]
        if s == 1:
            ### Sample n_agonist
            n_ag = np.random.randint(100, size=1)[0]
            #print(n_ag)
        else:
            n_ag = 0

        ### draw host populaitons ###
        n_h = np.random.binomial(20000, p_h, 1)[0]


        MVdiff = FlatInterfaceFP(ka = ka, kd = kd, max_mv = max_mv, kon=kon, kp=kp,
        koff=koff, kact = kact, sig_threshold=sig_threshold, sigma=sigma, N=N, n_0=[n_h,n_ag],
        tau_step_=tau_step, T_decision=T_decision, vog_points = False, record_AF = False, record_CF = False, signal_activated = True)

        MVdiff.reset()
        while MVdiff.done==False:
            MVdiff.tau_step()
            MVdiff.update_state()
        #FPT_array[0,ind] = FI.t
        print('Activation Status', MVdiff.activation)

        ### Determine Gamma Sample
        if n_ag > 0 and MVdiff.activation == 1:
            X[pPLUS,ind] = 1
        elif n_ag > 0 and MVdiff.activation == 0:
            X[pPLUS,ind] = 0
        elif n_ag == 0 and MVdiff.activation == 1:
            X[pPLUS,ind] = 0
        elif n_ag == 0 and MVdiff.activation == 0:
            X[pPLUS, ind] = 1

        print('Decision Accuracy Sample', X[ind])




    #print(str(kd_list[ind]))
    # data_file = h5py.File('MV_instantaneous_area_coverage.h5', 'a')
    # data_file.create_dataset('IAF_time_kd_'+str(kd_list[ind]), data=FI.MV_state.t_AF[:,0])
    # data_file.create_dataset('IAF_kd_'+str(kd_list[ind]), data=FI.MV_state.t_AF[:,1])
    # data_file.close()
    #MVdiff.plot_model_current_state()
    #MVdiff.plot_AF()
print(np.mean(X))
# print('FI' , FI.t)
# # print('antigen state',FI.antigen_state)
# print('mv state',FI.MV_state.mv_state)
# #FI.plot_model_current_state()
# print('AF', FI.MV_state.t_AF[:,1])
# data_file = h5py.File('FPT_array_tau_leap.h5', 'a')
# data_file.create_dataset('default_params_n_1000_1_koff_5', data=FPT_array)
# data_file.close()
