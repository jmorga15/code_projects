# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:21:10 2022

@author: jomor
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 19:23:44 2022

@author: jomor
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from scipy.spatial import distance_matrix
import h5py

""" This is a practice siumulation using classes to define 2 dimensional stochastic diffusion.
    Here, I also check the 2 dimensional result with theoretical results of 2D 
    diffusion with and without boundaries. Note: particles are sizeless """

class MV_diffusion():
    def __init__(self, rMV=50, L=1, K=1000, D=0.0001, n_molecules = 100, x0 = [50,50], tau_step_ = 0.01, tf = 200.0, ka = 1.0, kd = 1.0, random_walk_test = False, record_af = False, record_CF = False):
        ### Initial setup for 1 dimensional diffusion ###
        ### need the length of the domain, L ###
        ### need the width of each bin, h = L/K ###
        ### need to specify the diffusion constant and specify d = D/(h^2) ###
        ### need to specify the number of molecules in each bin at time t0, n0 ###
        self.K = K
        self.n_molecules = n_molecules
        self.D = D
        self.rMV = rMV
        self.tau_step_ = tau_step_
        self.tf = tf
        self.ka = ka
        self.kd = kd
        
        ### options ###
        self.record_af = record_af
        self.random_walk_test = random_walk_test
        self.record_cf = record_CF
        
        ### for recording area fraction mean ###
        self.mean_af = 0
        self.iteration = 0
        self.mean_af_count2 = 0
        
        
        
        
        ### in the tau-leaping strategy, going to track individual molecules and coordinates ###
        ### this will make much more sense when dealing with "MV" ###
        ### there are three indices for each molecule ###
        ### col0 -> binx #, col1 -> biny #, col2 -> isLeftTransferBlocked or col3 -> isRightTransferBlocked,  ###
        ### col4 -> isTopTransferBlocked or col5 -> isBottomTransferBlocked ###
        ### col1 and col 2 will be used to determine the transfer event ###
        #self.prev_molecule_state = np.zeros((6,1))
        self.molecule_state = np.zeros((7,1))
        
        ### width of bin ###
        h = L/K
        
        ### stochastic 1d diffusion rate ###
        self.d = D/(h**2)
        
        ### initial bin populations ###
        ### initial positions of sized particles ###
        ### no overlap is allowed ###
        
        x = np.random.randint(self.K+1, size=1)
        y = np.random.randint(self.K+1, size=1)
        
        self.molecule_state[0,:] =  x
        self.molecule_state[1,:] =  y
            
        ### if true, only diffusing a single mv molecule starting at center ###
        if self.random_walk_test == True:
            self.n_molecules = 1
            self.walk_array = np.zeros((1,2))
            self.molecule_state[0,:] =  int(self.K/2)
            self.molecule_state[1,:] =  int(self.K/2)
            
        if record_CF == True:
            ### this array will record the time and area fraction ###
            self.t_CF = np.array([0,0])
            
        ### before updating propensities, need to checked blocked status ###
        self.compute_isBlocked()
        
        ### initial propensities ###
        self.compute_propensity()
        
            
        ### initialize time ###
        self.t = 0
        
        ### the termination signal ###
        self.done = False
        
    def dist_check(self):
        
        #active_coordinates = self.molecule_state[0:2,:]
        
        ### get the distances from all Mv centers to all MV centers ###
        ### this will be a num_molecule x num_molecule matrix ###
        d = distance_matrix(self.molecule_state[0:2,:].transpose(), self.molecule_state[0:2,:].transpose())
        
        d = d<2*self.rMV
        
        #print(d)
        #print((np.sum(d,axis=1) > 1).shape)
        
        ### find the indices of mv that fail the distance check ###
        overlapping_indices = np.where(np.sum(d,axis=1) > 1)[0]
        
        
        ### check the radial distances from the centers of MV contacts
        #radial_distance = np.sqrt( (x - active_coordinates[0,:])**2 + (y - active_coordinates[1,:])**2 )
        
        return overlapping_indices
            
    def generate_and_dist_check(self, MVi, points):
        ### uniform in box ###
        
        ### choose next vogel point from points###
        x = points[MVi,0]
        y = points[MVi,1]
        
        # R = 1
        # u = np.random.uniform(0,1,1)
        # v = np.random.uniform(0,1,1)
        
        # radius = R*np.sqrt(v)
        # theta = 2*np.pi*u
         
         
        
        # x = radius * np.cos(theta)
        # y = radius * np.sin(theta)
        
        
        ### right now, this operates on a grid, so cast to integer ###
        x = int(x)
        y = int(y)
        
        ### get the mv x and y position in the grid ###
        active_coordinates = self.molecule_state[0:2,:]
        
        #print('active_coordinates', active_coordinates)
        
        ### check the radial distances from the centers of MV contacts
        radial_distance = np.sqrt( (x - active_coordinates[0,:])**2 + (y - active_coordinates[1,:])**2 )
        
        return x,y,radial_distance
        
        
        
    def MV_locations_init(self, vog_points):
        
        
        ### implement Vogel method for placing MV ###
        n_vp = vog_points
        
        ### radius of center of MV ###
        r_vp = np.sqrt(np.arange(n_vp)/n_vp)*700
        
        ### golden angle ###
        gold_angle = np.pi*(3-np.sqrt(5))
        
        ### angle of MV center ###
        theta = gold_angle*np.arange(n_vp)
        
        ### vector of length number of vogel points
        points = np.zeros((n_vp,2))
        
        ### get x and y coordinates ###
        points[:,0] = np.cos(theta)
        
        points[:,1] = np.sin(theta)
        
        points *= r_vp.reshape((n_vp,1))
        
        ### translate the points centered to current grid location ###
        points += 500
        
        ### this is an indicator if the algorithm failed to pack a MV ###
        packing_failure = False
        
        ### the index of current vog point where the algorithm is ### 
        ### trying to place a MV ###
        MVi = 1
        
        ### this is for verification test that the sized particle obeys properties of 2D random walk
        if self.random_walk_test == True:
        
            MVi = 1
        
        
        while packing_failure != True:
            
            
            ### reset the allowed attempts ###
            attempts = 0
            
            ### get new coordinates and distance ###
            x,y,radial_distance = self.generate_and_dist_check(MVi,points)
            
            ### attempted placement at vogel point MVi, no increment to next vogel point ###
            MVi += 1
            
            ### test packing failure ###
            ### fails when all vogel points have been tried ###
            if MVi >= n_vp:
                packing_failure = True
            
            if np.any(radial_distance < 2*self.rMV):
                while np.any(radial_distance < 2*self.rMV) and packing_failure == False:
                    
                    ### get new coordinates and distance ###
                    x,y,radial_distance = self.generate_and_dist_check(MVi,points)
                    
                    ### add 1 to attempts ###
                    MVi += 1
                    
                    ### test packing failure ###
                    ### fails when all vogel points have been tried ###
                    if MVi >= n_vp:
                        packing_failure = True
            

            
            ### the newly added MV state vector ###
            new_col = np.array([x, y, 0, 0, 0, 0, 0]).transpose()
            
            ### orient correctly to append as a new column ###
            new_col=np.reshape(new_col, (7,1))
            
            ### append to global MV_state
            if MVi < n_vp and (x > 1000 or x < 0 or y > 1000 or y < 0) == False:
                self.molecule_state = np.append(self.molecule_state, new_col, 1)
            
            
         
        ### this is used in order to avoid overlapping mv ###
        self.prev_molecule_state = np.copy(self.molecule_state[:,:])

        
    def compute_isBlocked(self):
        
        ### get indexes of molecules on left edge ###
        edge_indexes = np.where(self.molecule_state[0,:]==0)[0]
        
        self.molecule_state[2,edge_indexes] = 1 
        
        ### reset indexes of molecules not on edge ###
        edge_indexes = np.where(self.molecule_state[0,:]>0)[0]
        
        self.molecule_state[2,edge_indexes] = 0
                
        ### get indexes of molecules on right edge ###
        edge_indexes = np.where(self.molecule_state[0,:]==self.K)[0]
        
        self.molecule_state[3,edge_indexes] = 1 
        
        ### reset indexes of molecules not on edge ###
        edge_indexes = np.where(self.molecule_state[0,:]<self.K)[0]
        
        self.molecule_state[3,edge_indexes] = 0
        
        ### get indexes of molecules on top edge ###
        edge_indexes = np.where(self.molecule_state[1,:]==self.K)[0]
        
        self.molecule_state[4,edge_indexes] = 1 
        
        ### reset indexes of molecules not on edge ###
        edge_indexes = np.where(self.molecule_state[1,:]<self.K)[0]
        
        self.molecule_state[4,edge_indexes] = 0
        
        
        ### get indexes of molecules on top edge ###
        edge_indexes = np.where(self.molecule_state[1,:]==0)[0]
        
        self.molecule_state[5,edge_indexes] = 1 
        
        ### reset indexes of molecules not on edge ###
        edge_indexes = np.where(self.molecule_state[1,:]>0)[0]
        
        self.molecule_state[5,edge_indexes] = 0
        
        
        
        
        
        
    def compute_propensity(self):
        ### this is a method to compute the propensities of the diffusion reactions ###
        ### the method used is to compute the total propensity of all moleulces ###
        ### this is computed together regardless of a transfer left or right ###
        ### in other words, it is the propensity of "leaving bins" ###
        
        ### compute total propensities ###
        
        # number of MV that can make contact #
        self.n_canContact = np.sum(self.molecule_state[6,:]==0)
        
        # number of MV that can be removed #
        self.n_canRemove = np.sum(self.molecule_state[6,:]==1)
        
        # number of MV that can diffuse #
        self.n_canDiffuse = np.sum(self.molecule_state[6,:]==0)
        
        self.alpha0 = 4*self.d * self.n_canDiffuse + self.ka*self.n_canContact + self.kd*self.n_canRemove
        
        
    def tau_update(self):
        ### update tau with tau_leaping algorithm  ###
        self.t = self.t + self.tau_step_
        
        ### this is for recording the instantaneous area fraction ###
        if self.record_af == True:
        
            ### incrementing the iteration ###
            self.iteration += 1
            
            ### a conditional that computes the sum of active MV contacts every 100 iterations ###
            ### and appends to a list ###
            if self.t > 30.0 and self.iteration % 100 == 0:
                self.mean_af = self.mean_af + np.sum(self.molecule_state[6,:]==1)
                self.mean_af_count2 += 1
                
                print("time and active number mv",[self.t,self.mean_af / (self.mean_af_count2)])
    
    def update_state_transfers(self, diffuse_events, contact_events, remove_events):


        ### the events are {diffuse left, diffuse right, diffuse down, diffuse up,
        ### mv makes contact, mv disengages} ###
        
        "There is likely some inaccuracy here in the microvilli dynamics"
        " The inaccuracy has to do with the fixed tau update, and that actively \
             extended microvilli cannot diffuse. So, if, on a given tau update,\
             a microvilli both, diffuses and makes contact with the opposing cell,\
             then i must decide how to implement this."
        " The first thing I tried was to implement microvilli contact with priority.\
          But this produced an error in the instantaneous area fraction, when \
          compared with theoretical steady state values."
        " Thus, the current implementation is to allow both events to occur. \
            i.e., a microvilli can diffuse and make contact on the same tau update."
        
        " It seems the active mv count is short. This may be because a diffusing MV should not stop it fomr being active"
        " but here it does."
        
        
        
        ### get the indexes of molecules that can diffuse ###
        ### for this simple case, it is all the indexes that isInContact == 0 ###
        transfer_indexes = np.where(self.molecule_state[6,:]==0)[0]
        
        ### choose the indexes that diffuse ###
        transfer_indexes = random.sample(transfer_indexes.tolist(), diffuse_events)
        transfer_indexes = np.array(transfer_indexes)

        ### this array should contain the indices of MV in the state vector that are diffusing ###
        diffuse_indexes = transfer_indexes
        
        ### don't run diffuse_indexes is empty ###
        ### will produce error ###
        if len(diffuse_indexes)>0:

            ### now the diffusing events ###
            
            ### generate random vector to determine direction of transfers ###
            ### using len of diffuse index list ###
            r_direction = np.random.rand(len(diffuse_indexes))
            
            ### left movement ###
            #get left transfer attempt indexes from random vector (<0.5 is a left transfer)
            r_left_indexes = np.where(r_direction<0.25)[0]
            
            # get the corresponding indeices of molecules that attempt a left transfer #
            left_transfer_indexes = diffuse_indexes[r_left_indexes]
            
            if len(left_transfer_indexes)>0:
                
                # get indices of molecules that are not left blocked
                left_transfer_indexes2 = np.where( (self.molecule_state[2,left_transfer_indexes]==0) & (self.molecule_state[6,left_transfer_indexes]==0) )[0]
                
                # update state vector #
                self.molecule_state[0,left_transfer_indexes[left_transfer_indexes2]] = self.molecule_state[0,left_transfer_indexes[left_transfer_indexes2]] - 1
                
            
            ### right movement ###
            #get left transfer attempt indexes from random vector (<0.5 is a left transfer)
            r_right_indexes = np.where((r_direction>0.25) & (r_direction<0.5))[0]
            
            # get the corresponding indices of molecules that attempt a right transfer #
            right_transfer_indexes = diffuse_indexes[r_right_indexes]
            
            if len(right_transfer_indexes)>0:
            
                # get indices of molecules that are not right blocked
                right_transfer_indexes2 = np.where( (self.molecule_state[3,right_transfer_indexes]==0) & (self.molecule_state[6,right_transfer_indexes]==0) )[0]
                
                # update state vector #
                self.molecule_state[0,right_transfer_indexes[right_transfer_indexes2]] = self.molecule_state[0,right_transfer_indexes[right_transfer_indexes2]] + 1
                
            
            ### up movement ###
            #get left transfer attempt indexes from random vector (<0.5 is a left transfer)
            r_up_indexes = np.where((r_direction>0.5) & (r_direction<0.75))[0]
            
            # get the corresponding indeices of molecules that attempt a right transfer #
            up_transfer_indexes = diffuse_indexes[r_up_indexes]
            
            if len(up_transfer_indexes)>0:
                # get indices of molecules that are not up blocked
                up_transfer_indexes2 = np.where( (self.molecule_state[4,up_transfer_indexes]==0) & (self.molecule_state[6,up_transfer_indexes]==0) )[0]
                
                # update state vector #
                self.molecule_state[1,up_transfer_indexes[up_transfer_indexes2]] = self.molecule_state[1,up_transfer_indexes[up_transfer_indexes2]] + 1
            
            #print(r_right_indexes)
            
            ### down movement ###
            #get left transfer attempt indexes from random vector (<0.5 is a left transfer)
            r_down_indexes = np.where(r_direction>0.75)[0]
            
            # get the corresponding indeices of molecules that attempt a right transfer #
            down_transfer_indexes = diffuse_indexes[r_down_indexes]
            
            
            if len(down_transfer_indexes)>0:
                
                # get indices of molecules that are not down blocked
                down_transfer_indexes2 = np.where( (self.molecule_state[5,down_transfer_indexes]==0) & (self.molecule_state[6,down_transfer_indexes]==0) )[0]
                
                # update state vector #
                self.molecule_state[1,down_transfer_indexes[down_transfer_indexes2]] = self.molecule_state[1,down_transfer_indexes[down_transfer_indexes2]] - 1

        if remove_events > 0:
            ### get the indexes of the MV that are disengaging { isInContact {1->0} }###
            transfer_indexes = np.where(self.molecule_state[6,:]==1)[0]

            ### choose the indexes of mv that remove ###
            transfer_indexes = random.sample(transfer_indexes.tolist(), remove_events)
            transfer_indexes = np.array(transfer_indexes)
            
            ### this array should contain the indices of MV in the state vector that are making contact
            remove_indexes = np.copy(transfer_indexes)
            
            
            
          
        if contact_events > 0:
            ### lastly, I get the indexes of the MV that are making contact { isInContact {1->0} }###
            transfer_indexes = np.where(self.molecule_state[6,:]==0)[0]
            
            ### choose the indexes of mv that make contact ###
            transfer_indexes = random.sample(transfer_indexes.tolist(), contact_events)
            transfer_indexes = np.array(transfer_indexes)
            
            ### this array should contain the indices of MV in the state vector that are making contact
            contact_indexes = np.copy(transfer_indexes)

            
        " Note: I wait until now to update the contact and remove states "
        " This is because if I update one before selecting the population \
            of the other, it will affect the outcome. i.e., it would be possible \
            to both be removed and added in the same tau update. "
            
        if remove_events > 0:
            ### now update mv that are removed ###
            self.molecule_state[6,remove_indexes] = 0

        if contact_events > 0:
            ### now update mv that made contact ###
            self.molecule_state[6,contact_indexes] = 1
            
            ### vogel point captures ###
            if self.record_cf == True:
                self.vogel_points_capture()
            
        ### check for non-overlapping condition ###
        overlapping_indices = self.dist_check()
        
        ### this conditional is true if some microvilli diffused within the boundary of other microvilli ###
        if len(overlapping_indices)>0:
            
            ### the current solutions to this is to reset the microvilli to their previously \
            ### recorded state ###
            self.molecule_state[0:2,overlapping_indices] = self.prev_molecule_state[0:2,overlapping_indices]
        
        ### recorded to prevent overlapping microvilli ###
        self.prev_molecule_state[:,:] = self.molecule_state[:,:]
 
    def diffusive_transfer_step(self):
        
        ### how many diffusion events occurred ###
        diffuse_events = np.random.poisson((4*self.d)*self.n_canDiffuse*self.tau_step_, 1)
        
        ### check to make surre there are not more events than possible
        diffuse_events = np.min([diffuse_events[0],self.n_canDiffuse])
        
        ### how many diffusion or contact events occurred ###
        contact_events = np.random.poisson((self.ka)*self.n_canContact*self.tau_step_, 1)
        
        ### check to make surre there are not more events than possible
        contact_events = np.min([contact_events[0],self.n_canContact])
        
        ### how many removal events occurred ###
        remove_events = np.random.poisson((self.kd)*self.n_canRemove*self.tau_step_, 1)
        
        ### check to make surre there are not more events than possible
        remove_events = np.min([remove_events[0],self.n_canRemove])
        
        
        ### if using a really small tau_step_, this condition could run false 
        ### for the majority of updates. Might save some computation by 
        ### putting state updates in the conditional
        if diffuse_events > 0 or  remove_events > 0 or contact_events > 0:
            # indexes of antigen that can transition
            # determine which transfers occurred
            # need to have function that takes transfer events as input #
            # and will return specific (right or left) transfer event vector #
            self.update_state_transfers(diffuse_events, contact_events, remove_events)
            
            ### before updating propensities, need to check blocked status ###
            self.compute_isBlocked()
            
            ### compute new propensities ###
            self.compute_propensity()
            
        ### update simulation time ###
        self.tau_update()
        
        ### for random walk verification ###
        if self.random_walk_test == True:
            
            ### apending x,y, coordinates of the single mv in the simulation ###
            self.walk_array = np.append(self.walk_array, self.molecule_state[0:2,:].transpose(), 0)

        
        ### checking if time is greater than cell contact time ###
        if self.t > self.tf:
            self.done = True
            
        return self.done
        

    
    def plot_histogram(self):
        
        import matplotlib.pyplot as plt
        
        
        ### figure resolution ###
        fig = plt.figure(dpi=200) 
        
        ### setting equal height and width ###
        fig.set_figwidth(8)
        fig.set_figheight(8)
        ax = fig.add_subplot(1, 1, 1)
        
        ### this pulls up each individual mv local state vector ###
        for mv in self.molecule_state.transpose():
            if mv[0]>-1:
                
                ### plotting mv as a circle ###
                circle2 = plt.Circle((mv[0], mv[1]), self.rMV, color='b', fill=False)
                ax.add_patch(circle2)
        
        ### the width of the 2d domain is [0,K] ###
        ### right now, set to more so to observe outside of boundaries ###
        ax.set_xlim([0-500, self.K + 500])
        ax.set_ylim([0-500, self.K + 500])
        
        plt.show()
        
    " Left off here "
        
    def vogel_points_init(self, num_points):
        
        " This commented code is for spreading points in a disk "
        
        # self.tot_vogel_points = num_points
 
        # radius = np.sqrt(np.arange(self.tot_vogel_points) / float(self.tot_vogel_points))
         
        # golden_angle = np.pi * (3 - np.sqrt(5))
        # theta = golden_angle * np.arange(self.tot_vogel_points)
         
        # ### three dimensions, captured {0 or 1 -> captured}, and x,y coordinates
        # self.vog_coordinates = np.zeros((self.tot_vogel_points, 3))
        # self.vog_coordinates[:,1] = np.cos(theta)
        # self.vog_coordinates[:,2] = np.sin(theta)
        # self.vog_coordinates *= radius.reshape((self.tot_vogel_points, 1))
        
        " This is just the number of pixels in the grid (mv in a box)"
        #x_vec = np.arange(1000)
        self.cf_tracker = np.zeros((self.K, self.K))
        
    def vogel_points_capture(self):
        ### removing vogel points that have been captured by MV contacts ###
        
        rows,cols = np.where(self.cf_tracker==0) 
        #print(rows.shape)
        # coordinates_not_found = self.cf_tracker[rows][cols]
        # print(coordinates_not_found)
        
        
        active_MV = np.where(self.molecule_state[6,:]==1)[0]
        active_MV = self.molecule_state[:, active_MV]
        
        for ind in range(len(rows)):
            
            x = rows[ind]
            y = cols[ind]
            #print(x,y)
            ### distance from vog point to all mv centers ###
            radial_distance = np.sqrt( (x - active_MV[0,:])**2 + (y - active_MV[1,:])**2 )
            #print(radial_distance)
            ### find which indexes are less than rMV from a center ###
            if np.any(radial_distance < self.rMV):
                self.cf_tracker[x,y] = 1
                
        print("vogel capture",np.sum(self.cf_tracker==1))
            
            
        
        
        # for i in coordinates_not_found:
        #     print(i)
        
        " Vogel Point stuff "
        # ### get the distances from every vogel point that has not been found to every active MV ###
        # dm = distance_matrix(active_MV[:,1:],coordinates_not_found[:,1:])
        
        # ### find the vogel points that have been captured ###
        # dm = dm < self.rMV
        # dm = np.sum(dm,axis=0)
        # #print(len(dm))
        # newly_located_index = np.where(dm == 1)[0]
        
        # ### update the vogel point state vector ###
        # self.vog_coordinates[newly_located_index,0] = 1
        
        # self.vog_coordinates = np.delete(self.vog_coordinates, newly_located_index, 0)
                
        
        
    def plot_random_walk(self):
        
        import matplotlib.pyplot as plt
        
        ### this is a plot of the random walk path ###
        fig = plt.figure(dpi=200) 
        
        fig.set_figwidth(8)
        fig.set_figheight(8)
        
        ax = fig.add_subplot(1, 1, 1)
        
        
        ax.scatter(self.walk_array[:,0],self.walk_array[:,1])
     
                
        ax.set_xlim([0, self.K])
        ax.set_ylim([0, self.K])
        
        plt.show()
         
        

    
# trial = 0
# endpoint_array = np.zeros((2,10000))
# while trial < 1:
#     mv_diff = MV_diffusion( rMV=50, L=1, K=1000, D=0.0001, n_molecules = 200, x0 = [500,500], tau_step_ = 0.001, tf = 50.0, ka = 0.315368, kd = 1.0/7.0, random_walk_test = False, record_CF=False, record_af = False)  
#     mv_diff.MV_locations_init(1000)
    
    
#     mv_diff.plot_histogram()
    
#     mv_diff.vogel_points_init(10000)
    
#     mv_diff.diffusive_transfer_step()
    
#     #mv_diff.vogel_points_capture()
    
#     done = False
    
#     while done == False:
#         done = mv_diff.diffusive_transfer_step()
#         #mv_diff.vogel_points_capture()
    
#     endpoint_array[:,trial] = mv_diff.molecule_state[0:2,-1]
#     mv_diff.plot_histogram()
#     print(trial)
#     trial += 1
    
# print(np.sum(mv_diff.molecule_state[6,:]==1))
#mv_diff.plot_random_walk()




# data_file = h5py.File('2d_mvDiff_ka0_randomWalk.h5', 'a')
# data_file.create_dataset('tau_leaping_K_100_tau01_t10', data=endpoint_array)
# #data_file.create_dataset('t_end', data=10.0)
# data_file.close()          
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        