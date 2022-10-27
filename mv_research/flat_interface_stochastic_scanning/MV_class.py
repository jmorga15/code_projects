# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:23:19 2022

@author: jomor
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class MV_scan():
    def __init__(self, ka=1.0, kd=1.0, ks=1.0, ku=1.0, rMV=0.05, max_mv=50, vog_points = False, record_AF = False, record_CF = False):

        ### bookkeeping ###
        self.vog_points = vog_points
        self.record_AF = record_AF
        self.record_CF = record_CF

        ### mv parameters ###
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ku = ku
        self.rMV = rMV
        self.max_mv = max_mv

        # MV state_vector (state, x_location, y_location)
        # state is the mv state {0->inactive, 1-> active and placed, 2-> stabilized}
        self.mv_state = np.zeros((self.max_mv,3),dtype=np.float64)

        # active MV locations #
        ### initialy there are no active MV ###
        #self.antigen_locations()

        # max attempts at MV placement #
        self.max_attempts = 500

        if vog_points == True:
            self.vog_num_points = 10000
            self.vogel_points_init(self.vog_num_points)

        if record_AF == True:
            ### this array will record the time and area fraction ###
            self.t_AF = np.array([0,0])

        if record_CF == True:
            ### this array will record the time and area fraction ###
            self.t_CF = np.array([0,0])



    def generate_MV_coordinates_step(self,num_to_add):

        # cell contact area radius #
        R = 1

        ### generate the coordinates for the MV to be added ###
        u = np.random.uniform(0,1,num_to_add)
        v = np.random.uniform(0,1,num_to_add)

        radius = (R-self.rMV)*np.sqrt(v)
        theta = 2*np.pi*u

        new_coordinates = np.zeros((num_to_add, 3))
        new_coordinates[:,1] = radius * np.cos(theta)
        new_coordinates[:,2] = radius * np.sin(theta)


        # if np.sum(self.mv_state[:,0]==0) >= 1 and num_to_add >= 1:
        #     new_coordinates = self.distance_check(new_coordinates)

        return new_coordinates

    def generate_MV_coordinates_init(self,num_to_add):

        # cell contact area radius #
        R = 1

        ### generate the coordinates for the MV to be added ###
        u = np.random.uniform(0,1,num_to_add)
        v = np.random.uniform(0,1,num_to_add)

        radius = (R-self.rMV)*np.sqrt(v)
        theta = 2*np.pi*u

        new_coordinates = np.zeros((num_to_add, 3))
        new_coordinates[:,1] = radius * np.cos(theta)
        new_coordinates[:,2] = radius * np.sin(theta)


        if np.sum(self.mv_state[:,0]==0) >= 1 and num_to_add >= 1:
            new_coordinates = self.distance_check(new_coordinates)

        return new_coordinates

    def distance_check(self,new_coordinates):

        ### get the indexes of active MV contacts ###
        active_mv_indexes = np.where(self.mv_state[:,0] >= 1)[0]
        active_coordinates = self.mv_state[active_mv_indexes,1:]

        ### index is the current new coordinate index we are checking ###
        index = 0
        for x,y in new_coordinates[:,1:]:

            if index < 1000:

                #print(x,y)
                #print('what the fuck!!!!!', new_coordinates[index,0] )
                #print('active coordinates',active_coordinates)

                ### check the radial distances from the centers of MV contacts
                radial_distance_old = np.sqrt( (x - active_coordinates[:,0])**2 + (y - active_coordinates[:,1])**2 )

                ### check the radial distances from the centers of new MV contacts
                radial_distance_new = np.sqrt( (x - new_coordinates[0:index+1,1])**2 + (y - new_coordinates[0:index+1,2])**2 )


                #print('radial distance',np.any(radial_distance_new < 2*self.rMV))
                # check if MV contacts are overlapping #
                max_attempts = 0

                ### the conditional is dumb and i am dumb.
                if np.any(radial_distance_old < 2*self.rMV) or np.any(radial_distance_new < 2*self.rMV):
                    # there is a limit to how many times to try and place a mv contact #
                    #max_attempts = 0

                    ### np.sum(radial_distance_new < 2*self.rMV)>1) is >1 because this distance check includes the MV in question, so itself ###
                    while (np.any(radial_distance_old < 2*self.rMV) or np.sum(radial_distance_new < 2*self.rMV)>1) and max_attempts < self.max_attempts:
                        #print('radial distance',np.any(radial_distance < 2*self.rMV))

                        ### try a new coordinate pair ###
                        new_coordinates[index,:] = self.generate_MV_coordinates_step(1)

                        x = new_coordinates[index,1]
                        y = new_coordinates[index,2]

                        ### check the radial distances from the centers of MV contacts
                        radial_distance_old = np.sqrt( (x - active_coordinates[:,0])**2 + (y - active_coordinates[:,1])**2 )

                        ### check the radial distances from the centers of new MV contacts
                        radial_distance_new = np.sqrt( (x - new_coordinates[0:index+1,1])**2 + (y - new_coordinates[0:index+1,2])**2 )

                        max_attempts += 1
                    #print('radial distance',np.any(radial_distance_new < 2*self.rMV))
                    if max_attempts >= self.max_attempts:
                        #print("reached max attempts")
                        ### cannot find a place for mv given max_attempt tries ###
                        #print(new_coordinates)
                        assert new_coordinates[index,0] == 0, "Invalid flag, why is this not 0?"

                        new_coordinates[index,0] = 0

                        index = 1000

                        ### have to count back the index here ###
                        #index += -1

                    else:
                        new_coordinates[index,0] = 1


                index += 1

        #print('new_coordinates', new_coordinates)

        return new_coordinates



    def add_MV(self, num_to_add):
        ### uniform in disk ###
        # disk radius #
        R = 1

        ### get the coordinates of MV
        current_coordinates = self.mv_state[:,:]

        ### the total number that can be added ###
        MV_available = np.sum(self.mv_state[:,0]==0)

        num_to_add = np.min([MV_available, num_to_add])

        #print("num to add", num_to_add)

        if num_to_add > 0:

            ### generate the coordinates for the MV to be added ###
            # u = np.random.uniform(0,1,num_to_add)
            # v = np.random.uniform(0,1,num_to_add)

            # radius = (R-self.rMV)*np.sqrt(v)
            # theta = 2*np.pi*u

            # new_coordinates = np.zeros((num_to_add, 2))
            # new_coordinates[:,0] = radius * np.cos(theta)
            # new_coordinates[:,1] = radius * np.sin(theta)
            new_coordinates = self.generate_MV_coordinates_init(num_to_add)


            ### choose which MV got added ###
            inactive_mv_indexes = np.where(current_coordinates[:,0] == 0)[0]
            transitioning_MV_indexes = random.sample(inactive_mv_indexes.tolist(), num_to_add)

            #print(transitioning_MV_indexes)

            current_coordinates[transitioning_MV_indexes,:] = new_coordinates

            #print(current_coordinates)

            ### update state coordinates ###
            self.mv_state[:,:] = current_coordinates

            ### switch new MV contacts to active ###
            #self.mv_state[transitioning_MV_indexes,0] = 1

            #print(self.mv_state)



    # def newly_covered_antigen(self,antigen_state):
    #     ### this method finds the antigen that are covered and returns the new antigen state
    #     ### get the indexes of active MV contacts ###
    #     active_mv_indexes = np.where(self.mv_state[:,0] == 1)[0]
    #     active_coordinates = self.mv_state[active_mv_indexes,1:]

    #     ### index is the current antigen coordinate index we are checking ###
    #     index = 0
    #     for x,y in antigen_state[:,2:]:
    #         #print(x,y)

    #         ### check the radial distances from the centers of MV contacts to the antigen in question
    #         radial_distance = np.sqrt( (x - active_coordinates[:,0])**2 + (y - active_coordinates[:,1])**2 )

    #         # check if any MV cover antigen are overlapping #
    #         #print( 'radial distance',np.any(radial_distance < 2*self.rMV) )
    #         if np.any(radial_distance < self.rMV):

    #             if antigen_state[index,0]==-1:
    #                 # update the antigen state (-1 -> 0) #

    #                 # check that the antigen state was -1
    #                 #assert antigen_state[index,0] == -1, "Invalid flag, why is this not -1?"

    #                 antigen_state[index,0] = 0



    #         index += 1

    #     return antigen_state

    def stabilize_MV(self, num_to_stabilize, transition_indexes_stab):
        ### get the coordinates of MV
        current_coordinates = self.mv_state[:,:]

        ### the total number that can be stabilized ###
        MV_available = len(transition_indexes_stab)

        num_to_stabilize = np.min([MV_available, num_to_stabilize])

        #print("num to add", num_to_remove)

        if num_to_stabilize > 0:

            ### generate the coordinates for the MV to be added ###
            # u = np.random.uniform(0,1,num_to_add)
            # v = np.random.uniform(0,1,num_to_add)

            # radius = (R-self.rMV)*np.sqrt(v)
            # theta = 2*np.pi*u

            # new_coordinates = np.zeros((num_to_add, 2))
            # new_coordinates[:,0] = radius * np.cos(theta)
            # new_coordinates[:,1] = radius * np.sin(theta)
            #new_coordinates = self.generate_MV_coordinates(num_to_add)


            ### choose which MV got removed ###
            #active_mv_indexes = np.where(current_coordinates[:,0] == 1)[0]
            transition_indexes_stab = transition_indexes_stab[0]
            #print(type(transition_indexes_destab.tolist()))
            #print(num_to_destabilize)
            transitioning_MV_indexes = random.sample(transition_indexes_stab.tolist(), int(num_to_stabilize))

            #print('transition',transitioning_MV_indexes)

            current_coordinates[transitioning_MV_indexes,0] = 2

            #print('current coordinates',current_coordinates)

            ### update state coordinates ###
            self.mv_state[:,:] = current_coordinates

            ### switch new MV contacts to active ###
            #self.mv_state[transitioning_MV_indexes,0] = 1

            #print(self.mv_state)
            

    def destabilize_MV(self, num_to_destabilize, transition_indexes_destab):
        ### get the coordinates of MV
        current_coordinates = self.mv_state[:,:]

        ### the total number that can be removed ###
        MV_available = len(transition_indexes_destab)

        num_to_destabilize = np.min([MV_available, num_to_destabilize])

        #print("num to add", num_to_remove)

        if num_to_destabilize > 0:

            ### generate the coordinates for the MV to be added ###
            # u = np.random.uniform(0,1,num_to_add)
            # v = np.random.uniform(0,1,num_to_add)

            # radius = (R-self.rMV)*np.sqrt(v)
            # theta = 2*np.pi*u

            # new_coordinates = np.zeros((num_to_add, 2))
            # new_coordinates[:,0] = radius * np.cos(theta)
            # new_coordinates[:,1] = radius * np.sin(theta)
            #new_coordinates = self.generate_MV_coordinates(num_to_add)


            ### choose which MV got removed ###
            #active_mv_indexes = np.where(current_coordinates[:,0] == 1)[0]
            transition_indexes_destab = transition_indexes_destab[0]
            #print(type(transition_indexes_destab.tolist()))
            #print(num_to_destabilize)
            transitioning_MV_indexes = random.sample(transition_indexes_destab.tolist(), int(num_to_destabilize))

            #print('transition',transitioning_MV_indexes)

            current_coordinates[transitioning_MV_indexes,0] = 1

            #print('current coordinates',current_coordinates)

            ### update state coordinates ###
            self.mv_state[:,:] = current_coordinates

            ### switch new MV contacts to active ###
            #self.mv_state[transitioning_MV_indexes,0] = 1

            #print(self.mv_state)
            
   

    def remove_MV(self, num_to_remove):
        ### get the coordinates of MV
        current_coordinates = self.mv_state[:,:]

        ### the total number that can be removed ###
        MV_available = np.sum( (self.mv_state[:,0]==1) )

        num_to_remove = np.min([MV_available, num_to_remove])

        #print("num to add", num_to_remove)

        if num_to_remove > 0:

            ### generate the coordinates for the MV to be added ###
            # u = np.random.uniform(0,1,num_to_add)
            # v = np.random.uniform(0,1,num_to_add)

            # radius = (R-self.rMV)*np.sqrt(v)
            # theta = 2*np.pi*u

            # new_coordinates = np.zeros((num_to_add, 2))
            # new_coordinates[:,0] = radius * np.cos(theta)
            # new_coordinates[:,1] = radius * np.sin(theta)
            #new_coordinates = self.generate_MV_coordinates(num_to_add)


            ### choose which MV got removed ###
            active_mv_indexes = np.where(current_coordinates[:,0] == 1)[0]
            transitioning_MV_indexes = random.sample(active_mv_indexes.tolist(), num_to_remove)

            #print('transition',transitioning_MV_indexes)

            current_coordinates[transitioning_MV_indexes,:] = 0

            #print('current coordinates',current_coordinates)

            ### update state coordinates ###
            self.mv_state[:,:] = current_coordinates

            ### switch new MV contacts to active ###
            #self.mv_state[transitioning_MV_indexes,0] = 1

            #print(self.mv_state)

    def newly_covered_antigen(self,antigen_state):
        ### this method finds the antigen that are covered and returns the new antigen state
        ### get the indexes of active MV contacts ###
        active_mv_indexes = np.where(self.mv_state[:,0] == 1)[0]
        active_coordinates = self.mv_state[active_mv_indexes,1:]

        ### index is the current antigen coordinate index we are checking ###
        index = 0
        for x,y in antigen_state[:,2:]:
            #print(x,y)

            ### check the radial distances from the centers of MV contacts to the antigen in question
            radial_distance = np.sqrt( (x - active_coordinates[:,0])**2 + (y - active_coordinates[:,1])**2 )

            # check if any MV cover antigen  #
            if np.any(radial_distance < self.rMV):

                if antigen_state[index,0]==-1:
                    # update the antigen state (-1 -> 0) #

                    # check that the antigen state was -1
                    #assert antigen_state[index,0] == -1, "Invalid flag, why is this not -1?"

                    antigen_state[index,0] = 0

            elif antigen_state[index,0]==0:

                antigen_state[index,0] = -1





            index += 1

        return antigen_state

    def vogel_points_init(self, num_points):

        self.tot_vogel_points = num_points

        radius = np.sqrt(np.arange(self.tot_vogel_points) / float(self.tot_vogel_points))

        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(self.tot_vogel_points)

        ### three dimensions, captured {0 or 1 -> captured}, and x,y coordinates
        self.vog_coordinates = np.zeros((self.tot_vogel_points, 3))
        self.vog_coordinates[:,1] = np.cos(theta)
        self.vog_coordinates[:,2] = np.sin(theta)
        self.vog_coordinates *= radius.reshape((self.tot_vogel_points, 1))

    def vogel_points_capture(self):
        ### removing vogel points that have been captured by MV contacts ###

        coordinates_not_found = np.where(self.vog_coordinates[:,0]==0)[0]
        coordinates_not_found = self.vog_coordinates[coordinates_not_found,:]

        active_MV = np.where(self.mv_state[:,0]==1)[0]
        active_MV = self.mv_state[active_MV,:]

        ### get the distances from every vogel point that has not been found to every active MV ###
        dm = distance_matrix(active_MV[:,1:],coordinates_not_found[:,1:])

        ### find the vogel points that have been captured ###
        dm = dm < self.rMV
        dm = np.sum(dm,axis=0)
        #print(len(dm))
        newly_located_index = np.where(dm == 1)[0]

        ### update the vogel point state vector ###
        self.vog_coordinates[newly_located_index,0] = 1

        self.vog_coordinates = np.delete(self.vog_coordinates, newly_located_index, 0)
        #print(np.sum(self.vog_coordinates[:,0]==1))

    def get_cumulative_area_fraction(self, t):
        ### getting the area coverage for active MV ###
        CF = 1-len(self.vog_coordinates)/self.vog_num_points

        if self.record_CF == True:
            self.t_CF = np.vstack([self.t_CF,[t,CF]])

    def get_instantaneous_area_fraction(self, t):
        ### getting the area coverage for active MV ###
        AF = np.sum(self.mv_state[:,0]>=1) * self.rMV **2

        if self.record_AF == True:
            self.t_AF = np.vstack([self.t_AF,[t,AF]])

        #return AF

    def plot_disk(self):

        circle2 = plt.Circle((0, 0), 1, color='b', fill=False)


        fig = plt.figure(dpi=400)

        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax = fig.add_subplot(1, 1, 1)

        #circle2 = plt.Circle((mv[1], mv[2]), self.rMV, color='b', fill=False)

        ax.add_patch(circle2)

        for mv in self.mv_state:
            if mv[0]==1:
                circle2 = plt.Circle((mv[1], mv[2]), self.rMV, color='b', fill=False)

                ax.add_patch(circle2)

            if mv[0]==2:
                circle2 = plt.Circle((mv[1], mv[2]), self.rMV, color='r', fill=False)

                ax.add_patch(circle2)

        if self.vog_points == True:
            undiscovered_vog_points = np.where(self.vog_coordinates[:,0]==0)[0]
            undiscovered_vog_points = self.vog_coordinates[undiscovered_vog_points,1:]
            ax.scatter(undiscovered_vog_points[:,0], undiscovered_vog_points[:,1], color='C0', s=2)


        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])


        #color = np.sqrt((self.antigen_state[:,2:]**2).sum(axis = 1))/np.sqrt(2.0)
        #rgb = plt.get_cmap('jet')(color)
        #ax.scatter(self.antigen_state[:,2], self.antigen_state[:,3], color = rgb)
        plt.show()


# mv_component = MV_scan(vog_points = True)
# mv_component.add_MV(10)
# mv_component.vogel_points_capture()
# mv_component.plot_disk()
# # #print(mv_component.vog_coordinates.shape)
# mv_component.add_MV(10)
# mv_component.vogel_points_capture()
# mv_component.plot_disk()

# mv_component.add_MV(10)
# mv_component.vogel_points_capture()
# mv_component.plot_disk()
#mv_component.vogel_points_capture()

# mv_component.plot_disk()
