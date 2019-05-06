# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey
import matplotlib.pyplot as plt

screen_width = 600
screen_height = 400
velocity_max = 10
eta = 0.20       # Learning Rate
gamma = 0.9      # Discount Factor
explore = 0.10   # Probabillity of exploring

training_iters = 500

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.grav = None

        self.explore = explore
        self.learn_on = True
        self.iter = 0
        self.switch_explore = 150
        
        # Create an array to map states and action to value
        # [action, gravity, dist, top_dist, bot_dist, monkey_vel]
        self.w_bin = 16
        self.h_bin = 16
        self.v_bin = 12

        self.act_vec  = np.array([0,1])
        self.grav_vec = np.array([1,4])
        self.dist_vec = np.linspace(0,screen_width/2, self.w_bin)
        self.top_vec  = np.linspace(0,screen_height, self.h_bin)
        self.bot_vec  = np.linspace(0,screen_height, self.h_bin)
        self.vel_vec  = np.linspace(-velocity_max/2,velocity_max/2, self.v_bin)

        self.Q = np.zeros((self.act_vec.size, self.grav_vec.size, self.dist_vec.size, self.top_vec.size, self.bot_vec.size, self.vel_vec.size))
        self.Q_confidence = np.ones((self.act_vec.size, self.grav_vec.size, self.dist_vec.size, self.top_vec.size, self.bot_vec.size, self.vel_vec.size))

        print(self.Q.shape)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.grav = None
        self.iter += 1

        if self.iter>self.switch_explore:
        	self.explore = 0.00000005

        print(self.explore)
        

    def discretize_state(self, action, state):
        # [action, gravity, dist, top_dist, bot_dist, monkey_vel]
        a_idx=[]

        a_idx.append(self.findNearestIdx(action,self.act_vec))
        a_idx.append(self.findNearestIdx(self.grav,self.grav_vec))
        a_idx.append(self.findNearestIdx(state['tree']['dist'],self.dist_vec))
        a_idx.append(self.findNearestIdx(state['tree']['top']-state['monkey']['top'],self.top_vec))
        a_idx.append(self.findNearestIdx(state['tree']['bot']-state['monkey']['bot'],self.bot_vec))
        a_idx.append(self.findNearestIdx(state['monkey']['vel'],self.vel_vec))

        a_idx = np.array(a_idx)
        return a_idx


    def findNearestIdx(self,state,vector):
        idx = (np.abs(vector-state)).argmin()
        return idx

    def getQ(self, test):
        return self.Q[test[0],test[1],test[2],test[3],test[4],test[5]]

    def getQc(self, test):
        return self.Q_confidence[test[0],test[1],test[2],test[3],test[4],test[5]]

    def putQ(self, test, val):
        self.Q[test[0],test[1],test[2],test[3],test[4],test[5]] = val
        self.Q_confidence[test[0],test[1],test[2],test[3],test[4],test[5]] += 1


    def action_callback(self, state):
        new_state  = state
    
        # If we have never taken an action, don't jump
        if self.last_state is None:
            new_action = 0

        # If we don't know what gravity is yet, calculate it.
        # This works becasue we force the monkey not to jump in the first frame
        if self.grav is None:
            if self.last_state is not None: 
                self.grav = -(new_state['monkey']['vel'] - self.last_state['monkey']['vel'])
                #print('Gravity is: %0.5f'%(self.grav))

        # Once we know gravity, start doing actual learning
        if self.grav is not None:
            if self.learn_on:
                dis_state_old = self.discretize_state(self.last_action, self.last_state)
                confidence = self.getQc(dis_state_old)
                # epsilon greedy decision
                if npr.rand() < (1-self.explore/confidence):
                    new_action = np.argmax([self.getQ(self.discretize_state(0, new_state)), self.getQ(self.discretize_state(1, new_state))])
                else:
                    new_action = npr.choice([1,0])
                # Update Q
                Q_old = self.getQ(dis_state_old)
                Q_max = np.max([self.getQ(self.discretize_state(0, new_state)), self.getQ(self.discretize_state(1, new_state))])
                self.putQ(dis_state_old, (1-eta)*Q_old + eta*(self.last_reward+gamma*Q_max))

            else:
                new_action = np.argmax([self.getQ(self.discretize_state(0, new_state)), self.getQ(self.discretize_state(1, new_state))])

        
        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action


    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        #print(learner.Q)

        #print(np.where(learner.Q!=0))

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()
	agent.switch_explore=training_iters/2

	# Empty list to save history.
	hist = []

	# Run games. If the last input = 0, the game runs as fast as possible.
	run_games(agent, hist, training_iters, 0)
	np.save('hist_train',np.array(hist))
	print('\n'+"TRAINING:_____________")
	print("AVERAGE SCORE: %0.3f"%(np.mean(hist)))
	print("MAX SCORE: %0.3f"%(np.max(hist)))

	# Test the result!
	hist = []
	eta=0
	agent.learn_on = False
	run_games(agent, hist, 50, 0)
	print('\n'+"FINAL TEST:_____________")
	print("AVERAGE SCORE: %0.3f"%(np.mean(hist)))
	print("MAX SCORE: %0.3f"%(np.max(hist)))

	#Save history. 
	np.save('hist_test',np.array(hist))


