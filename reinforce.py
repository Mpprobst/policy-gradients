"""
reinforceAgent.py
Author: Michael Probst
Purpose: Implements an agent using the REINFORCE policy gradient algorithm
"""
import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.98
LEARNING_RATE = 0.3

class Net(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(Net, self).__init__()
        self.outputDims = outputDims
        self.inputDims = inputDims
        self.fc1 = nn.Linear(self.inputDims, 8)    #first layer
        self.fc2 = nn.Linear(8, 4)                #second layer
        self.fc3 = nn.Linear(4, self.outputDims)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    #Implements a feed forward network. state is a one hot vector indicating current state
    def Forward(self, state):
        x = F.logsigmoid(self.fc1(state))
        x = F.logsigmoid(self.fc2(x))
        actions = self.fc3(x)
        return actions

class ReinforceAgent:
    def __init__(self, env):
        self.epsilon = 1
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.net = Net(1, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.qTable = np.zeros([self.n_states, env.action_space.n])     # for the NN, this is for debugging only
        self.successCount = 0

    def GetBestAction(self, state):
        return 0

    def SuggestAction(self, env, state):
        return 0

    def UpdateModels(self, state, nextState, action, reward):
        return 0
