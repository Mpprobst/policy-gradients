"""
reinforceAgent.py
Author: Michael Probst
Purpose: Implements an agent using the REINFORCE policy gradient algorithm
"""
import gym
import numpy as np
import random
import neuralNet as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.98
LEARNING_RATE = 0.0075
MEMORY_SIZE = 10000

class ReinforceAgent:
    def __init__(self, env):
        self.name = 'REINFORCE'
        self.epsilon = 1
        self.stateDims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.net = nn.Net(self.stateDims, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.rewardMemory = []
        self.actionMemory = []
        self.successCount = 0

    def GetStateTensor(self, state):
        return T.tensor(state).to(self.net.device).float()

    def GetBestAction(self, state):
        stateTensor = self.GetStateTensor(state)
        probabilities = F.softmax(self.net.Forward(stateTensor))
        actionProbs = T.distributions.Categorical(probabilities)
        #action = actions[T.argmax(actions).item()].item()
        action = actionProbs.sample()
        logProbs = actionProbs.log_prob(action)
        self.actionMemory.append(logProbs)
        #options = []
        #for i in range(len(actions)):
        #    if actions[i].item() == action:
        #        options.append(i)
        return action.item()

    def EpsilonGreedy(self, env, state):
        # explore
        if random.random() < self.epsilon:
            return self.actionMemory.append()
        # exploit
        return self.GetBestAction(state)

    def SuggestAction(self, env, state):
        #return self.EpsilonGreedy(env, state)
        return self.GetBestAction(state)

    def UpdateModels(self, state, nextState, action, reward):
        self.rewardMemory.append(reward)

    def Learn(self):
        self.optimizer.zero_grad()

        Gt = np.zeros_like(self.rewardMemory, dtype=np.float64)
        for t in range(len(self.rewardMemory)):
            sum = 0
            discount = 1
            for k in range(t, len(self.rewardMemory)):
                sum += self.rewardMemory[k] * discount
                discount *= GAMMA
            Gt[t] = sum
        Gt = T.tensor(Gt, dtype=T.float).to(self.net.device)
        loss = 0
        baseline = np.average(self.rewardMemory)
        for g, logprob in zip(Gt, self.actionMemory):
            #print(f'f={g} b={baseline}')
            loss += (-g + baseline) * logprob
        loss.backward()
        self.optimizer.step()

        self.actionMemory = []
        self.rewardMemory = []
