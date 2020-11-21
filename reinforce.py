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
BATCH_SIZE = 5

class ReinforceAgent:
    def __init__(self, env, lr):
        self.name = 'REINFORCE'
        self.epsilon = 1
        self.stateDims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.net = nn.Net(self.stateDims, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.recentAction = None
        self.rewards = []
        self.actions = []
        self.actionMemory = []
        self.rewardMemory = []
        self.returnMemory = []
        self.longestEpisodeInBatch = 0
        self.losses = []


    def GetStateTensor(self, state):
        return T.tensor(state).to(self.net.device).float()

    def GetBestAction(self, state):
        stateTensor = self.GetStateTensor(state)
        probabilities = F.softmax(self.net.Forward(stateTensor), dim=0)
        actionProbs = T.distributions.Categorical(probabilities)
        action = actionProbs.sample()
        logProbs = actionProbs.log_prob(action)
        self.recentAction = logProbs
        return action.item()

    def SuggestAction(self, env, state):
        #return self.EpsilonGreedy(env, state)
        return self.GetBestAction(state)

    def UpdateModels(self, state, nextState, action, reward):
        self.actions.append(self.recentAction)
        self.rewards.append(reward)

    def Learn(self):
        Gt = np.zeros_like(self.rewards, dtype=np.float64)
        for t in range(len(self.rewards)):
            sum = 0
            discount = 1
            for k in range(t, len(self.rewards)):
                sum += self.rewards[k] * discount
                discount *= GAMMA
            Gt[t] = sum
        #Gt = T.tensor(Gt, dtype=T.float).to(self.net.device)

        self.actionMemory.append(self.actions)
        self.rewardMemory.append(self.rewards)
        self.returnMemory.append(Gt)

        if len(self.actions) > self.longestEpisodeInBatch:
            self.longestEpisodeInBatch = len(self.actions)

        #complete the batch
        if len(self.returnMemory) >= BATCH_SIZE:
            # pad memory with 0s
            for ep in range(len(self.returnMemory)):
                self.returnMemory[ep] = np.pad(self.returnMemory[ep], (0, self.longestEpisodeInBatch - len(self.returnMemory[ep])), 'constant')

            avgReturn = np.zeros([self.longestEpisodeInBatch])
            avgReturn = np.mean(self.returnMemory, axis=0)

            losses = []
            for batch in range(len(self.actionMemory)):
                #print(f'returns={Gt}')
                loss = 0
                baselineIndex = 0
                for g, logprob in zip(self.returnMemory[batch], self.actionMemory[batch]):
                    loss += (g - avgReturn[baselineIndex]) * -logprob
                    #print("(%.5f - %.5f) * %.5f" % (g, avgReturn[baselineIndex], -logprob))
                    baselineIndex += 1
                #loss /= len(self.actionMemory[batch])
                #print(f'LOSS:{loss}\n')
                losses.append(loss)

            loss = T.mean(T.stack(losses))
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

            self.actionMemory = []
            self.returnMemory = []
            self.longestEpisodeInBatch = 0

        self.actions = []
        self.rewards = []
