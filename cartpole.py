"""
cartpole.py
Author: Michael Probst
Purpose: Implement an interface to interact with the cartpole environment
"""

import gym
import csv
import numpy as np

TEST_INDEX = 10   # test after every 1000 training episodes
NUM_TESTS = 10

class CartPole:
    def __init__(self, numEpisodes, agentFunc, verbose):
        env = gym.make('CartPole-v0')
        self.verbose = verbose
        agent = agentFunc(env)
        filename = f'results/cartpole_{agent.name}.csv'
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            for i in range(1, numEpisodes):
                if i % TEST_INDEX == 0:
                    scores = []
                    losses = []
                    for t in range(NUM_TESTS):
                        value = self.Run(env, agent, isTest=True)
                        scores.append(value)
                        losses.append(agent.loss)
                    print(f'TEST {i / TEST_INDEX}:\t Avg Reward = {np.average(scores)}\tloss={np.average(losses)}')
                    writer.writerow([i / TEST_INDEX, np.average(scores)])
                else:
                    self.Run(env, agent, isTest=False)
                    agent.Learn()

        env.close()

    def Run(self, env, agent, isTest=False):
        # state is an array of 4 floats [position, velocity, pole angle, pole angular velocity]
        currentState = env.reset()

        done = False
        stepCount = 0
        score = 0
        #Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while stepCount < 200 and done == False:
            stepCount += 1

            actionToTake = 0
            if isTest:
                actionToTake = agent.GetBestAction(currentState)
            else:
                actionToTake = agent.SuggestAction(env, currentState)

            #Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
            nextState, reward, done, _ = env.step(actionToTake)

            score += reward

            if not isTest:
                agent.UpdateModels(currentState, nextState, actionToTake, reward)

            if isTest and self.verbose:
                #Render visualizes the environment
                env.render()

            currentState = nextState
        return score
