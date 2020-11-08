"""
randomAgent.py
Author: Michael Probst
Purpose: Implements a basic agent that picks moves randomly
"""

import random

class RandomAgent:
    def __init__(self):
        self.value = 0

    def GetBestAction(self, state):
        return random.choice([0,1])

    def SuggestAction(self, env, state):
        return env.action_space.sample()

    def UpdateModels(self, state, nextState, action, reward):
        return 0
