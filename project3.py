"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
import csv
import cartpole
import lunarlander
import reinforce
import randomAgent

ENVIRONMENTS_MAP = {'cartpole' : cartpole.CartPole,
                    'lunarlander' : lunarlander.LunarLander }

# might add an A3C agent later on
AGENTS_MAP = {'random' : randomAgent.RandomAgent,
              'REINFORCE' : reinforce.ReinforceAgent }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent', choices=AGENTS_MAP.keys(), default='random', help='Can be random, or REINFORCE')
parser.add_argument('--env', choices=ENVIRONMENTS_MAP.keys(), default='cartpole', help='Can be cartpole or lunarlander')
parser.add_argument('--numEpisodes', type=int, default = 10, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Print more information.', action='store_true')
args = parser.parse_args()

agentFunc = AGENTS_MAP[args.agent]
envFunc = ENVIRONMENTS_MAP[args.env]

env = envFunc(args.numEpisodes, agentFunc, args.verbose)
