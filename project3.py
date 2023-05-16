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

ENVIRONMENTS_MAP = {'cartpole' : cartpole.CartPole,
                    'lunarlander' : lunarlander.LunarLander }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--env', choices=ENVIRONMENTS_MAP.keys(), default='cartpole', help='Can be cartpole or lunarlander')
parser.add_argument('--numEpisodes', type=int, default = 500, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
args = parser.parse_args()

envFunc = ENVIRONMENTS_MAP[args.env]

env = envFunc(args.numEpisodes, reinforce.ReinforceAgent, args.verbose)
