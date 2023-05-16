# Policy Gradients
Author: Michael Probst
Date: 11/21/2020

## Overview
This repository contains a solution to Problem Set 3 for CS660 - Sequential Decision Making. The purpose of this project was to solve the open ai gym Cartpole and Lunar Lander environments using an advanced reinforcement learning technique.

Files include: project3.py, cartpole.py, lunarlander.py, neurlaNet.py, reinforce.py, and writeup.pdf

## Runing This Program
The program is controlled by project3.py which utilizes various files based on which technique is used to solve the problem. 

To run the program, simply enter `python project3.py` 
However, several other arguements can be passed to grant more control over the program which are: --env, --numEpisodes, and --verbose

For more information on each of the arguments, enter `python project3.py --help`

The results of the program are saved in the results/ directory in a csv file.

## Quick Commands
For testing Cartpole: `python project3.py --env cartpole --numEpisodes 500`

For testing Lunar Lander: `python project3.py --env lunarlander --numEpisodes 5000`


