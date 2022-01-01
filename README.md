[//]: # (Image References)



# Project 3: Collaboration and Competition
This is a solution to this Udacity project:
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

### Introduction

In this project I worked with the (by-now-deprecated) (https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

!["Trained Agent"](./trained_agent_playing.gif)

(run example with the model I trained in 988 episodes)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

0. Prepare the environment as described in The [Udacity repo README](https://github.com/udacity/deep-reinforcement-learning#dependencies). Note that this repo uses pytorch 0.4.0 (!) which required me to have my machine support cuda 9:
```
conda install pytorch=0.4.0 cuda90 -c pytorch
conda install -c anaconda cudatoolkit==9.0
```

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. unzip the zip file in the repository root.

### Instructions

Run the [P3_solution](./P3_solution.ipynb) notebook to train a shared model for the two agents.

The notebook starts the environment, trains a model and saves the checkpoints and final models.

Use notebook [run_trained_model.ipynb](./run_trained_model.ipynb) to load the trained model and run it.

See the [Report.md](./Report.md) for information about the solution.

