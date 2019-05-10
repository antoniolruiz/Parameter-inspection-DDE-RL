### Parameter inspection for diversity-driven exploration in Reinforcement Learning.
Final project for the course of Big Data and Machine Learning at Columbia University.

Parameter inspection for diversity-driven exploration in reinforcement learning
Tobias Braun, Gerardo Antonio Lopez Ruiz, and Carlos Omar Pardo Gomez

### Overview

This repository contains the A2C and DQN algorithms with exploration strategies implemented to favor policies that are "different" from the previously seen. In order to apply this concept, it is possible to introduce a distance factor `D` that measures the dissimilarity of the new policy from the previous ones and subtract it from the standard loss function `L(π)`. Henceforth, we propose different distance metrics `D` as well as various distributions over the batch used for optimization andverify them empirically using A2C and DQN with and without exploration. Our research concluded that there are distances and distributions that can be used to improve the algorithms previously mentioned. For additional information on the project, please read our paper found in this repository. 

### Important dependencies
 - gym
 - Tensorflow >= 1.13.0
 - tensorflow_probability
 - Tested to work for Python 3.6

### Contents

- `a2c_exploration.py`: Advantage actor critic algorithm with exploration strategies implementation.
- `dqn_exploration.py`: Deep Q-Network algorithm with exploration strategies implementation.
- `testing.ipynb`: Main testing file. Gridsearch-like approach to finding optimal exploration strategies for CartPole-v0 and Acrobot-v1.
- `Parameter inspection for DDE in RL.pdf`: Final paper of project. 

### Instructions on Running 
Download the entire zip folder of our repo and run the `testing` jupyter notebook file. 
