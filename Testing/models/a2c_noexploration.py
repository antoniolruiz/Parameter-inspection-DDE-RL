import seaborn as sns
import tensorflow as tf
import tensorflow_probability.python.distributions as tfp
from collections import deque
import random as rand
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym


class Policy(object):
    def __init__(self, obssize, actsize, sess, optimizer):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # BUILD PREDICTION GRAPH
        # build the input
        state = tf.placeholder(tf.float32, [None, obssize])
        
        initializer = tf.contrib.layers.xavier_initializer()
        W1 = tf.Variable(initializer([obssize, 50]))
        b1 = tf.Variable(initializer([50]))
        W2 = tf.Variable(initializer([50, actsize]))
        b2 = tf.Variable(initializer([actsize]))
        
        h1 = tf.nn.sigmoid(tf.matmul(state, W1) + b1)
        h = tf.matmul(h1, W2) + b2
        
        prob = tf.nn.softmax(h, axis=1)  # prob is of shape [None, actsize], axis should be the one for each sample = axis = 1
        
        # BUILD LOSS 
        Q_estimate = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])
        
        actions_one_hot = tf.one_hot(actions, actsize) # actions_one_hot will be a matrix of shape(n,2)
        action_probabilities = tf.reduce_sum(prob * actions_one_hot, axis=1) # this will connect the action that was 
                                                                             # actually played to to its probability
        surrogate_loss = -tf.reduce_mean(tf.log(action_probabilities)*Q_estimate)
        self.train_op = optimizer.minimize(surrogate_loss)
        
        # some bookkeeping
        self.state = state
        self.prob = prob
        self.actions = actions
        self.Q_estimate = Q_estimate
        self.loss = surrogate_loss
        self.optimizer = optimizer
        self.sess = sess
    
    def compute_prob(self, states):
        """
        compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        return self.sess.run(self.prob, feed_dict={self.state:states})

    def train(self, states, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        self.sess.run(self.train_op, feed_dict={self.state:states, self.actions:actions, self.Q_estimate:Qs})   
        
        
# define value function as a class
class ValueFunction(object):
    def __init__(self, obssize, sess, optimizer):
        """
        obssize: size of states
        """
        state = tf.placeholder(tf.float32, [None, obssize])
        
        initializer = tf.contrib.layers.xavier_initializer()
        W1 = tf.Variable(initializer([obssize, 50]))
        b1 = tf.Variable(initializer([50]))
        W2 = tf.Variable(initializer([50, 1]))
        b2 = tf.Variable(initializer([1]))
        
        h1 = tf.nn.sigmoid(tf.matmul(state, W1) + b1) # dim = [num_of_samples, 30]
        predictions = tf.matmul(h1, W2) + b2 # dim = [num_of_samples, 1]
        
        self.predictions = predictions
        self.obssize = obssize
        self.optimizer = optimizer
        self.sess = sess
        self.state = state
        
        
        targets = tf.placeholder(tf.float32, [None])# first None because the number of sampled trajectories
        # is not known beforehand and furthermore, the length of each trajectory might vary and is not constant. 
        # But in the end it is just a list of unkown size where each element is a number representing the value
        # for the corresponding state in self.states
        error = predictions - targets
        loss = tf.reduce_mean(tf.square(error))
        
        self.targets = targets
        self.train_op = optimizer.minimize(loss)

    def compute_values(self, states):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        return self.sess.run(self.predictions, feed_dict={self.state:states})

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        return self.sess.run(self.train_op, feed_dict={self.state:states, self.targets:targets}) 
    
    
def discounted_rewards(r, gamma):
    """ 
    take 1D float array of rewards and compute discounted reward 
    returns a list where the first element is the complete discounted reward for the whole trajectory (already summed),
    the second element is the complete discounted reward for the trajectory starting at t=1 and so on...
    """
    
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def a2c_noexp(episodes = 100):
    # tune/add hyper-parameters numtrajs too high ? code is very slow
    # parameter initializations
    alpha = 1e-3  # learning rate for PG
    beta = 1e-3  # learning rate for baseline
    numtrajs = 15  # num of trajecories to collect at each iteration 
    iterations = 100  # total num of iterations
    envname = "CartPole-v1"  # environment name
    gamma = .99  # discount
    #episodes = 1

    # initialize environment
    env = gym.make(envname)
    obssize = env.observation_space.low.size
    actsize = env.action_space.n


    # sess
    sess = tf.Session()

    # optimizer
    optimizer_p = tf.train.AdamOptimizer(alpha)
    optimizer_v = tf.train.AdamOptimizer(beta)

    # initialize networks
    actor = Policy(obssize, actsize, sess, optimizer_p)  # policy initialization
    baseline = ValueFunction(obssize, sess, optimizer_v)  # baseline initialization

    # initialize tensorflow graphs
    sess.run(tf.global_variables_initializer())
    #replay buffer
    # main iteration
    for ite in range(episodes):    

        # trajs records for batch update
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages (to update policy)
        VAL = []  # value functions (to update baseline)

        for num in range(numtrajs):
            # record for each episode
            obss = []  # observations
            acts = []   # actions
            rews = []  # instant rewards

            obs = env.reset()
            done = False

            while not done:

                prob = actor.compute_prob(np.expand_dims(obs,0))
                action = np.random.choice(actsize, p=prob.flatten(), size=1)
                newobs, reward, done, _ = env.step(action[0])

                # record
                obss.append(obs)
                acts.append(action[0])
                rews.append(reward)
                #print(reward)

                # update
                obs = newobs

            # compute returns from instant rewards for one whole trajectory
            returns = discounted_rewards(rews, gamma)

            # record for batch update
            VAL += returns # NOTE that the list of returns just gets extended. 
                           # There is no separate entry created for each trajectory
            OBS += obss
            ACTS += acts

        # update baseline
        VAL = np.array(VAL)# represents an array where the discounted reward lists are concatenated to each other.
        # the size of VAL should be [numtrajs * len_of_traj_i, 1] where len_of_traj_i is variable depending on the length 
        # of each trajectory.
        OBS = np.array(OBS)# represents an array where the list of states for all trajectories are concatenated.
        # the size of OBS should be [numtrajs * len_of_traj_i, obssize]
        ACTS = np.array(ACTS)

        baseline.train(OBS, VAL)  # update only one step

        # update policy
        BAS = baseline.compute_values(OBS)  # compute baseline for variance reduction
        ADS = VAL - np.squeeze(BAS,1) # computes advantages. An array of (targets)-(estimated from our network) for each
                                      # state

        actor.train(OBS, ACTS, ADS)  # update only one step
        
        
    # Code Evaluation: DO NOT CHANGE CODE HERE
    # after training, we will evaluate the performance of the agent
    # on a target environment
    eval_episodes = 100
    record = []
    env = gym.make('CartPole-v1')
    eval_mode = True
    for ite in range(eval_episodes):

        obs = env.reset()
        done = False
        rsum = 0

        while not done:

            # epsilon greedy for exploration
            if eval_mode:
                p = actor.compute_prob(np.expand_dims(obs,0)).ravel()
                action = np.random.choice(np.arange(2), size=1, p=p)[0]
            else:
                raise NotImplementedError

            newobs, r, done, _ = env.step(action)
            rsum += r
            obs = newobs

        record.append(rsum)

    print("eval performance of PG agent: {}".format(np.mean(record)), '\n')
    
    return np.mean(record)