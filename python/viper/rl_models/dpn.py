# This model is currently used for following games:
# - Cartpole

import os
import shutil
import tensorflow as tf
import numpy as np

from ..util.log import *


# Parameters for training a policy neural net.
#
# state_dim: int (the number of mdp states)
# n_actions: int (the number of mdp actions)
# hidden_size: int (the number of hidden units in the neural net)
# discount: float (this mdp discount factor)
# step_size: float (the training step size)
# n_epochs: int (the number of batches)
# batch_size: int (the number of episodes in a batch)
# animate: bool (whether to animate the environment)
# dirname: str (directory in which to save the polcy neural net)
class PolicyNNParams:
    def __init__(self, state_dim, n_actions, n_hidden, hidden_size, discount,
                 step_size, n_epochs, batch_size, save_iters, animate, dirname):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.discount = discount
        self.step_size = step_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_iters = save_iters
        self.animate = animate
        self.dirname = dirname


# A policy neural net with fully connected hidden layers
#
# params: PolicyNNParams (the number of mdp actions)
# sess: tensorflow session
# x: tensorflow placeholder (placeholder for state)
# probs: tensorflow graph (computes the action probabilities)
class PolicyNN:
    def __init__(self, params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Step 0: Basic initialization
            self.sess = tf.Session()
            self.params = params

            # Step 1: Define model

            # Step 1a: Input
            self.x = tf.placeholder(tf.float32, [None, self.params.state_dim])

            # Step 1b: Hidden layers
            hidden = tf.contrib.slim.fully_connected(self.x,
                                                     self.params.hidden_size,
                                                     activation_fn=tf.nn.relu)
            for i in range(self.params.n_hidden - 1):
                hidden = tf.contrib.slim.fully_connected(hidden,
                                                         self.params.hidden_size,
                                                         activation_fn=tf.nn.relu)

            # Step 1c: Output layer
            self.probs = tf.contrib.slim.fully_connected(hidden,
                                                         self.params.n_actions,
                                                         activation_fn=tf.nn.softmax)

            # Step 2: Training steps

            # Step 2a: Training inputs
            self.actions_taken = tf.placeholder(tf.int32, [None])
            self.advantages = tf.placeholder(tf.float32, [None])

            # Step 2b: Indices of actions taken
            indices = tf.range(0, tf.shape(self.probs)[0]) * \
                      tf.shape(self.probs)[1] + self.actions_taken
            actions_taken_probs = tf.gather(tf.reshape(self.probs, [-1]),
                                            indices)

            # Step 2c: Loss function
            loss = -tf.reduce_mean(
                tf.multiply(self.advantages, tf.log(actions_taken_probs)))

            # Step 2d: Gradient step
            self.train_step = tf.train.RMSPropOptimizer(
                self.params.step_size).minimize(loss)

    # Loads the policy neural net.
    def load(self):
        with self.graph.as_default():
            tf.train.Saver().restore(self.sess, os.path.abspath(
                self.params.dirname + '/policy_nn.ckpt'))

    # Trains a policy neural net for the given environment.
    #
    # env: {reset: () -> O, render: () -> (), step: A -> (O, float, bool, M)}
    # ident: int (a unique identifier)
    # O = environment observation
    # A = environment action
    # M = metadata (discarded)
    def train(self, env):
        with self.graph.as_default():
            # Step 1: Initialization
            self.sess.run(tf.global_variables_initializer())
            if os.path.isdir(self.params.dirname):
                tf.train.Saver().restore(self.sess, os.path.abspath(
                    self.params.dirname + '/policy_nn.ckpt'))

            # Step 2: Iterate through epochs
            for i in range(self.params.n_epochs):

                # Step 3: Build batch
                batched_observations, batched_actions, batched_rewards, batched_qs = get_rollouts(
                    env, self, self.params.discount, self.params.animate,
                    self.params.batch_size)

                # Step 4: Normalize rewards
                batched_qs = (batched_qs - np.mean(batched_qs)) / (
                    np.std(batched_qs) + 1e-10)

                # Step 5: Training step
                self.sess.run(self.train_step,
                              feed_dict={self.x: batched_observations,
                                         self.actions_taken: batched_actions,
                                         self.advantages: batched_qs})

                # Step 6: Save neural net and log reward
                if i % self.params.save_iters == 0:
                    if os.path.isdir(self.params.dirname):
                        shutil.rmtree(self.params.dirname, ignore_errors=True)
                    os.makedirs(self.params.dirname)
                    save_path = tf.train.Saver().save(self.sess,
                                                      os.path.abspath(
                                                          self.params.dirname + '/policy_nn.ckpt'))
                    log('Saved policy NN in: %s' % save_path, INFO)
                    log('Reward: ' + str(np.average(batched_rewards)), INFO)

    # Return optimal action according to the policy probabilities.
    #
    # obs: np.array([state_dim])
    # return: int (an mdp action)
    def predict(self, obss):
        with self.graph.as_default():
            probs = self.sess.run(self.probs, feed_dict={self.x: obss})
            return np.argmax(probs, axis=1)

    # Return Q values according to the policy probabilities.
    def predict_q(self, obss):
        with self.graph.as_default():
            probs = self.sess.run(self.probs, feed_dict={self.x: obss})
            qs = np.log(probs + 1e-10)
            return qs

    # Return optimal action according to the policy probabilities for a single observation.
    #
    # obs: np.array([state_dim])
    # return: int (an mdp action)
    def get_action(self, obs):
        with self.graph.as_default():
            probs = \
                self.sess.run(self.probs, feed_dict={self.x: np.array([obs])})[
                    0]
            return np.argmax(probs)


# Represents a uniformly random policy.
#
# n_actions: int (the number of actions)
class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_action(self, obs):
        return np.random.choice(self.n_actions)


# Computes the mean and standard deviation of the state vector
# under the uniform random policy.
#
# env: E
# params: PolicyNNParams
# n_rollouts: int (number of rollouts to use in average)
# E = envirnoment (see get_rollout in policy_gradient.py)
def get_normalization(env, params, n_rollouts):
    # Step 1: Use a random policy
    policy = RandomPolicy(params.n_actions)

    # Step 2: Perform rollouts
    all_observations, _, _, _ = get_rollouts(env, policy, params.discount,
                                             params.animate, n_rollouts)

    # Step 3: Construct mean and standard deviation of state vector
    return (np.mean(all_observations, axis=0), np.std(all_observations, axis=0))


# Returns the default normalization.
#
# state_dim: int (the state dimension)
# return: (np.array([state_dim]), np.array([state_dim]))
def get_default_normalization(state_dim):
    return (np.zeros([state_dim]), np.ones([state_dim]))


# Saves normalization values to file.
#
# normalization: (np.array([state_im]), np.array([state_dim]))
# dirname: str
def save_normalization(normalization, dirname):
    os.makedirs(dirname)
    f = open(dirname + 'norm.txt', 'w')
    for xs in normalization:
        for x in xs:
            f.write(str(x) + '\n')
    f.close()


# Loads normalization values from file.
#
# dirname: str
# return: (np.array([state_im]), np.array([state_dim]))
def load_normalization(dirname):
    f = open(dirname + 'norm.txt')
    vals = []
    for line in f:
        vals.append(float(line[:-1]))
    f.close()
    if len(vals) % 2 != 0:
        raise Exception('Invalid normalization!')
    n = int(len(vals) / 2)
    return (np.array(vals[:n]), np.array(vals[n:]))


# Returns whether the normalization values file exists.
#
# dirname: str
# return: bool
def exists_normalization(dirname):
    return os.path.isfile(dirname + 'norm.txt')
