# This model is currently used for following games:
# - Acrobot
# - Mountaincar

from collections import deque
from ..util.log import *
from gym import wrappers, spaces
from .memory import Memory

import gym
import math
import numpy as np
import os
import random
import tensorflow as tf

TENSORFLOW_VERSION = int(tf.__version__[0])

# Also, see this guide for migration between TF versions:
# https://www.tensorflow.org/guide/migrate
if TENSORFLOW_VERSION == 1:
    from tensorflow import Session
    from tensorflow import variable_scope
    from tensorflow import placeholder
    from tensorflow import layers
    from tensorflow import train
    from tensorflow import get_collection
    from tensorflow import GraphKeys
    from tensorflow import global_variables_initializer
    from tensorflow import train
    from tensorflow.contrib.layers import xavier_initializer as KernelInitializer
    from tensorflow import squared_difference
elif TENSORFLOW_VERSION == 2:
    from tensorflow.compat.v1 import Session
    from tensorflow.compat.v1 import variable_scope
    from tensorflow.compat.v1 import placeholder
    from tensorflow.compat.v1 import layers
    from tensorflow.compat.v1 import train
    from tensorflow.compat.v1 import get_collection
    from tensorflow.compat.v1 import GraphKeys
    from tensorflow.compat.v1 import global_variables_initializer
    from tensorflow.compat.v1 import train
    from tensorflow.keras.initializers import GlorotUniform as KernelInitializer
    from tensorflow.math import squared_difference
else:
    raise Exception("Tensorflow v{} is not supported".format(
        TENSORFLOW_VERSION))


class PolicyNNParams:
    def __init__(
        self,
        # Default environment
        env=None,
        env_name="Acrobot-v1",
        hidden_sizes=(50, 50, 50),
        gamma=0.99,
        explore_start=1.0,
        explore_stop=0.001,
        decay_rate=0.000001,
        step_size=10000,
        n_epochs=50000,
        batch_size=50,
        learning_rate=1e-3,
        memory_size=10000,
        max_tau=5000,
        animate=False,
        dirname=os.getcwd(),
    ):
        if not env:
            self.env = wrappers.Monitor(
                gym.make(env_name), dirname, video_callable=animate
            )
        else:
            self.env = env
        self.state_dim = self._get_state_dim(self.env.observation_space)
        self.n_actions = self.env.action_space.n
        # log("### state_dim = {}".format(self.state_dim), INFO)
        # log("### n_actions = {}".format(self.n_actions), INFO)
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.max_tau = max_tau
        self.animate = animate
        self.dirname = dirname

    def _get_state_dim(self, observation_space):
        if isinstance(observation_space, spaces.Discrete):
            return 1
        elif isinstance(observation_space, spaces.Tuple):
            return len(observation_space)
        else:
            return observation_space.shape[0]


class DDDQNNet:
    def __init__(self, params, name):
        self.params = params
        self.name = name

        with variable_scope(self.name):
            self.states = placeholder(
                tf.float32, [None, self.params.state_dim], name="states"
            )
            self.importance_sampling_weights = placeholder(
                tf.float32, [None, 1], name="importance_sampling_weights"
            )
            self.actions = placeholder(tf.int32, [None], name="actions")
            self.one_hot_actions = tf.one_hot(
                self.actions, self.params.n_actions, name="one_hot_actions"
            )
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = placeholder(
                tf.float32, [None], name="target_Q"
            )

            layer = self.states
            for i, hidden_size in enumerate(self.params.hidden_sizes[:-1]):
                layer = layers.dense(
                    inputs=layer,
                    units=hidden_size,
                    activation=tf.nn.elu,
                    kernel_initializer=KernelInitializer(),
                    name="hidden_layer_{}".format(i + 1),
                )

            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = layers.dense(
                inputs=layer,
                units=self.params.hidden_sizes[-1],
                activation=tf.nn.elu,
                kernel_initializer=KernelInitializer(),
                name="value_fc",
            )
            self.value = layers.dense(
                inputs=self.value_fc,
                units=1,
                activation=None,
                kernel_initializer=KernelInitializer(),
                name="value",
            )
            # The one that calculate A(s,a)
            self.advantage_fc = layers.dense(
                inputs=layer,
                units=self.params.hidden_sizes[-1],
                activation=tf.nn.elu,
                kernel_initializer=KernelInitializer(),
                name="advantage_fc",
            )
            self.advantage = layers.dense(
                inputs=self.advantage_fc,
                units=self.params.n_actions,
                activation=None,
                kernel_initializer=KernelInitializer(),
                name="advantages",
            )

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            if TENSORFLOW_VERSION == 2:
                self.output = self.value + tf.subtract(
                    self.advantage, tf.reduce_mean(self.advantage, axis=1,
                                                   keepdims=True)
                )
            elif TENSORFLOW_VERSION == 1:
                self.output = self.value + tf.subtract(
                    self.advantage, tf.reduce_mean(self.advantage, axis=1,
                                                   keep_dims=True)
                )

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(
                tf.multiply(self.output, self.one_hot_actions), axis=1
            )

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(
                self.target_Q - self.Q
            )  # for updating Sumtree

            self.loss = tf.reduce_mean(
                self.importance_sampling_weights
                * squared_difference(self.target_Q, self.Q)
            )

            self.optimizer = train.RMSPropOptimizer(
                self.params.learning_rate
            ).minimize(self.loss)


class PolicyNN:
    def __init__(self, params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = Session()

            self.params = params

            # Instantiate the DQNetwork
            self.deep_q_network = DDDQNNet(self.params, name="DQNetwork")

            # Instantiate the target network
            self.target_network = DDDQNNet(self.params, name="TargetNetwork")

            # Instantiate memory
            self.memory = Memory(self.params.memory_size)

            for i in range(self.params.memory_size):
                # If it's the first step
                if i == 0:
                    # First we need a state
                    state = np.atleast_1d(self.params.env.reset())

                # Random action
                action = self.params.env.action_space.sample()

                # Get the rewards
                next_state, reward, done, _ = self.params.env.step(action)
                next_state = np.atleast_1d(next_state)

                experience = state, action, reward, next_state, done
                self.memory.store(experience)

                if done:
                    state = np.atleast_1d(self.params.env.reset())
                else:
                    state = next_state

            # Wait for env to be done
            while True:
                # Random action
                action = self.params.env.action_space.sample()

                # Get the rewards
                next_state, reward, done, _ = self.params.env.step(action)

                if done:
                    break

    def predict_action(self, decay_step, input_states):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = self.params.explore_stop + (
            self.params.explore_start - self.params.explore_stop
        ) * np.exp(-self.params.decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = self.params.env.action_space.sample()
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = self.sess.run(
                self.deep_q_network.output,
                feed_dict={
                    self.deep_q_network.states: input_states.reshape(
                        (1, self.params.state_dim)
                    )
                },
            )

            # Take the biggest Q value (= the best action)
            action = np.argmax(Qs)

        return action, explore_probability

    def update_target_graph(self):
        # Get the parameters of our DQNNetwork
        from_vars = get_collection(
            GraphKeys.TRAINABLE_VARIABLES, "DQNetwork"
        )

        # Get the parameters of our Target_network
        to_vars = get_collection(
            GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork"
        )

        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def train(self):
        with self.graph.as_default():
            # Initialize the variables
            self.sess.run(global_variables_initializer())

            # Initialize the decay rate (that will use to reduce epsilon)
            decay_step = 0

            # Set tau = 0
            tau = 0

            # Update the parameters of our TargetNetwork with DQN_weights
            update_target = self.update_target_graph()
            self.sess.run(update_target)

            # Max total reward seen so far
            max_seen_total_reward = -np.inf

            loss = np.inf

            for episode in range(self.params.n_epochs):
                # Set step to 0
                step = 0

                # Initialize the rewards of the episode
                episode_rewards = []

                # Make a new episode and observe the first state
                state = np.atleast_1d(self.params.env.reset())

                while step < self.params.step_size:
                    step += 1

                    # Increase the C step
                    tau += 1

                    # Increase decay_step
                    decay_step += 1

                    # With Epsilon select a random action atat, otherwise select a = argmaxQ(st,a)
                    action, explore_probability = self.predict_action(decay_step, state)

                    # Do the action
                    next_state, reward, done, _ = self.params.env.step(action)
                    next_state = np.atleast_1d(next_state)

                    # Add the reward to total reward
                    episode_rewards.append(reward)

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    self.memory.store(experience)

                    # If the game is finished
                    if done:
                        # Set step = max_steps to end the episode
                        step = self.params.step_size

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        log("Episode: {}".format(episode), INFO)
                        log("  Total reward: {}".format(total_reward), INFO)
                        log("  Training loss: {:.4f}".format(loss), INFO)
                        log("  Explore P: {:.4f}".format(explore_probability), INFO)
                    else:
                        # st+1 is now our current state
                        state = next_state

                    ### LEARNING PART
                    # Obtain random mini-batch from memory
                    (
                        tree_idx,
                        batch,
                        importance_sampling_weights_mb,
                    ) = self.memory.sample(self.params.batch_size)

                    states_mb = np.array([each[0][0] for each in batch])
                    actions_mb = np.array([each[0][1] for each in batch])
                    rewards_mb = np.array([each[0][2] for each in batch])
                    next_states_mb = np.array([each[0][3] for each in batch])
                    dones_mb = np.array([each[0][4] for each in batch])

                    target_Qs_batch = []

                    ### DOUBLE DQN Logic
                    # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                    # Use TargetNetwork to calculate the Q_val of Q(s',a')

                    # Get Q values for next_state
                    q_next_state = self.sess.run(
                        self.deep_q_network.output,
                        feed_dict={self.deep_q_network.states: next_states_mb},
                    )

                    # Calculate Qtarget for all actions that state
                    q_target_next_state = self.sess.run(
                        self.target_network.output,
                        feed_dict={self.target_network.states: next_states_mb},
                    )

                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                    for i in range(0, len(batch)):
                        # We got a'
                        action = np.argmax(q_next_state[i])

                        # If we are in a terminal state, only equals reward
                        if dones_mb[i]:
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            # Take the Qtarget for action a'
                            target = (
                                rewards_mb[i]
                                + self.params.gamma * q_target_next_state[i][action]
                            )
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    _, loss, absolute_errors = self.sess.run(
                        [
                            self.deep_q_network.optimizer,
                            self.deep_q_network.loss,
                            self.deep_q_network.absolute_errors,
                        ],
                        feed_dict={
                            self.deep_q_network.states: states_mb,
                            self.deep_q_network.target_Q: targets_mb,
                            self.deep_q_network.actions: actions_mb,
                            self.deep_q_network.importance_sampling_weights: importance_sampling_weights_mb,
                        },
                    )

                    # Update priority
                    self.memory.batch_update(tree_idx, absolute_errors)

                    if tau > self.params.max_tau:
                        # Update the parameters of our TargetNetwork with DQN_weights
                        update_target = self.update_target_graph()
                        self.sess.run(update_target)
                        tau = 0

                # Save model every 5 episodes
                total_reward = np.sum(episode_rewards)
                if total_reward > max_seen_total_reward:
                    max_seen_total_reward = total_reward
                    model_path = os.path.abspath(
                        self.params.dirname + "/policy_nn_{}.ckpt".format(total_reward)
                    )
                    save_path = train.Saver().save(
                        self.sess, model_path, global_step=decay_step
                    )
                    log("Saved policy NN in: %s" % save_path, INFO)
                    log("  Max rewards: {}".format(total_reward), INFO)

        self.params.env.close()

    def load(self):
        with self.graph.as_default():
            train.Saver().restore(
                self.sess, os.path.abspath(self.params.dirname + "/policy_nn.ckpt")
            )

    def predict(self, observations):
        with self.graph.as_default():
            Qs = self.sess.run(
                self.deep_q_network.output,
                feed_dict={self.deep_q_network.states: observations},
            )

            # Take the biggest Q value (= the best action)
            return np.argmax(Qs, axis=1)

    def predict_q(self, observations):
        with self.graph.as_default():
            return self.sess.run(
                self.deep_q_network.output,
                feed_dict={self.deep_q_network.states: observations},
            )

    def get_action(self, observation):
        return self.predict(np.array(observation))[0]


if __name__ == "__main__":
    params = PolicyNNParams()
    nn = PolicyNN(params)
    nn.train()
