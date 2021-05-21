from viper.gridworld.environment import ACTION_RIGHT
from viper.gridworld.environment import ACTION_LEFT
from viper.gridworld.environment import DIAGONAL_SUM
from viper.gridworld.environment import NUM_ACTIONS

import random


class GridworldPolicy(object):
    def predict_single(self, observation):
        if observation[0] + observation[1] > DIAGONAL_SUM:
            return ACTION_RIGHT
        else:
            return ACTION_LEFT

    def predict(self, observations):
        actions = []
        for i in range(len(observations)):
            actions.append(self.predict_single(observations[i]))
        return actions

    def predict_q(self, observations):
        # The correct way would be to calculate correct Q values.
        # Using random as a hack, since algorithm will fail with all
        # same Q values. Note that we still do not use q-value
        # sampling in Dagger, but we need to return them like this
        # because of how code is designed.
        q_values = []
        for i in range(len(observations)):
            q_value = []
            for j in range(NUM_ACTIONS):
                q_value.append(random.random())
            q_values.append(q_value)
        return q_values
