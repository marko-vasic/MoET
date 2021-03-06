# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from viper.rl_models.dqn import DQNPolicy

import numpy as np
import gym
from .atari_wrappers_deprecated import wrap_dqn


def create_gym_env():
    return wrap_dqn(gym.make('PongNoFrameskip-v4'))


def load_rl_policy():
    model_path = '../data/rl_models/model-atari-pong-1/saved'
    # TODO: Do not provide env to DQNPolicy, instead provide
    # num_actions and input_shape
    env = create_gym_env()
    return DQNPolicy(env, model_path)


def _get_our_paddle(obs_t):
    the_slice = obs_t[:, 74]
    pad_color = np.full(the_slice.shape, 147)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3


def _get_enemy_paddle(obs_t):
    the_slice = obs_t[:, 9]
    pad_color = np.full(the_slice.shape, 148)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3


def _get_ball(obs_t):
    """
    Returns position (coordinates) of ball in observation matrix.
    Arguments:
        obs_t: Observation matrix of 83 x 84 dimension.
    """
    if (np.max(obs_t) < 200):
        return np.array([0, 0])
    idx = np.argmax(obs_t)
    return np.unravel_index(idx, obs_t.shape)


def get_state_transformer():
    return get_pong_symbolic


def get_pong_symbolic(obs):
    """Converts observation matrix into symbols.

    Arguments:
        obs: 84 x 84 x 4 matrix, where the last dimension represents number of last frames taken.
    """
    obs = obs[:83, :, :]
    obs = [obs[:, :, 3], obs[:, :, 2], obs[:, :, 1], obs[:, :, 0]]
    # Ball 1st coordinate
    b = [_get_ball(ob)[0] for ob in obs]
    # Ball 2n coordinate
    c = [_get_ball(ob)[1] for ob in obs]
    # Paddle 1st coordinate (other coordinate is fixed).
    p = [_get_our_paddle(ob) for ob in obs]

    # new state
    symb = []

    # position
    symb.append(b[0] - p[0])
    symb.append(c[0])

    # velocity
    symb.append(b[0] - b[1])
    symb.append(c[0] - c[1])
    symb.append(p[0] - p[1])

    # acceleration
    symb.append(p[0] - 2.0 * p[1] + p[2])

    # jerk
    symb.append(p[0] - 3.0 * p[1] + 3.0 * p[2] - p[3])

    return np.array(symb)
