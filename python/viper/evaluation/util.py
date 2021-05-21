import importlib
import re
import math

import numpy as np

from viper.evaluation.constants import EXPERTS
from viper.evaluation.constants import EXPERT_DEPTHS
from viper.evaluation.constants import DEPTHS


def call_function(subject_name, function_name):
    module = importlib.import_module('viper.{0}.{1}'.
                                     format(subject_name, subject_name))
    func = getattr(module, function_name)
    return func()


def create_gym_env(subject_name):
    return call_function(subject_name, 'create_gym_env')


def load_rl_policy(subject_name):
    return call_function(subject_name, 'load_rl_policy')


def get_state_transformer(subject_name):
    return call_function(subject_name, 'get_state_transformer')


def get_config_names(subject, model):
    configs = list()
    if model == 'ViperPlus':
        for depth in DEPTHS[subject]:
            configs.append('d{}'.format(depth))
    else:
        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                configs.append('e{}_d{}'.format(experts, depth))
    return configs


def get_config_names_for_depth(subject, model, depth):
    """
    Returns names of all configurations for a given subject, model and depth.
    """
    configs = []
    if model == 'ViperPlus':
        configs.append('d{}'.format(depth))
    elif "MOE" in model:
        for experts in EXPERTS[subject]:
            configs.append('e{}_d{}'.format(experts, depth))
    return configs


def get_experts_from_config_name(config_name):
    """
    Extracts number of experts used from configuration name.
    If configuration name does not contain information about experts
    (like in Viper) than None is returned.
    Returns:
        int: number of experts, or -1 if experts are not used in configuration.
    """
    m = re.match('e(?P<experts>[0-9]+)_d(?P<depth>[0-9]+)',
                 config_name)
    if m:
        return int(m.group('experts'))
    else:
        return -1

def get_depth_from_config_name(config_name):
    """
    Extracts DT depth used from configuration name.
    Returns:
        int: DT depth.
    """
    m = re.match('e(?P<experts>[0-9]+)_d(?P<depth>[0-9]+)',
                 config_name)
    return int(m.group('depth'))


def identify_pareto(scores):
    """
    reference code: https://pythonhealthcare.org/tag/pareto-front/
    :param scores: 2D numpy array of shape (N, 2) -- N is number of points.
      Each row in the array contains two scores (criteria)
        based on which pareto fronts are constructed.
      Note that in both criteria higher value is considered better.
    :return: 1D numpy array containing IDs of points on the pareto front.
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
