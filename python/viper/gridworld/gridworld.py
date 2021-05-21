from viper.gridworld.policy import GridworldPolicy
from viper.gridworld.environment import GridworldEnvironment


def create_gym_env():
    return GridworldEnvironment()


def get_state_transformer():
    return lambda obs: obs


def load_rl_policy():
    return GridworldPolicy()
