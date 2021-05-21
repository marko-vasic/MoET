import gym
from viper.rl_models.dddqn import *


def create_gym_env():
    return gym.make('Acrobot-v1')


def get_state_transformer():
    return lambda obs: obs


def load_rl_policy():
    model_path = '../data/rl_models/model-acrobot-dddqn'
    env = create_gym_env()

    params = PolicyNNParams(env=env,
                            dirname=model_path,
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
                            animate=False)
    policy = PolicyNN(params)
    policy.load()

    return policy


if __name__ == '__main__':
    params = PolicyNNParams(env_name='Acrobot-v1')
    nn = PolicyNN(params)
    nn.train()
