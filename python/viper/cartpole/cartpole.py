import gym
from viper.rl_models.dpn import PolicyNNParams
from viper.rl_models.dpn import PolicyNN


def create_gym_env():
    return gym.make('CartPole-v0')


def get_state_transformer():
    return lambda obs: obs


def load_rl_policy():
    # Step 1: Policy NN parameters
    n_hidden = 1
    hidden_size = 8
    discount = 0.99
    step_size = 1e-2
    n_epochs = 1000
    batch_size = 50
    save_iters = 20
    animate = False
    model_path = '../data/rl_models/model-cartpole'

    # Step 2: Environment parameters
    env = create_gym_env()
    state_dim = env.env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Step 3: Build parameters
    params = PolicyNNParams(state_dim, n_actions, n_hidden, hidden_size,
                            discount, step_size, n_epochs, batch_size,
                            save_iters, animate, model_path)

    # Step 4: Load policy
    policy = PolicyNN(params)
    policy.load()

    return policy
