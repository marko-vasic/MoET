import gym
from viper.lunarlander.lunarlander import load_rl_policy
import numpy as np

if __name__ == '__main__':
    policy = load_rl_policy()
    env = gym.make('LunarLander-v2')
    print('action_space: ' + str(env.action_space))
    print('observation space: ' + str(env.observation_space))

    observation, done = env.reset(), False
    G = 0
    while not done:
        # Rendering works with newer version of tensorflow (2.4) but not older
        # env.render()
        action = policy.predict(np.array([observation]))[0]
        observation, reward, done, info = env.step(action)
        G += reward
    print("G = {}".format(G))
