from abc import ABCMeta, abstractmethod


class HybridAgent:
    __metaclass__ = ABCMeta

    def __init__(self, rl_agent):
        self.rl_agent = rl_agent
        # Total number of predictions.
        self.num_predictions = 0
        # Number of predictions using RL agent.
        self.num_predictions_rl_agent = 0

    def predictions_by_inferred_policy(self):
        """Returns ration of predictions that is made by inferred policy (not
        by RL agent).

        """
        return 1 - (float(self.num_predictions_rl_agent) / self.num_predictions)

    @abstractmethod
    def modify_action(self, obs, action):
        pass

    def predict(self, obss):
        acts = self.rl_agent.predict(obss)
        for i in range(len(obss)):
            new_action = self.modify_action(obss[i], acts[i])
            acts[i] = new_action
        return acts

    def get_action(self, obs):
        return self.modify_action(obs, self.rl_agent.get_action(obs))
