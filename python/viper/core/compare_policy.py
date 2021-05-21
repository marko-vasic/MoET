class ComparePolicy:
    """Class for comparing two policies."""

    def __init__(self, policy1, policy2):
        """policy1 is a default policy."""
        self.policy1 = policy1
        self.policy2 = policy2
        self.num_predictions = 0
        self.num_mispredictions = 0

    def predict(self, obss):
        predictions1 = self.policy1.predict(obss)
        predictions2 = self.policy2.predict(obss)
        for i in range(len(obss)):
            prediction1 = predictions1[i]
            prediction2 = predictions2[i]
            self.num_predictions += 1
            if prediction1 != prediction2:
                self.num_mispredictions += 1
        return predictions1

    def mispredictions_ratio(self):
        """Ratio of predictions made that are different from predictions that
        RL agent would make."""
        return float(self.num_mispredictions) / self.num_predictions
