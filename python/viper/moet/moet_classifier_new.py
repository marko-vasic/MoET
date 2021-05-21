from viper.moet.moet_classifier import MOETClassifier
from viper.moet.moet_classifier import test_accuracy
from viper.moet.moet_classifier import test_mountaincar
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score

ADAM_EPSILON = 1e-7
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999


class MOETClassifierNew(MOETClassifier):
    weights = None

    def initialize_weights(self, regularization_mode, gating):
        # TODO: Think about other initialization options.
        self.weights = np.random.rand(gating.shape[0], gating.shape[1])

        if regularization_mode == 0:
            # No regularization.
            pass
        elif regularization_mode == 1:
            # Indexes of elements that are not weighted.
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.2 * x_normalized.shape[0]))
            self.weights[indexes, :] = np.ones(self.experts_no)
        elif regularization_mode == 2:
            # Indexes of selected elements.
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.8 * x_normalized.shape[0]))
            x_normalized = x_normalized[indexes, :]
            y = y[indexes]
            self.weights = self.weights[indexes, :]
        else:
            raise Exception('Unrecognized regularization mode: {}'
                            .format(regularization_mode))

    def train_experts(self, x_normalized, y, pdf, max_depth):
        for j in range(self.experts_no):
            if max_depth == 0:
                # DT does not support depth 0, thus make it happen! :)
                self.dtc_list[j] = DTC(max_depth=1,
                                       min_samples_split=len(x_normalized) + 1)
            else:
                self.dtc_list[j] = DTC(max_depth=max_depth)
            self.dtc_list[j].fit(x_normalized,
                                 y,
                                 sample_weight=self.weights[:, j].T)
            dt_probs = self._dt_proba(j, x_normalized)
            pdf[:, j] = dt_probs[np.arange(len(y)), y].astype(self.default_type)

    def train_gating(self, x_normalized, gating, pdf, learn_rate):
        h = self.h_fun(gating, pdf)
        self.weights = h
        dsdtetag = self.ds_dtetag(x_normalized, self.tetag, self.experts_no)
        e = self.e_fun(h, gating, dsdtetag)

        if not self.use_adam:
            R = self.R_fun(gating, dsdtetag, self.experts_no)
            if np.linalg.cond(R) < 1e7:
                self.tetag += learn_rate * np.linalg.inv(R).dot(e)
            else:
                self.tetag += learn_rate * e
            return R
        else:
            t = self.iter_no
            self.adam_m = ADAM_BETA_1 * self.adam_m + (1 - ADAM_BETA_1) * e
            self.adam_v = ADAM_BETA_2 * self.adam_v + (1 - ADAM_BETA_2) * np.power(e, 2)
            m_hat = self.adam_m / (1 - np.power(ADAM_BETA_1, t))
            v_hat = self.adam_v / (1 - np.power(ADAM_BETA_2, t))
            self.tetag += learn_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            self.iter_no += 1
            
    def _fit_epoch(self,
                   x_normalized,
                   y,
                   max_depth,
                   learn_rate,
                   regularization_mode,
                   is_first_epoch,
                   train_gating_multiple_times):
        """Fit MOE for one epoch.
        Args:
          x_normalized: Array of normalized feature vectors.
          y: Class labels associated with the feature vectors.
          max_depth: Maximum depths of expert DTs.
          learn_rate: Learning rate.
          mode: Regularization mode.
        """
        self.dtc_list = [None for i in range(self.experts_no)]
        gating = self.softmax(x_normalized, self.tetag, self.experts_no)

        if is_first_epoch:
            # Move this elsewhere
            self.iter_no = 1
            self.adam_m = 0
            self.adam_v = 0
            self.initialize_weights(regularization_mode, gating)

        pdf = np.zeros([x_normalized.shape[0], self.experts_no],
                       dtype=self.default_type)
        self.train_experts(x_normalized, y, pdf, max_depth)
        if train_gating_multiple_times:
            for i in range(5):
                self.train_gating(x_normalized, gating, pdf, learn_rate)
        else:
            self.train_gating(x_normalized, gating, pdf, learn_rate)


if __name__ == '__main__':
    test_mountaincar(lambda num_experts, num_classes:
                     MOETClassifierNew(experts_no=num_experts,
                                       no_class=num_classes,
                                       use_adam=True))
    # test_accuracy(lambda num_experts, num_classes:
    #               MOETClassifierNew(experts_no=num_experts,
    #                                 no_class=num_classes,
    #                                 use_adam=True))
