"""MOE for multiclass classification."""

import copy
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from viper.moet.moet_base import MOETBase
from viper.util.log import *


class MOETClassifier(MOETBase):
    # This is added for backward compatibility with old pickle files
    # (to be removed later).
    default_type = np.float32
    use_adam = False

    def __init__(self,
                 experts_no,
                 no_class,
                 default_type=np.float32,
                 use_adam=False):
        """
        Args:
          experts_no: Number of experts.
          no_class: Number of classes in output space.
          default_type: Default type used for operations (choose
            np.float64, np.float32 or np.float16). There is a
            trade-off between speed and precision.
        """

        super(MOETClassifier, self).__init__(
            experts_no=experts_no,
            default_type=default_type,
        )
        self.no_class = no_class
        # Initialized during training.
        self.tetag = None
        # Initialized during training.
        self.scaler = None
        # Collection of decision tree experts.
        # Initialized during training.
        self.dtc_list = None
        # Whether to use adam optimizer.
        self.use_adam = use_adam

    def _preprocess_train_data(self, x):
        """
        Preprocess training data.
        NOTE: This method has side effects, as it constructs self.scaler
        object
        """
        x = x.astype(self.default_type)
        self.scaler = StandardScaler()
        self.scaler.fit(x)

    def _normalize_x(self, x):
        """
        This function normalizes features.
        Args:
            x: Array of feature vectors.
        Returns:
            Array of normalized feature vectors.
        """
        x = self.scaler.transform(x)
        # Add a bias term to every input.
        # TODO: Provide x without bias when training DT.
        return np.append(x,
                         np.ones([x.shape[0], 1], dtype=self.default_type),
                         axis=1)

    def _dt_proba(self, expert_id, x_normalized):
        """
        Calculates the class probabilities using the given expert
          for given feature vectors.
        Args:
          expert_id: Index of expert to be used.
          x_normalized: Array of normalized feature vectors.
        """
        dt = self.dtc_list[expert_id]
        # Returns array of labels not present in DT.
        # This happens when DT is trained on examples missing some label.
        check = np.where(np.isin(np.arange(self.no_class),
                                 dt.classes_) == False)[0]
        dt_probs = dt.predict_proba(x_normalized)
        if check.size != 0:
            for i in check:
                dt_probs = np.insert(dt_probs, [i], 0, axis=1)
        return dt_probs

    def _expert_proba(self, x_normalized):
        """
        Args:
            x_normalized: Array of normalized feature vectors.
        Returns:
            Array of dimensions (N, C, E) where element at index (i, c, e) is a
            probability of input i being classified with class c by expert e.
        """
        N = x_normalized.shape[0]
        E = self.experts_no
        C = self.no_class

        probs = np.zeros([N, C, E], dtype=self.default_type)
        for expert_id in range(self.experts_no):
            # element at index (i, c) is a probability of input i being
            # classified with class c by the current expert.
            # dims: (N, C)
            expert_probs = self._dt_proba(expert_id, x_normalized)
            probs[:, :, expert_id] = expert_probs
        return probs

    def predict_expert_proba(self, x):
        """
        For each input point computes probabilities of choosing
          different experts.
        Args:
          x: Array of feature vectors.
        Returns:
          Array containing a probability distribution across experts for
            each data point in the input array.
        """
        x_normalized = self._normalize_x(x)
        return self.softmax(x_normalized, self.tetag, self.experts_no)

    def predict_expert(self, x):
        """
        For each input computes the most probable expert.
        Args:
          x: Array of feature vectors.
        Returns:
          Array containing the most probable expert for each point in the input
          array.
        """
        gating = self.predict_expert_proba(x)
        return np.argmax(gating, axis=1)

    def predict_with_expert(self, x, experts):
        """
        Make a prediction using predefined expert.
        Args:
            x: Array of feature vectors.
            experts: Array of expert ids determining which expert should be
                used for prediction. This array should be of same length as x.
        Returns:
            Predictions.
        """
        x_normalized = self._normalize_x(x)

        N = x.shape[0]
        E = self.experts_no
        C = self.no_class

        # Indicator array containing 1 for expert that is chosen for input, 0
        # for other experts.
        # dims: (N, E).
        gating = np.zeros([N, E])
        gating[np.arange(N), experts] = 1

        # dims: (N, C, E).
        probs = self._expert_proba(x_normalized)
        probs *= np.repeat(gating[:, np.newaxis, :], C, axis=1)

        result = np.argmax(np.sum(probs, axis=2), axis=1)
        # TODO: Why round? It should be int already.
        return np.round(result).astype(int)

    def predict(self, x):
        """
        Predicts the output for given inputs using softening (outputs of
          experts are weighted based on the gating value).
        Args:
          x: Array of feature vectors.
        """
        x_normalized = self._normalize_x(x)

        N = x.shape[0]
        E = self.experts_no
        C = self.no_class

        # Contains a probability distribution of experts for each input.
        # dims: (N, E).
        gating = self.predict_expert_proba(x)

        # dims: (N, C, E).
        probs = self._expert_proba(x_normalized)
        probs *= np.repeat(gating[:, np.newaxis, :], C, axis=1)

        result = np.argmax(np.sum(probs, axis=2), axis=1)
        # TODO: Why round? It should be int already.
        return np.round(result).astype(int)

    def predict_hard(self, x):
        """
        Predicts the output for given inputs using hard prediction (output
          of the most probable expert is chosen).
        Args:
          x: Array of feature vectors.
        """
        return self.predict_with_expert(x, self.predict_expert(x))

    def _fit_epoch(self,
                   x_normalized,
                   y,
                   max_depth,
                   learn_rate,
                   regularization_mode,
                   is_first_epoch,
                   train_gating_multiple_times=False):
        """Fit MOE for one epoch.
        Args:
          x_normalized: Array of normalized feature vectors.
          y: Class labels associated with the feature vectors.
          max_depth: Maximum depths of expert DTs.
          learn_rate: Learning rate.
          mode: Regularization mode.
          is_first_epoch (bool): Whether it is the first epoch during
              training.
        """
        self.dtc_list = [None for i in range(self.experts_no)]
        gating = self.softmax(x_normalized, self.tetag, self.experts_no)

        if regularization_mode == 0:
            weights = gating
        elif regularization_mode == 1:
            # Indexes of elements that are not weighted.
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.2 * x_normalized.shape[0]))
            weights = gating
            weights[indexes, :] = np.ones(self.experts_no)
        elif regularization_mode == 2:
            # Indexes of selected elements.
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.8 * x_normalized.shape[0]))
            x_normalized = x_normalized[indexes, :]
            y = y[indexes]
            gating = gating[indexes, :]
            weights = gating
        else:
            raise Exception('Unrecognized regularization mode: {}'
                            .format(regularization_mode))

        pdf = np.zeros([x_normalized.shape[0], self.experts_no],
                       dtype=self.default_type)

        for j in range(self.experts_no):
            if max_depth == 0:
                # DT does not support depth 0, thus make it happen! :)
                self.dtc_list[j] = DTC(max_depth=1,
                                       min_samples_split=len(x_normalized) + 1)
            else:
                self.dtc_list[j] = DTC(max_depth=max_depth)
            self.dtc_list[j].fit(x_normalized, y, sample_weight=weights[:, j].T)
            dt_probs = self._dt_proba(j, x_normalized)
            pdf[:, j] = dt_probs[np.arange(len(y)), y].astype(
                self.default_type)

        h = self.h_fun(gating, pdf)
        dsdtetag = self.ds_dtetag(x_normalized, self.tetag, self.experts_no)
        e = self.e_fun(h, gating, dsdtetag)
        R = self.R_fun(gating, dsdtetag, self.experts_no)
        if np.linalg.cond(R) < 1e7:
            self.tetag += learn_rate * np.linalg.inv(R).dot(e)
        else:
            self.tetag += learn_rate * e
        return R

    def fit(self, x, y,
            max_depth,
            max_epoch=100,
            init_learning_rate=2,
            learning_rate_decay=0.95,
            regularization_mode=0):
        """
        Fit the model.
        Args:
          x: Array of feature vectors.
          y: Labels. Size of the first dimension has to be equal to
            the number of examples. 
          max_depth: Maximum depth of expert DTs.
          max_epoch: Maximum number of epoch for training.
          init_learning_rate: Initial learning rate.
          learning_rate_decay: Decay of the learning rate.
          regularization_mode: 0 - no regularization;
                               1 - batch regularization with 0.8 weighted and 0.2 none;
                               2 - batch regularization with 0.8 used.
        """

        self._preprocess_train_data(x)
        x_normalized = self._normalize_x(x)

        self.tetag = np.random.rand(x_normalized.shape[1]
                                    * self.experts_no).astype(self.default_type)
        learn_rate = [
            init_learning_rate * (learning_rate_decay ** max(float(i), 0.0))
            for i in range(max_epoch)]

        for epoch_id in range(max_epoch):
            self._fit_epoch(x_normalized,
                            y,
                            max_depth,
                            learn_rate[epoch_id],
                            regularization_mode,
                            epoch_id == 0)

    def train(self,
              x, y,
              x_test, y_test,
              max_depth,
              max_epoch=100,
              init_learning_rate=1.,
              learning_rate_decay=0.98,
              log_frequency=None,
              stop_count=None,
              regularization_mode=0,
              return_best_epoch=True,
              gradually_increase_max_depth=True,
              train_gating_multiple_times=False):
        """
        Train the model. Similar to fit, but also makes use of test data.
        Args:
          x: Array of feature vectors.
          y: Array of labels. Size of the first dimension has to be
            equal to the number of examples.
          x_test: Array of feature vectors for testing (i.e.,
          validation).
          y_test: Array of labels for testing (i.e., validation).
          max_depth: Maximum depth of expert DTs.
          max_epoch: Maximum number of epoch for training.
          init_learning_rate: Initial learning rate.
          learning_rate_decay: Decay of the learning rate.
          log_frequency: On how many epochs to log stats. Set to None
            if you do not want to log.  (note that this can be
            expensive operation and thus you might want to consider
            how frequently to log).
          stop_count: If test accuracy does not improve for stop_count
            times, then training is stopped. Early stopping is not
            used if value of this parameter is None.
          regularization_mode:
            0 - no regularization;
            1 - batch regularization with 0.8 weighted and 0.2 none;
            2 - batch regularization with 0.8 used.
          return_best_epoch: Returns value of the parameters from the epoch
            that had the best test score.
          gradually_increase_max_depth: Runs one 'pre-epoch' with
            0-depth experts,
            then uses max_depth for the depth of experts for other epochs.
            This came as an observation that when max_depth is too large,
            decision trees can fit very well in the first epoch without leaving
            much space for the gating function to play in the decision making.
            TODO: Experiment with gradually increasing max_depth, instead of
            jumping from depth 0 to max_depth.
        """

        self._preprocess_train_data(x)
        x_normalized = self._normalize_x(x)

        self.tetag = np.random.rand(x_normalized.shape[1]
                                    * self.experts_no).astype(self.default_type)
        learn_rate = [
            init_learning_rate * (learning_rate_decay ** max(float(i), 0.0))
            for i in range(max_epoch)]

        # Preserves value of parameters for the best performing epoch.
        best_tetag = None
        # Decision trees for the best performing epoch.
        best_dtc_list = None
        # Value of the best performance on the test data.
        best_test_perf = -1.
        # Current test performance of MOET soft.
        test_perf = -1.
        # How many times test accuracy did not improve.
        no_improvement_count = 0

        is_first_epoch = True

        if gradually_increase_max_depth:
            self._fit_epoch(x_normalized,
                            y,
                            max_depth=0,
                            learn_rate=learn_rate[0],
                            regularization_mode=regularization_mode,
                            is_first_epoch=is_first_epoch,
                            train_gating_multiple_times=train_gating_multiple_times)
            is_first_epoch = False

        for epoch_id in range(max_epoch):
            current_max_depth = max_depth
            # if gradually_increase_max_depth and epoch_id < 5:
            #     # current_max_depth = math.floor(max_depth * float(epoch_id) / float(max_epoch)) + 1
            #     # current_max_depth = max(1, min(max_depth, current_max_depth))
            #     current_max_depth = 1
            
            R = self._fit_epoch(x_normalized,
                                y,
                                current_max_depth,
                                learn_rate[epoch_id],
                                regularization_mode,
                                is_first_epoch=is_first_epoch,
                                train_gating_multiple_times=train_gating_multiple_times)
            is_first_epoch = False

            old_test_perf = test_perf
            test_perf = accuracy_score(y_test,
                                       self.predict(x_test))

            if return_best_epoch and test_perf > best_test_perf:
                best_test_perf = test_perf
                best_tetag = self.tetag.copy()
                best_dtc_list = copy.deepcopy(self.dtc_list)
            
            if stop_count is not None:
                # Early stopping.
                if test_perf <= old_test_perf:
                    no_improvement_count += 1
                    if no_improvement_count == stop_count:
                        log('Early stopping training after {} epochs.'
                            .format(epoch_id + 1), INFO)
                        break
                else:
                    no_improvement_count = 0

            if (log_frequency is not None and log_frequency > 0 and
                (epoch_id == 0 or (epoch_id + 1) % log_frequency == 0)):
                log('Epoch {} stats:'.format(epoch_id + 1), INFO)

                if R is not None:
                    try:
                        log('Hessian={:5.6}'.format(
                            np.linalg.det(np.linalg.inv(R))),
                            INFO)
                    except np.linalg.LinAlgError:
                        log('Hessian is a singular matrix.',
                            INFO)
                    log('COND_number={:5.6}'.format(np.linalg.cond(R)), INFO)

                train_score_soft = f1_score(y, self.predict(x),
                                            average='weighted')
                train_score_hard = f1_score(y, self.predict(x),
                                            average='weighted')
                log('Train (f1 score): soft={:1.2f}, hard={:1.2f}'.format(
                    train_score_soft, train_score_hard), INFO)

                test_score_soft = f1_score(y_test, self.predict(x_test),
                                           average='weighted')
                test_score_hard = f1_score(y_test, self.predict_hard(x_test),
                                           average='weighted')
                log('Test (f1 score): soft={:1.2f}, hard={:1.2f}'.format(
                    test_score_soft, test_score_hard), INFO)

        if return_best_epoch:
            self.tetag = best_tetag
            self.dtc_list = best_dtc_list


def test_accuracy(create_model):
    """
    Testing accuracy of MOE model on a synthetic data generated from
    uniform distribution.
    
    Args:
        create_model: Lambda function with signature: MOEBase
            (int:num_experts, int:num_classes) that creates a MOE
            classifier model.
    """
    
    num_examples = 500
    num_features = 4
    max_value = 4
    number_of_classes = 5

    x = max_value * np.random.rand(num_examples, num_features)
    y = np.round(np.random.rand(num_examples) * (number_of_classes - 1)).astype(int)

    num_test_examples = int(num_examples / 5)
    x_test = max_value * np.random.rand(num_test_examples, num_features)
    y_test = (np.round(np.random.rand(num_test_examples) * (number_of_classes - 1)).
              astype(int))

    for max_depth in [4, 5, 6, 7, 8, 9, 10]:
        dtree = DTC(max_depth=max_depth)
        dtree.fit(x, y)
        dt_results = dtree.predict(x)
        prec_dt = accuracy_score(y, dt_results)
        print('Accuracy of DT with depth {} on a train set: {:.2f}'.format(max_depth, prec_dt))

    experts_depth = 4
    for num_experts in range(2, 16):
        model = create_model(num_experts, number_of_classes)
        model.train(x=x, y=y,
                    x_test=x_test, y_test=y_test,
                    max_depth=experts_depth,
                    regularization_mode=0, log_frequency=None,
                    init_learning_rate=1, learning_rate_decay=0.98,
                    max_epoch=100)
        moe_results = model.predict(x)
        moe_results_hard = model.predict_hard(x)
        prec_moe = accuracy_score(y, moe_results)
        prec_moe_hard = accuracy_score(y, moe_results_hard)
        print(
            'Accuracy (train set) of MOE ({} experts, {} depth) soft: {:.2f}, hard {:.2f}'
            .format(num_experts, experts_depth, prec_moe, prec_moe_hard))


def test_memory():
    no_instances = 100
    no_classes = 6
    no_features = 4
    number_range = 4
    x = number_range * np.random.rand(no_instances, no_features)
    y = np.round(np.random.rand(no_instances) * (no_classes - 1)).astype(int)

    model = MOEClassifier(16, no_classes)
    model.fit(x=x, y=y, max_depth=4)
    print('If you see this line, memory test passed! :)')
    moe_results = model.predict(x)
    prec_moe = accuracy_score(y, moe_results)
    print('Accuracy: {0}'.format(prec_moe))


def test_mountaincar(create_model):
    from sklearn.model_selection import train_test_split
    import os

    def train_test_val_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=50)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=50)
        return X_train, X_test, X_val, y_train, y_test, y_val
        
    basepath = os.path.dirname(__file__)
    X = np.loadtxt(open(os.path.join(basepath, "./test_data/mountaincar_observations.csv"),
                        "rb"), delimiter=",")
    y = np.loadtxt(open(os.path.join(basepath, "./test_data/mountaincar_actions.csv"),
                                     "rb"), delimiter=",").astype(int)
    
    m = X.shape[0]
    assert X.shape == (m, 2)
    assert y.shape == (m,)
    
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X, y)
    number_of_classes = 3

    for num_experts in range(2, 4):
        for experts_depth in range(5):
            model = create_model(num_experts, number_of_classes)
            model.train(x=X_train, 
                        y=y_train,
                        x_test=X_val, 
                        y_test=y_val,
                        max_depth=experts_depth,
                        regularization_mode=0, 
                        log_frequency=None,
                        stop_count=None,
                        init_learning_rate=1.,
                        learning_rate_decay=1.,
                        max_epoch=50,
                        return_best_epoch=True,
                        gradually_increase_max_depth=True,
                        train_gating_multiple_times=True)
            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            test_accuracy = accuracy_score(y_test, model.predict(X_test))
            print('experts={},experts_depth={}, train_acc={:.2f},test_acc={:.2f}'.
                  format(num_experts, experts_depth, train_accuracy, test_accuracy))
    

if __name__ == '__main__':
    test_accuracy(lambda num_experts, num_classes:
                  MOETClassifier(experts_no=num_experts, no_class=num_classes))
