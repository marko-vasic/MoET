from viper.core.dt import accuracy
from viper.core.dt import split_train_test
from viper.util.log import *
from viper.moet.moet_classifier import MOETClassifier
from viper.moet.moet_classifier_new import MOETClassifierNew
from viper.moet.moet_regressor import MOETRegressor
import copy
import math


class MOEPolicy:
    # This is added for backward compatibility with old pickle files.
    # Can be deleted when pickle files are recreated.
    hard_prediction = False
    num_classes = 2
    max_epoch = 80
    init_learning_rate = 2
    learning_rate_decay = 0.95
    log_frequency = 10
    use_adam_optimizer = False

    def __init__(self,
                 experts_no,
                 dts_depth,
                 num_classes,
                 hard_prediction,
                 max_epoch,
                 init_learning_rate,
                 learning_rate_decay,
                 log_frequency,
                 stop_count,
                 regularization_mode,
                 use_new_formula,
                 use_adam_optimizer):
        """
        Args:
          experts_no: Number of experts.
          dts_depth: Depths of expert Decision Trees.
          num_classes: Number of classes in the output space (0 -
            special value used for regression).
          hard_prediction: Whether to use hard prediction with MOE.
          max_epoch: Maximum number of epochs in MOE training.
          init_learning_rate: Initial learning rate in MOE.
          learning_rate_decay: Learning rate decay.
          log_frequency: Log frequency in MOE.
          use_new_formula: Whether to use a new formula for MOET.
        """
        self.experts_no = experts_no
        self.dts_depth = dts_depth
        self.num_classes = num_classes
        self.hard_prediction = hard_prediction
        self.max_epoch = max_epoch
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.log_frequency = log_frequency
        self.stop_count = stop_count
        self.regularization_mode = regularization_mode
        self.use_new_formula = use_new_formula
        self.use_adam_optimizer = use_adam_optimizer

    def get_node_count(self):
        """
        Get number of nodes in all DTs.
        Gating is calculated as Mat.ceil(log_2(num_experts)).
        """
        sum = 0
        for tree in self.moe.dtc_list:
            sum += tree.tree_.node_count
        sum += int(math.ceil(math.log(self.experts_no, 2)))
        return sum

    def get_depth(self):
        max_dt_depth = 0
        for tree in self.moe.dtc_list:
            max_dt_depth = max(tree.tree_.max_depth, max_dt_depth)
        experts_num = len(self.moe.dtc_list)
        return int(math.ceil(math.log(experts_num, 2))) + max_dt_depth

    def _fit(self, obss, acts, obss_test, acts_test):
        if self.num_classes > 0:
            if not self.use_new_formula:
                self.moe = MOETClassifier(experts_no=self.experts_no,
                                          no_class=self.num_classes)
            else:
                self.moe = MOETClassifierNew(experts_no=self.experts_no,
                                             no_class=self.num_classes,
                                             use_adam=self.use_adam_optimizer)
        else:
            self.moe = MOETRegressor(experts_no=self.experts_no)

        self.moe.train(obss, acts,
                       obss_test, acts_test,
                       max_depth=self.dts_depth,
                       max_epoch=self.max_epoch,
                       init_learning_rate=self.init_learning_rate,
                       learning_rate_decay=self.learning_rate_decay,
                       log_frequency=self.log_frequency,
                       stop_count=self.stop_count,
                       regularization_mode=self.regularization_mode)

    def train(self, obss, acts, train_frac):
        # TODO:
        # Issue with the split when sampling using q-value re-weighting is done
        # is that the obss set can contain multiple instances of the same point.
        # Thus, the train/test split could potentially contain the same points.
        # This relates to DAGGER (rl.py)
        obss_train, acts_train, obss_test, acts_test = split_train_test(
            obss, acts, train_frac)
        self._fit(obss_train, acts_train, obss_test, acts_test)
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)),
            INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)),
            INFO)

    def predict(self, obss):
        if not self.hard_prediction:
            return self.moe.predict(obss)
        else:
            return self.moe.predict_hard(obss)

    def clone(self):
        clone = MOEPolicy(experts_no=self.experts_no,
                          dts_depth=self.dts_depth,
                          num_classes=self.num_classes,
                          hard_prediction=self.hard_prediction,
                          max_epoch=self.max_epoch,
                          init_learning_rate=self.init_learning_rate,
                          learning_rate_decay=self.learning_rate_decay,
                          log_frequency=self.log_frequency,
                          stop_count=self.stop_count,
                          regularization_mode=self.regularization_mode,
                          use_new_formula=self.use_new_formula,
                          use_adam_optimizer=self.use_adam_optimizer)
        clone.moe = copy.copy(self.moe)
        return clone
