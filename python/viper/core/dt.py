# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from ..util.log import *


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


def save_dt_policy_viz(dt_policy, dirname, fname,
                       feature_names=None, action_names=None):
    save_dt_viz(dt_policy.tree, dirname, fname,
                feature_names, action_names)


def save_dt_viz(dt, dirname, fname,
                feature_names=None, action_names=None):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if not feature_names or not action_names:
        export_graphviz(dt, dirname + '/' + fname)
    else:
        export_graphviz(dt,
                        out_file=dirname + '/' + fname,
                        feature_names=feature_names,
                        class_names=action_names,
                        filled=True,
                        rounded=True,
                        special_characters=True)


class DTPolicy:
    def __init__(self, max_depth, regression=False):
        # Viper originally only supports discrete action space, and thus uses classifier.
        # We extend it to support continuous space as well, and thus add option for regression model.
        self.max_depth = max_depth
        self.regression = regression

    def get_node_count(self):
        """Get number of nodes in a DT."""
        return self.tree.tree_.node_count

    def get_depth(self):
        """Get depth of DT."""
        return self.tree.tree_.max_depth

    def fit(self, obss, acts):
        if not self.regression:
            self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        else:
            self.tree = DecisionTreeRegressor(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss,
                                                                        acts,
                                                                        train_frac)
        self.fit(obss_train, acts_train)
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)),
            INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)),
            INFO)
        log('Number of nodes in DT: {}'.format(self.tree.tree_.node_count),
            INFO)
        log('Depth of DT: {}'.format(self.tree.tree_.max_depth), INFO)

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone
