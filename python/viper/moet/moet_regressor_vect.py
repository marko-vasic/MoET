"""
MOE Regressor model with multiple outputs.
"""

# Comments:
# With 5000 examples training already very slow.
# Inverse of Singular Matrix message still appears from time to time.
#   One such case is when number of experts is one (this is arguable
#   not a valid case, but it might be good for debugging).
# With increasing number of experts, error does not decrease much
#   (sometimes even jumps).

import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTC
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import StandardScaler

from moet_base import MOETBase

class MOERegressorVect(MOETBase):

    @staticmethod
    def normpdf(y, var, mean):
        denom = ((2 * math.pi) ** y.shape[1] * np.linalg.det(var)) ** .5
        inv = np.linalg.inv(var)
        diff = y - mean
        num = np.exp(
            -0.5 * np.tensordot(np.tensordot(diff, inv, axes=(1, 1)), diff,
                                axes=(1, 1))[0, :])
        return num / denom

    def __init__(self, experts_no, default_type=np.float32):
        super(MOETRegressorVect, self).__init__(experts_no=experts_no,
                                                default_type=default_type)

    def predict(self, x_test, no_outputs):
        x_test = self.scaler.transform(x_test)
        x_test = np.append(x_test, np.ones([x_test.shape[0], 1]), axis=1)
        mu_test = np.zeros([x_test.shape[0], no_outputs, self.experts_no])
        gating = self.softmax(x_test, self.tetag, self.experts_no)
        for j in range(self.experts_no):
            mu_test[:, :, j] = self.dtc_list[j].predict(x_test) * np.tile(
                gating[:, j], (no_outputs, 1)).T
        result = np.sum(mu_test, axis=2)
        return result

    def predict_hard(self, x_test, no_outputs):
        x_test = self.scaler.transform(x_test)
        x_test = np.append(x_test, np.ones([x_test.shape[0], 1]), axis=1)
        mu_test = np.zeros([x_test.shape[0], no_outputs, self.experts_no])
        gating = self.softmax(x_test, self.tetag, self.experts_no)
        index = np.argmax(gating, axis=1)
        row = np.arange(gating.shape[0])
        gating = np.zeros(gating.shape)
        gating[row, index] = 1
        for j in range(self.experts_no):
            mu_test[:, :, j] = self.dtc_list[j].predict(x_test) * np.tile(
                gating[:, j], (no_outputs, 1)).T
        result = np.sum(mu_test, axis=2)
        return result

    def fit(self, x, y, max_epoch=300, max_depth=4, init_learning_rate=4,
            learning_rate_decay=0.95):
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        x = np.append(x, np.ones([x.shape[0], 1]), axis=1)
        self.tetag = np.random.randn(x.shape[1] * self.experts_no)
        learnRate = [init_learning_rate * (learning_rate_decay **
                                           max(float(i), 0.0)) for i in
                     range(max_epoch)]
        mu = np.zeros([x.shape[0], y.shape[1], self.experts_no])
        val = np.zeros([y.shape[1], y.shape[1], self.experts_no])
        pdf = np.zeros([x.shape[0], self.experts_no])

        for i in range(max_epoch):
            gating = self.softmax(x, self.tetag, self.experts_no)
            self.dtc_list = []
            for j in range(self.experts_no):
                self.dtc_list.append([])
                self.dtc_list[j] = DTC(max_depth=max_depth)
                self.dtc_list[j].fit(x, y, sample_weight=gating[:, j].T)
                mu[:, :, j] = self.dtc_list[j].predict(x)
                values = y - mu[:, :, j]
                val[:, :, j] = np.cov(values.T)
                pdf[:, j] = self.normpdf(y, val[:, :, j], mu[:, :, j])
            h = self.h_fun(gating, pdf)
            dsdtetag = self.ds_dtetag(x, self.tetag, self.experts_no)
            e = self.e_fun(h, gating, dsdtetag)
            R = self.R_fun(gating, dsdtetag, self.experts_no)
            if np.linalg.cond(R) < 1e7:
                self.tetag = self.tetag + learnRate[i] * np.linalg.inv(R).dot(e)
            else:
                self.tetag = self.tetag + learnRate[i] * e
            self.gating = gating
            self.gating = gating


if __name__ == '__main__':
    No_inst = 50
    no_outputs = 2
    x = np.random.rand(No_inst, 4)
    y = np.random.randn(No_inst, no_outputs)
    # x = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    # y = np.array([1., 2., 2., 1., 1., 2.])

    dtree = DTC(max_depth=4)
    dtree.fit(x, y)
    dt_results = dtree.predict(x)

    mse_dt = mean_squared_error(y, dt_results)
    print('Mean squared error DT: {0}'.format(mse_dt))

    for num_experts in range(2, 21):
        model = MOERegressorVect(num_experts)
        model.fit(x=x, y=y, max_depth=4)
        moe_results = model.predict(x, no_outputs)
        moe_results_hard = model.predict_hard(x, no_outputs)

        mse_moe = mean_squared_error(y, moe_results)
        mse_moe_hard = mean_squared_error(y, moe_results_hard)
        print('Mean squared error MOE ({0} experts): {1}, hard {2}'.format(
            num_experts, mse_moe, mse_moe_hard))
