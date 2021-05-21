"""
MOE Regressor model with one output.
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
from sklearn.preprocessing import StandardScaler
from viper.moet.moet_base import MOETBase
import math

ADAM_EPSILON = 1e-7
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999


class MOETRegressor(MOETBase):

    @staticmethod
    def normpdf(y, var, mean):
        denom = (2 * math.pi * var) ** .5
        num = np.exp(-(y - mean) ** 2 / (2 * var))
        return num / denom

    def __init__(self, experts_no, default_type=np.float32, use_adam=True):
        super(MOETRegressor, self).__init__(
            experts_no=experts_no,
            default_type=default_type,
        )
        self.use_adam = use_adam

    def predict(self, x_test):
        x_test = self.scaler.transform(x_test)
        x_test = np.append(x_test, np.ones([x_test.shape[0], 1]), axis=1)
        mu_test = np.zeros([x_test.shape[0], self.experts_no])
        gating = self.softmax(x_test, self.tetag, self.experts_no)
        for j in range(self.experts_no):
            mu_test[:, j] = self.dtc_list[j].predict(x_test)
        result = np.sum(gating * mu_test, axis=1)
        return result

    def fit(self, x, y, max_epoch=300, max_depth=4, init_learning_rate=2,
            learning_rate_decay=0.95):
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        x = np.append(x, np.ones([x.shape[0], 1]), axis=1)
        self.tetag = np.random.randn(x.shape[1] * self.experts_no)
        learnRate = [init_learning_rate * (learning_rate_decay **
                                           max(float(i), 0.0)) for i in
                     range(max_epoch)]
        mu = np.zeros([x.shape[0], self.experts_no])
        val = np.zeros(self.experts_no)
        pdf = np.zeros([x.shape[0], self.experts_no])

        for epoch_id in range(max_epoch):
            gating = self.softmax(x, self.tetag, self.experts_no)
            self.dtc_list = []
            for j in range(self.experts_no):
                self.dtc_list.append([])
                self.dtc_list[j] = DTC(max_depth=max_depth)
                self.dtc_list[j].fit(x, y, sample_weight=gating[:, j].T)
                mu[:, j] = self.dtc_list[j].predict(x)
                val[j] = np.cov(y - mu[:, j])
                pdf[:, j] = MOETRegressor.normpdf(y, val[j], mu[:, j])
            h = self.h_fun(gating, pdf)
            dsdtetag = self.ds_dtetag(x, self.tetag, self.experts_no)
            e = self.e_fun(h, gating, dsdtetag)
            # R = self.R_fun(gating, dsdtetag, self.experts_no)
            # if np.linalg.cond(R) < 1e7:
            #     self.tetag = self.tetag + learnRate[i] * np.linalg.inv(R).dot(e)
            # else:
            #     self.tetag = self.tetag + learnRate[i] * e
            # self.gating = gating
            # self.gating = gating
            if not self.use_adam:
                R = self.R_fun(gating, dsdtetag, self.experts_no)
                if np.linalg.cond(R) < 1e7:
                    self.tetag += learnRate[epoch_id] * np.linalg.inv(R).dot(e)
                else:
                    self.tetag += learnRate[epoch_id] * e
                return R
            else:
                t = epoch_id + 1
                if t == 1:
                    self.adam_m = 0
                    self.adam_v = 0
                
                self.adam_m = ADAM_BETA_1 * self.adam_m + (1 - ADAM_BETA_1) * e
                self.adam_v = ADAM_BETA_2 * self.adam_v + (1 - ADAM_BETA_2) * np.power(e, 2)
                m_hat = self.adam_m / (1 - np.power(ADAM_BETA_1, t))
                v_hat = self.adam_v / (1 - np.power(ADAM_BETA_2, t))
                self.tetag += learnRate[epoch_id] * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)


if __name__ == '__main__':
    No_inst = 50
    x = np.random.rand(No_inst, 2)
    y = np.random.randn(No_inst)
    # x = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    # y = np.array([1., 2., 2., 1., 1., 2.])

    dtree = DTC(max_depth=4)
    dtree.fit(x, y)
    dt_results = dtree.predict(x)

    mse_dt = np.square(dt_results - y).mean(axis=0)
    print('Mean squared error DT: {0}'.format(mse_dt))

    for num_experts in range(2, 21):
        model = MOETRegressor(num_experts)
        model.fit(x=x, y=y, max_depth=4, max_epoch=500)
        moe_results = model.predict(x)

        mse_moe = np.square(moe_results - y).mean(axis=0)
        print('Mean squared error MOE ({0} experts): {1}'.format(num_experts,
                                                                 mse_moe))
