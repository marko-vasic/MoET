from viper.util.serialization import load_policy
from viper.core.rl import get_rollouts
from viper.evaluation.util import *
from viper.core.dt import save_dt_policy_viz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from viper.core.rl import test_policy
from viper.core.compare_policy import ComparePolicy
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
import os
from viper.util.util import ensure_dir_exists
from viper.util.util import ensure_parent_exists

# Directory containing best policies.
BEST_POLICIES_DIR = '/home/unk/projects/explainableRL/viper/data/mountaincar/best'
# Depths of decision trees.
DEPTHS = [1, 2, 3, 4, 5]
# Experts used for MOE.
EXPERTS = [2, 4, 8]


class MountainCarVisualization(object):

    def __init__(self, interpolaction_steps=100):
        self.interpolation_steps = float(interpolaction_steps)
        self.min_position = -1.2
        self.max_position = 0.6
        self.step_position = ((self.max_position - self.min_position)
                              / self.interpolation_steps)
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.step_velocity = ((self.max_velocity - self.min_velocity)
                              / self.interpolation_steps)

        self.rl_policy = load_rl_policy('mountaincar')

    def visualize_gate(self,
                       moe_policy,
                       hard_prediction=False,
                       out_file=None):

        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        moe = moe_policy.moe
        experts_num = moe.experts_no

        x, y = np.meshgrid(np.arange(min_position, max_position + step_position,
                                     step_position),
                           np.arange(min_velocity, max_velocity + step_velocity,
                                     step_velocity))
        z = np.zeros((experts_num, x.shape[0] - 1, x.shape[1] - 1),
                     dtype=np.float)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]
                if not hard_prediction:
                    probs = moe.predict_expert_proba(
                        np.array([[position, velocity]]))[0]
                else:
                    probs = np.zeros(experts_num)
                    expert = \
                    moe.predict_expert(np.array([[position, velocity]]))[0]
                    probs[expert] = 1.0
                for expert_id in range(experts_num):
                    z[expert_id, i, j] = probs[expert_id]

        z_min = 0.0
        z_max = 1.0

        fig, axes = plt.subplots(nrows=experts_num,
                                 figsize=(5, 3.5 * experts_num))

        for expert_id in range(experts_num):
            ax = axes[expert_id]
            c = ax.pcolormesh(x, y, z[expert_id, :, :],
                              cmap='Blues', vmin=z_min, vmax=z_max)
            ax.set_title('gating for expert {}'.format(expert_id))
            ax.set_xlabel('position')
            ax.set_ylabel('velocity')
            # set the limits of the plot to the limits of the data
            ax.axis([min_position, max_position, min_velocity, max_velocity])
            # fig.colorbar(c, ax=ax)

        # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
        #                     hspace=0.25, wspace=0.35)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_policy(self, policy, out_file=None):
        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        x, y = np.meshgrid(np.arange(min_position, max_position + step_position,
                                     step_position),
                           np.arange(min_velocity, max_velocity + step_velocity,
                                     step_velocity))
        z = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.int)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]
                decision = policy.predict(np.array([[position, velocity]]))[
                               0] - 1
                z[i, j] = decision

        z_min = -1
        z_max = 1

        fig, ax = plt.subplots(figsize=(5, 3.5))

        colorsList = ['green', 'yellow', 'blue']
        cmap = colors.ListedColormap(colorsList)

        c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])

        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_qvalues(self, out_file=None):
        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        x, y = np.meshgrid(np.arange(min_position, max_position + step_position,
                                     step_position),
                           np.arange(min_velocity, max_velocity + step_velocity,
                                     step_velocity))
        z = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]

                qvalues = \
                self.rl_policy.predict_q(np.array([[position, velocity]]))[0]
                qimportance = np.max(qvalues) - np.min(qvalues)
                z[i, j] = qimportance

        # Scale to range [0-1].
        z = (z - np.min(z)) / np.ptp(z)

        z_min = 0
        z_max = 1

        fig, ax = plt.subplots(figsize=(5, 3.5))

        c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=z_min, vmax=z_max)
        ax.set_title('Policy')
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])

        fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_mispredictions(self, policy1, policy2, out_file=None):
        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        x, y = np.meshgrid(
            np.arange(min_position, max_position + step_position,
                      step_position),
            np.arange(min_velocity, max_velocity + step_velocity,
                      step_velocity))
        # Indicator array with a value 1 in a case of misprediction,
        # and 0 in a case of agreement.
        z = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.int)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]
                decision1 = policy1.predict(np.array([[position, velocity]]))[0]
                decision2 = policy2.predict(np.array([[position, velocity]]))[0]
                value = 1 if decision1 != decision2 else 0
                z[i, j] = value

        fig, ax = plt.subplots(figsize=(5, 3.5))

        z_min = 0
        z_max = 1
        colorsList = ['blue', 'red']
        cmap = colors.ListedColormap(colorsList)

        c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            mispredictions = (float(np.sum(z)) / z.size) * 100
            with open(out_file + "_info", "w") as f:
                f.write("mispredictions: {0:.2f}%".format(mispredictions))
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_qweighted_mispredictions(self, policy1, policy2,
                                           out_file=None):
        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        x, y = np.meshgrid(
            np.arange(min_position, max_position + step_position,
                      step_position),
            np.arange(min_velocity, max_velocity + step_velocity,
                      step_velocity))
        # Array with q importance (difference between max and min q value in
        # a state).
        q = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)
        # Indicator array with a value of 1 in a case of misprediction,
        # and 0 in a case of agreement.
        ind = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]

                qvalues = \
                self.rl_policy.predict_q(np.array([[position, velocity]]))[
                    0]
                qimportance = np.max(qvalues) - np.min(qvalues)
                q[i, j] = qimportance

                decision1 = policy1.predict(np.array([[position, velocity]]))[0]
                decision2 = policy2.predict(np.array([[position, velocity]]))[0]
                value = 1 if decision1 != decision2 else 0
                ind[i, j] = value

        # Scale to range [0-1].
        q = (q - np.min(q)) / np.ptp(q)
        z = ind * q
        # Powering is done in order to enable better visualization.
        # z_powered = np.power(z, 1. / 3)

        # # Softmax on q values
        # q = np.exp(q) / np.sum(np.exp(q))
        # z *= q
        # z_powered = np.power(z, 1. / 20)

        z_min = 0
        z_max = 1

        fig, ax = plt.subplots(figsize=(5, 3.5))
        c = ax.pcolormesh(x, y, z, cmap='Reds',
                          vmin=z_min, vmax=z_max)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])
        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            mispredictions = (float(np.sum(ind)) / z.size) * 100
            weighted = (float(np.sum(z)) / np.sum(q)) * 100
            with open(out_file + "_info", "w") as f:
                f.write("mispredictions: {0:.2f}%\n".format(mispredictions))
                f.write("mispredictions weighted: {0:.2f}".format(weighted))
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_qweighted_misprediction_differences(self, policy1, policy2,
                                                      out_file=None):
        # Q weighted differences compared to rl agent
        min_position = self.min_position
        max_position = self.max_position
        step_position = self.step_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity
        step_velocity = self.step_velocity

        x, y = np.meshgrid(
            np.arange(min_position, max_position + step_position,
                      step_position),
            np.arange(min_velocity, max_velocity + step_velocity,
                      step_velocity))
        # Array with q importance (difference between max and min q value in
        # a state).
        q = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)
        # Indicator array with a value of 1 in a case of misprediction,
        # and 0 in a case of agreement.
        z1 = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)
        z2 = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.float)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                position = x[i][j]
                velocity = y[i][j]

                qvalues = \
                    self.rl_policy.predict_q(np.array([[position, velocity]]))[
                        0]
                qimportance = np.max(qvalues) - np.min(qvalues)
                q[i, j] = qimportance

                gold = self.rl_policy.predict(np.array([[position, velocity]]))[
                    0]
                decision1 = policy1.predict(np.array([[position, velocity]]))[0]
                decision2 = policy2.predict(np.array([[position, velocity]]))[0]

                if gold != decision1:
                    z1[i, j] = 1
                if gold != decision2:
                    z2[i, j] = 1

        # Scale to range [0-1].
        q = (q - np.min(q)) / np.ptp(q)
        z1 *= q
        z2 *= q

        z = z2 - z1  # z1 - z2

        z_min = -1
        z_max = 1

        fig, ax = plt.subplots(figsize=(5, 3.5))
        c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])
        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            diff = (float(np.sum(z)) / np.sum(q)) * 100
            with open(out_file + "_info", "w") as f:
                f.write("differences sum: {0:.2f}".format(diff))
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def scatter_plot(self, x, y, z, out_file=None):
        """
        Make a 2d plot where coordinates are stored in x and y arrays,
        and values in z array.
        Sizes of all arrays must match.

        NOTE: This is specialized for visualization of points in mountaincar.
        """

        min_position = self.min_position
        max_position = self.max_position
        min_velocity = self.min_velocity
        max_velocity = self.max_velocity

        fig, ax = plt.subplots(figsize=(5, 3.5))

        colorsList = ['green', 'yellow', 'blue']
        cmap = colors.ListedColormap(colorsList)

        ax.scatter(x, y, c=z, cmap=cmap,
                   s=(rcParams['lines.markersize'] ** 2) / 5,
                   alpha=0.7)

        ax.set_title('Title')
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')

        # set the limits of the plot to the limits of the data
        ax.axis([min_position, max_position, min_velocity, max_velocity])

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_viper_policies(self, out_dir, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'ViperPlus')
        for depth in depths:
            policy = load_policy(dirname, 'dt_policy_d{}.pk'.format(depth))
            out_file = os.path.join(out_dir,
                                    'policy_viper_d{}.pdf'.format(depth))
            self.visualize_policy(policy, out_file)

    def visualize_moe_policies(self, out_dir, hard_prediction,
                               experts=EXPERTS, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'MOE')
        for experts in experts:
            for depth in depths:
                policy = load_policy(dirname, 'moe_policy_e{}_d{}.pk'.
                                     format(experts, depth))
                policy.hard_prediction = hard_prediction
                policy_name = 'moe' if not hard_prediction else 'moehard'
                out_file = os.path.join(out_dir,
                                        'policy_{}_e{}_d{}.pdf'.
                                        format(policy_name, experts, depth))
                self.visualize_policy(policy, out_file)

    def visualize_viper_mispredictions(self, out_dir, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'ViperPlus')
        for depth in depths:
            policy = load_policy(dirname, 'dt_policy_d{}.pk'.format(depth))
            out_file = os.path.join(out_dir,
                                    'mispredictions_viper_d{}.pdf'.format(
                                        depth))
            self.visualize_mispredictions(self.rl_policy, policy,
                                          out_file=out_file)

            out_file = os.path.join(out_dir,
                                    'mispredictions_qweighted_viper_d{}.pdf'.format(
                                        depth))
            self.visualize_qweighted_mispredictions(self.rl_policy, policy,
                                                    out_file=out_file)

    def visualize_moe_mispredictions(self, out_dir, hard_prediction,
                                     experts=EXPERTS, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'MOE')
        for experts_num in experts:
            for depth in depths:
                policy = load_policy(dirname, 'moe_policy_e{}_d{}.pk'.
                                     format(experts_num, depth))
                policy.hard_prediction = hard_prediction
                policy_name = 'moe' if not hard_prediction else 'moehard'
                out_file = os.path.join(out_dir,
                                        'mispredictions_{}_e{}_d{}.pdf'.
                                        format(policy_name, experts_num, depth))
                self.visualize_mispredictions(self.rl_policy, policy,
                                              out_file=out_file)
                out_file = os.path.join(out_dir,
                                        'mispredictions_qweighted_{}_e{}_d{}.pdf'.
                                        format(policy_name, experts_num, depth))
                self.visualize_qweighted_mispredictions(self.rl_policy, policy,
                                                        out_file=out_file)

    def visualize_gates_soft(self, out_dir, experts=EXPERTS, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'MOE')
        for experts_num in experts:
            for depth in depths:
                policy = load_policy(dirname, 'moe_policy_e{}_d{}.pk'.
                                     format(experts_num, depth))
                policy.hard_prediction = False
                out_file = os.path.join(out_dir,
                                        'gating_e{}_d{}.pdf'.
                                        format(experts_num, depth))
                self.visualize_gate(moe_policy=policy,
                                    hard_prediction=False,
                                    out_file=out_file)

    def visualize_gates_hard(self, out_dir, experts=EXPERTS, depths=DEPTHS):
        dirname = os.path.join(BEST_POLICIES_DIR, 'MOEHard')
        for experts_num in experts:
            for depth in depths:
                policy = load_policy(dirname, 'moe_policy_e{}_d{}.pk'.
                                     format(experts_num, depth))
                policy.hard_prediction = False
                out_file = os.path.join(out_dir,
                                        'gatinghard_e{}_d{}.pdf'.
                                        format(experts_num, depth))
                self.visualize_gate(moe_policy=policy,
                                    hard_prediction=True,
                                    out_file=out_file)


class MOEDTGatePolicy:

    def __init__(self, moe_policy, dt_gate):
        self.moe = moe_policy.moe
        self.dt_gate = dt_gate

    def _fit(self, obss, acts, obss_test, acts_test):
        pass

    def train(self, obss, acts, train_frac):
        pass

    def predict(self, obss):
        experts = self.dt_gate.predict(obss)
        return self.moe.predict_with_expert(obss, experts)


class MOEAnalyticalPolicy:

    def __init__(self, moe):
        # MOE Classifier.
        self.moe = moe.moe
        # DT experts.
        self.experts = moe.moe.dtc_list
        # Number of experts.
        self.E = len(self.experts)
        self.F = moe.moe.tetag.shape[0] / self.E
        # Gate parameters
        # dimensions: (F, E).
        self.tetag = moe.moe.tetag.reshape(self.E, self.F).T

    def get_expert(self, x_normalized):
        """Find expert to be used for a given observation."""
        exponents = np.matmul(x_normalized, self.tetag)
        return np.argmax(exponents)

    def predict(self, obss):
        predictions = list()
        for observation in obss:
            normalized = self.moe._normalize_x([observation])
            expert_id = self.get_expert(normalized)
            expert_truth = self.moe.predict_expert([observation])[0]
            if expert_id != expert_truth:
                raise Exception('Incorrect')

            predictions.append(self.experts[expert_id].predict(normalized)[0])
        return np.array(predictions)


class CartpoleE2D1Policy:

    def __init__(self, moe):
        self.experts = moe.moe.dtc_list
        self.mean = np.array([-0.0898521, -0.04071702, 0.00027681, 0.01041096])
        self.scale = np.array([0.4222427, 0.3902054, 0.02939676, 0.23061956])
        self.tetag = np.array([[48.466267, -47.64857],
                               [148.83385, -147.38306],
                               [32.2918, -31.851715],
                               [306.46344, -304.98468],
                               [41.136673, -39.662857]])

    def get_expert(self, x_normalized):
        """Find expert to be used for a given observation."""
        exponents = np.matmul(x_normalized, self.tetag)
        return np.argmax(exponents)

    def predict(self, obss):
        predictions = list()
        for observation in obss:
            observation -= self.mean
            observation /= self.scale
            normalized = np.append(observation, [1])
            normalized = [normalized]
            expert_id = self.get_expert(normalized)

            # gdivs = np.array([[114.782960132, -112.846403265],
            # [381.424372907, -377.706356703],
            # [1098.481601374, -1083.51107401],
            # [1328.870109717, -1322.457990987],
            # [41.136673, -39.662857]])
            # m = np.array([-0.0898521, -0.04071702, 0.00027681, 0.01041096, 0.])
            # res = (np.matmul(np.append(observation, [1.]), gdivs)
            #        - np.matmul(m, gdivs))
            # expert_id = np.argmax(res)

            if expert_id == 0:
                predictions.append([1.])
            else:
                predictions.append([0.])

        return np.array(predictions)


class CartpoleVisualization(object):
    def __init__(self, interpolation_steps=100):
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.interpolation_steps = float(interpolation_steps)
        self.min_cp = -4.8
        self.max_cp = 4.8
        self.step_cp = ((self.max_cp - self.min_cp) / self.interpolation_steps)
        self.min_pa = -0.418
        self.max_pa = 0.418
        self.step_pa = ((self.max_pa - self.min_pa) / self.interpolation_steps)

        self.min_pv = -0.15
        self.max_pv = 0.15
        self.step_pv = ((self.max_pv - self.min_pv) / self.interpolation_steps)
        self.min_cv = -2.
        self.max_cv = 2.
        self.step_cv = ((self.max_cv - self.min_cv) / self.interpolation_steps)


        self.rl_policy = load_rl_policy('cartpole')

    def visualize_policy(self, policy, out_file=None):
        cp = 0.
        pa = 0.

        x, y = np.meshgrid(np.arange(self.min_pv,
                                     self.max_pv + self.step_pv,
                                     self.step_pv),
                           np.arange(self.min_cv,
                                     self.max_cv + self.step_cv,
                                     self.step_cv))
        z = np.zeros((x.shape[0] - 1, x.shape[1] - 1), dtype=np.int)

        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                pv = x[i][j]
                cv = y[i][j]
                decision = policy.predict(
                    np.array([[cp, cv, pa, pv]]))[0]
                z[i, j] = decision

        z_min = 0
        z_max = 1

        fig, ax = plt.subplots(figsize=(5, 3.5))

        colorsList = ['green', 'blue']
        cmap = colors.ListedColormap(colorsList)

        c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        ax.set_xlabel('pole angular velocity')
        ax.set_ylabel('cart velocity')
        # set the limits of the plot to the limits of the data
        ax.axis([self.min_pv, self.max_pv, self.min_cv, self.max_cv])

        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)

    def visualize_policy_3D(self, policy, out_file=None):
        cp = 0.
        x, y, d = np.meshgrid(np.arange(self.min_pv,
                                        self.max_pv,
                                        self.step_pv),
                              np.arange(self.min_cv,
                                        self.max_cv,
                                        self.step_cv),
                              np.arange(self.min_pa,
                                        self.max_pa,
                                        self.step_pa))
        predictions = np.zeros((x.shape[0],
                                x.shape[1],
                                x.shape[2]),
                               dtype=np.int)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    pv = x[i][j][k]
                    cv = y[i][j][k]
                    pa = d[i][j][k]
                    decision = policy.predict(
                        np.array([[cp, cv, pa, pv]]))[0]
                    predictions[i, j, k] = decision

        from mpl_toolkits.mplot3d import axes3d, \
            Axes3D  # <-- Note the capitalization!
        fig = plt.figure()

        ax = Axes3D(fig)

        colorsList = ['green', 'blue']
        cmap = colors.ListedColormap(colorsList)

        ax.scatter(x, y, d, c=predictions.flatten(), cmap=cmap)

        ax.set_xlabel('pole angular velocity')
        ax.set_ylabel('cart velocity')
        ax.set_zlabel('pole angle')
        # # set the limits of the plot to the limits of the data
        # ax.axis([self.min_pv, self.max_pv,
        #          self.min_cv, self.max_cv,
        #          self.min_pa, self.max_pa])

        # fig.colorbar(c, ax=ax)

        fig.tight_layout()
        if out_file:
            ensure_parent_exists(out_file)
            plt.savefig(out_file)
        else:
            plt.show()
        plt.close(fig)


def evaluate(student_policy, subject_name, n_test_rollouts):
    teacher_policy = load_rl_policy(subject_name)

    cmp_policy = ComparePolicy(teacher_policy, student_policy)
    env = create_gym_env(subject_name)
    reward = test_policy(env, cmp_policy, n_test_rollouts)
    mispredictions = cmp_policy.mispredictions_ratio() * 100

    print('reward: {}'.format(reward))
    print('miss: {}'.format(mispredictions))


def train_gate_dt(env, policy, n_rollouts, max_depth):
    # Collect rollouts
    # Train DT
    trace = get_rollouts(env, policy, False, n_rollouts)
    obss = []
    # TODO: Maybe state-transformer needed.
    obss.extend(obs for obs, _, _ in trace)
    obss = np.array(obss)

    gate_outputs = policy.moe.predict_expert(obss)

    dt_gate = DecisionTreeClassifier(max_depth=max_depth)
    dt_gate.fit(obss, gate_outputs)

    accuracy = accuracy_score(dt_gate.predict(obss), gate_outputs)
    print("DT gate accuracy: {}".format(accuracy))

    moedtgate_policy = MOEDTGatePolicy(policy, dt_gate)

    return moedtgate_policy


def train_dt_gate():
    subject_name = 'cartpole'
    env = create_gym_env(subject_name)
    policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/cartpole/0/MOE',
        'moe_policy_e2_d1.pk')
    n_rollouts = 100
    max_depth = 3
    student_policy = train_gate_dt(env, policy, n_rollouts, max_depth)
    print("mispredictions with new model")
    evaluate(student_policy, subject_name, n_rollouts)
    print("mispredictions with moe model")
    evaluate(policy, subject_name, n_rollouts)


def print_gate_weights(dirname, filename):
    moe_policy = load_policy(dirname, filename)
    print(moe_policy.moe.tetag)


def visualize_selected():
    out_dir = '/home/unk/Downloads/nips/selected'

    # Update font size
    # import matplotlib
    # matplotlib.rcParams.update({'font.size': 12})

    visualizer = MountainCarVisualization(200)

    viper_policy = load_policy(os.path.join(BEST_POLICIES_DIR, 'ViperPlus'),
                               'dt_policy_d3.pk')
    moehard_policy = load_policy(os.path.join(BEST_POLICIES_DIR, 'MOEHard'),
                                 'moe_policy_e4_d3.pk')
    moehard_policy.hard_prediction = True

    policy2 = load_policy(os.path.join(BEST_POLICIES_DIR, 'MOEHard'),
                          'moe_policy_e4_d1.pk')
    visualizer.visualize_gate(moe_policy=policy2,
                              hard_prediction=True,
                              out_file=os.path.join(out_dir, 'gates_e4_d1.pdf'))
    return

    visualizer.visualize_qweighted_misprediction_differences(viper_policy,
                                                             moehard_policy,
                                                             out_file=os.path.join(
                                                                 out_dir,
                                                                 'qweighted_mispr_diff.pdf'))

    visualizer.visualize_gate(moe_policy=moehard_policy,
                              hard_prediction=True,
                              out_file=os.path.join(out_dir, 'gates.pdf'))

    visualizer.visualize_viper_mispredictions(out_dir, depths=[3])

    visualizer.visualize_moe_mispredictions(out_dir, hard_prediction=True,
                                            experts=[4], depths=[3])

    visualizer.visualize_policy(visualizer.rl_policy,
                                out_file=os.path.join(out_dir,
                                                      'policy_agent.pdf'))
    visualizer.visualize_viper_policies(out_dir, depths=[3])
    visualizer.visualize_moe_policies(out_dir, hard_prediction=True,
                                      experts=[4], depths=[3])


def visualize_all():
    out_dir = '/home/unk/Downloads/nips'

    rl_policy = load_rl_policy('mountaincar')
    MountainCarVisualization().visualize_policy(rl_policy,
                                                out_file=os.path.join(out_dir,
                                                                      'policy_agent.pdf'))
    MountainCarVisualization().visualize_qvalues(
        os.path.join(out_dir, 'qimportance.pdf'))

    visualize_viper_policies('/home/unk/Downloads/nips/viper_policies')
    visualize_moe_policies('/home/unk/Downloads/nips/moe_policies',
                           hard_prediction=False)
    visualize_moe_policies('/home/unk/Downloads/nips/moehard_policies',
                           hard_prediction=True)

    visualize_viper_mispredictions(
        os.path.join(out_dir, 'viper_mispredictions'))
    visualize_moe_mispredictions(os.path.join(out_dir, 'moe_mispredictions'),
                                 hard_prediction=False)
    visualize_moe_mispredictions(
        os.path.join(out_dir, 'moehard_mispredictions'),
        hard_prediction=True)

    visualize_gates_soft('/home/unk/Downloads/nips/gates_soft')
    visualize_gates_hard('/home/unk/Downloads/nips/gates_hard')


def visualize_some():
    visualizer = MountainCarVisualization()

    moe_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/mountaincar/1/MOE',
        'moe_policy_e8_d2.pk')
    moe_hard_policy = moe_policy.clone()
    moe_hard_policy.hard_prediction = True

    moe_analytical_policy = MOEAnalyticalPolicy(moe_hard_policy)
    rl_policy = load_rl_policy('mountaincar')
    viper_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/mountaincar/1/ViperPlus/',
        'dt_policy_d2.pk')

    visualizer.visualize_gate(moe_policy=moe_policy,
                              hard_prediction=True,
                              out_file='/home/unk/Downloads/gates.pdf')

    visualizer.visualize_policy(rl_policy,
                                out_file='/home/unk/Downloads/policy_agent.pdf')
    visualizer.visualize_policy(moe_policy,
                                out_file='/home/unk/Downloads/policy_moe.pdf')
    visualizer.visualize_policy(moe_hard_policy,
                                out_file='/home/unk/Downloads/policy_moe_hard.pdf')
    visualizer.visualize_policy(moe_analytical_policy,
                                out_file='/home/unk/Downloads/policy_moe_analytical.pdf')
    visualizer.visualize_policy(viper_policy,
                                out_file='/home/unk/Downloads/policy_viper.pdf')

    visualizer.visualize_mispredictions(rl_policy, moe_policy,
                                        out_file='/home/unk/Downloads/mispredictions_moe.pdf')
    visualizer.visualize_mispredictions(rl_policy, viper_policy,
                                        out_file='/home/unk/Downloads/mispredictions_viper.pdf')


def mountaincar():
    moe_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/cartpole/best/MOEHard',
        'moe_policy_e2_d1.pk')
    moe_policy.predict_hard = True

    analytica_policy = MOEAnalyticalPolicy(moe_policy)

    cp_policy = CartpoleE2D1Policy(moe_policy)
    cp_policy.predict(np.array([[0., 0., 0., 0.]]))

    tests = (np.random.rand(1000, 4) * 2.) - 1.
    equal = np.array_equal(moe_policy.predict(tests),
                           cp_policy.predict(tests))

    print('Array equal: {}'.format(equal))
    assert (np.array_equal(moe_policy.predict(tests),
                           cp_policy.predict(tests)))

    # visualize_selected()
    # visualize_all()
    # visualize_some()

class Cartpole_moeth_extracted_policy:
    def predict(self, observations):
        # 11.0226837694925 * cp + 36.4527772940471 * cv + 104.177279627728 * pa + 127.855102723844 * pv + 5.04559027172444 > 0
        predictions = []
        for i in range(observations.shape[0]):
            observation = observations[i]
            cp, cv, pa, pv = observation
            # if 11.0226837694925 * cp + 36.4527772940471 * cv + 104.177279627728 * pa + 127.855102723844 * pv + 5.04559027172444 > 0:
            if 2.184617295 * cp + 7.22468043 * cv + 20.647193691 * pa + 25.339969327 * pv > -1.:
            # if 2.18 * cp + 7.22 * cv + 20.64 * pa + 25.33 * pv > -1.:
                decision = 1
            else:
                decision = 0
            predictions.append(decision)
        return predictions


class Cartpole_moeth_approximated_policy:
    def predict(self, observations):
        predictions = []
        for i in range(observations.shape[0]):
            observation = observations[i]
            cp, cv, pa, pv = observation
            if 7.22468043 * cv + 25.339969327 * pv > -1.:
                decision = 1
            else:
                decision = 0
            predictions.append(decision)
        return predictions


def cartpole():
    # NOTE: with 1500 interpolation_steps DT policy is presented
    # more correctly (there are edge cases
    visualizer = CartpoleVisualization(interpolation_steps=200)
    visualizer3D = CartpoleVisualization(interpolation_steps=15)

    rl_policy = load_rl_policy('cartpole')

    # from viper.cartpole.cartpole import create_gym_env
    # env = create_gym_env()
    # obs = env.reset()
    # obs = np.array([0., 2., 0., -0.15])
    # env.env.state = obs
    # while True:
    #     env.render()
    #     action = rl_policy.get_action(obs)
    #     obs, rew, done, _ = env.step(action)
    #     if done:
    #         break

    policy = Cartpole_moeth_extracted_policy()
    visualizer.visualize_policy(
        policy,
        out_file='/home/unk/Downloads/visualization/extracted.pdf')

    visualizer.visualize_policy(
        rl_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_drl_policy.pdf')
    visualizer3D.visualize_policy_3D(
        rl_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_drl_policy3D.pdf')

    viper_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/filtered_v3_only_pareto_results/cartpole/0/ViperPlus',
        'dt_policy_d6.pk')
    visualizer.visualize_policy(
        viper_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_viper_policy.pdf')
    visualizer3D.visualize_policy_3D(
        viper_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_viper_policy3D.pdf')
    save_dt_policy_viz(viper_policy,
                       '/home/unk/Downloads/visualization',
                       'dt_policy_d6.dot',
                       ['cp', 'cv', 'pa', 'pv'],
                       ['left', 'right'])

    moe_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/filtered_v3_only_pareto_results/cartpole/0/MOE',
        'moe_policy_e2_d0.pk')
    visualizer.visualize_policy(
        moe_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_moet_policy.pdf')
    visualizer3D.visualize_policy_3D(
        moe_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_moet_policy3D.pdf')

    moe_hard_policy = load_policy(
        '/home/unk/projects/explainableRL/viper/data/filtered_v3_only_pareto_results/cartpole/0/MOEHard',
        'moe_policy_e2_d0.pk')
    moe_hard_policy.hard_prediction = True
    visualizer.visualize_policy(
        moe_hard_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_moeth_policy.pdf')
    visualizer3D.visualize_policy_3D(
        moe_hard_policy,
        out_file='/home/unk/Downloads/visualization/cartpole_moeth_policy3D.pdf')


def main():
    # mountaincar()
    cartpole()


if __name__ == '__main__':
    main()
