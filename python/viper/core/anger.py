from .hybrid import *
from .rl import get_rollouts

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from scipy.special import gammainc
import pandas as pd
import numpy as np
from pyclustering.cluster.xmeans import xmeans


class LocalDT:
    def __init__(self, dt, center, radius):
        self.center = center
        self.radius = radius
        self.dt = dt

    def within_responsibility(self, point):
        return np.linalg.norm(self.center - point) < self.radius

    def predict(self, point):
        return self.dt.predict(point)


class DTEnsemble:
    def __init__(self, global_dt, local_dts):
        self.global_dt = global_dt
        self.local_dts = local_dts

    def predict(self, points):
        predictions = list()
        for point in points:
            found = False
            for local_dt in self.local_dts:
                if local_dt.within_responsibility(point):
                    predictions.append(local_dt.predict([point])[0])
            if not found:
                predictions.append(self.global_dt.predict([point])[0])
        return np.asarray(predictions)


class DTEnsembleHybrid(HybridAgent):
    """Hybrid Agent based on DT and LORE rules."""

    def __init__(self, global_dt, local_dts):
        super(DTEnsembleHybrid, self).__init__(global_dt)
        self.local_dts = local_dts

    def modify_action(self, obs, action):
        self.num_predictions += 1

        for local_dt in self.local_dts:
            if local_dt.within_responsibility(obs):
                return local_dt.predict([obs])[0]

        self.num_predictions_rl_agent += 1
        return action


class ClusterStrategy(object):
    def cluster(self, data):
        pass


class KMeansClusterStrategy(ClusterStrategy):

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def cluster(self, data):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(data)

        max_distances = np.zeros(kmeans.n_clusters, dtype=float)
        for data_idx in range(0, len(data)):
            # Compute radii of clusters.
            cluster_idx = kmeans.labels_[data_idx]
            datapoint = data[data_idx]
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            distance = np.linalg.norm(datapoint - cluster_center)
            if distance > max_distances[cluster_idx]:
                max_distances[cluster_idx] = distance
        return kmeans.cluster_centers_, max_distances


class XMeansClusterStrategy(ClusterStrategy):

    def __init__(self, max_clusters):
        self.max_clusters = max_clusters

    def cluster(self, data):
        xmeans_instance = xmeans(data, kmax=self.max_clusters)
        xmeans_instance.process()

        cluster_centers = xmeans_instance.get_centers()
        clusters = xmeans_instance.get_clusters()

        max_distances = list()
        for cluster_idx in xrange(0, len(clusters)):
            cluster_indices = clusters[cluster_idx]
            cluster_center = cluster_centers[cluster_idx]

            max_distance = 0.
            for item_idx in xrange(0, len(cluster_indices)):
                datapoint = data[cluster_indices[item_idx]]
                distance = np.linalg.norm(datapoint - cluster_center)
                if distance > max_distance:
                    max_distance = distance

            max_distances.append(max_distance)

        return cluster_centers, np.asarray(max_distances)


class SampleStrategy(object):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, center, radius):
        pass


class NormalSampleStrategy(SampleStrategy):
    """Sampling using normal distribution."""

    def __init__(self, num_samples):
        super(NormalSampleStrategy, self).__init__(num_samples)

    def sample(self, center, radius):
        """Sample using normal distribution around the cluster center."""

        # I choose that 2 sigma is equal to radius of cluster.
        sigma = radius / 2.
        return np.random.normal(center, sigma,
                                size=[self.num_samples, len(center)])


class UniformSampleStrategy(SampleStrategy):
    """Sampling using uniform distribution."""

    def __init__(self, num_samples):
        super(UniformSampleStrategy, self).__init__(num_samples)

    def sample(self, center, radius):
        """Sample using uniform distribution within the sphere
           (center, radius)."""

        # link: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        r = radius
        ndim = len(center)
        x = np.random.normal(size=(self.num_samples, ndim))
        ssq = np.sum(x ** 2, axis=1)
        fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
        frtiled = np.tile(fr.reshape(self.num_samples, 1), (1, ndim))
        p = center + np.multiply(x, frtiled)
        return p


class Anger(object):

    def __init__(self,
                 env,
                 rl_policy,
                 dt_policy,
                 n_rollouts,
                 cluster_strategy,
                 sample_strategy,
                 state_transformer,
                 regression=False,
                 action_comparison=None):
        self.env = env
        self.rl_policy = rl_policy
        self.dt_policy = dt_policy
        self.n_rollouts = n_rollouts
        self.cluster_strategy = cluster_strategy
        self.sample_strategy = sample_strategy
        self.state_transformer = state_transformer
        self.regression = regression
        self.action_comparison = action_comparison

    def create_misprediction_data(self):
        """
        Output is an array of observations for which mispredictions occur.
        """
        rollouts = get_rollouts(self.env, self.rl_policy, render=False,
                                n_batch_rollouts=self.n_rollouts)
        mispredictions = list()
        for i in xrange(0, len(rollouts)):
            obss = rollouts[i][0]
            teacher_act = rollouts[i][1]
            transformed_obss = self.state_transformer(obss)
            student_act = self.dt_policy.predict([transformed_obss])[0]
            if self.action_comparison:
                if not self.action_comparison(teacher_act, student_act):
                    mispredictions.append(transformed_obss)
            else:
                if teacher_act != student_act:
                    mispredictions.append(transformed_obss)
        return np.asarray(mispredictions)

    def anger(self, local_dts_depth):
        """Creates a mixture of local and global DTs"""

        # Algorithm:
        # 1. Create Mispredictions Data
        # 2. Cluster Mispredictions
        # 3. Sample points and build local DT (for each cluster)

        # Create Mispredictions Data
        data = self.create_misprediction_data()

        # Cluster Mispredictions
        cluster_centers, cluster_radii = self.cluster_strategy.cluster(data)
        # print('Number of clusters: ' + str(len(cluster_centers)))

        local_dts = list()
        for cluster_idx in range(len(cluster_centers)):
            cluster_center = cluster_centers[cluster_idx]
            cluster_radius = cluster_radii[cluster_idx]

            # Sample
            # TODO: No way to map back from symbolic space back to image (atari games).
            points = self.sample_strategy.sample(cluster_center, cluster_radius)

            teacher_acts = self.rl_policy.predict(points)
            if not self.regression:
                local_tree = DecisionTreeClassifier(max_depth=local_dts_depth)
            else:
                local_tree = DecisionTreeRegressor(max_depth=local_dts_depth)
            local_tree.fit(points, teacher_acts)

            # print('Cluster ' + str(cluster_idx))
            # print('Local DT acc: ' + str(np.mean(teacher_acts == local_tree.predict(points))))
            # print('Global DT acc: ' + str(np.mean(teacher_acts == dt_policy.predict(points))))
            local_dt = LocalDT(local_tree, cluster_center,
                               cluster_radii[cluster_idx])

            local_dts.append(local_dt)

        return local_dts
