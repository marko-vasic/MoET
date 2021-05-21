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

import numpy as np
import time
from viper.util.log import *
from viper.util.util import *
from viper.core.compare_policy import ComparePolicy


def get_rollout(env, policy, render):
    if env.__class__.__name__ == 'TorcsEnv':
        # Special handling for Torcs.
        # Set relaunch to True if you notice memory leak issues.
        # obs, done = np.array(env.reset(relaunch=True)), False
        obs, done = np.array(env.reset(relaunch=False)), False
        last_low_reward = 0
    else:
        obs, done = np.array(env.reset()), False
    rollout = []

    while not done:
        # Render
        if render:
            env.unwrapped.render()

        if (np.array(obs).ndim == 0):
            # This is for the case of 1-dimensional
            # observation space, like in frozenLake.
            obs = [obs]
            
        # Action
        act = policy.predict(np.array([obs]))[0]

        # Step
        next_obs, rew, done, info = env.step(act)

        # Rollout (s, a, r)
        rollout.append((obs, act, rew))

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)

        if env.__class__.__name__ == 'TorcsEnv':
            if rew < 0.5:
                last_low_reward += 1
            else:
                last_low_reward = 0
            if last_low_reward > 50:
                # If car is stuck, end training.
                break

    return rollout


def get_rollouts(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, render))
    return rollouts


def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)),
                               p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)),
                               replace=False)

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]


class TransformerPolicy:
    def __init__(self, policy, state_transformer):
        self.policy = policy
        self.state_transformer = state_transformer

    def predict(self, obss):
        return self.policy.predict(
            np.array([self.state_transformer(obs) for obs in obss]))


def test_policy(env, policy, n_test_rollouts):
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        trace = get_rollout(env, policy, False)
        cum_rew += sum((rew for _, _, rew in trace))
    return cum_rew / n_test_rollouts


def identify_best_policy_reward_only(env,
                                     policies,
                                     state_transformer,
                                     n_test_rollouts):
    log('Initial policy count: {}'.format(len(policies)), INFO)
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1) / 2)
        log('Current policy count: {}'.format(n_policies), INFO)

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env,
                                  TransformerPolicy(policy, state_transformer),
                                  n_test_rollouts)
            new_policies.append((policy, new_rew))
            log('Reward update: {} -> {}'.format(rew, new_rew), INFO)

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]


def identify_best_policy_reward_mispredictions(env,
                                               policies,
                                               state_transformer,
                                               n_test_rollouts,
                                               observations,
                                               teacher_actions):
    """Identifies best policy based on reward and mispredictions.

    Args:
      env: RL environment.
      policies: List of policies.
      state_transformer: Transformer of observation space.
      n_test_rollouts: Number of rollouts to perform for evaluation.
      observations: Set of observations (in transformed space) collected during training.
      teacher_actions: Set of actions that teacher performs coresponding to observations.
    """
    log('Identifying the best policy using mispredictions and reward', INFO)

    results = list()

    for policy_idx in range(0, len(policies)):
        log('Evaluating Policy {0}/{1}'.format(policy_idx, len(policies)), INFO)
        policy = policies[policy_idx][0]

        reward = test_policy(env,
                             TransformerPolicy(policy, state_transformer),
                             n_test_rollouts)

        student_actions = policy.predict(observations)
        precision = np.mean(teacher_actions == student_actions)
        mispredictions = 1 - precision
        results.append((policy, reward, mispredictions))
        log('Policy {0}: reward={1}; mispredictions={2}'.
            format(policy_idx, reward, mispredictions), INFO)

    sorted_results = sorted(results, key=lambda x: (-x[1], x[2]))
    best = sorted_results[0]
    best_policy_index = results.index(best)
    log('Choosing Policy {0} as the best.'.format(best_policy_index), INFO)
    return sorted_results[0][0]


def identify_best_policy_mispredictions_nodes(env,
                                              policies,
                                              state_transformer,
                                              n_test_rollouts,
                                              observations,
                                              teacher_actions):
    """Identifies best policy based on reward and mispredictions and node count.

    Args:
      env: RL environment.
      policies: List of policies.
      state_transformer: Transformer of observation space.
      n_test_rollouts: Number of rollouts to perform for evaluation.
      observations: Set of observations (in transformed space) collected during training.
      teacher_actions: Set of actions that teacher performs coresponding to observations.
    """
    log('Identifying the best policy using mispredictions and reward', INFO)

    results = list()

    for policy_idx in range(0, len(policies)):
        log('Evaluating Policy {0}/{1}'.format(policy_idx, len(policies)), INFO)
        policy = policies[policy_idx][0]

        reward = test_policy(env,
                             TransformerPolicy(policy, state_transformer),
                             n_test_rollouts)

        student_actions = policy.predict(observations)
        precision = np.mean(teacher_actions == student_actions)
        mispredictions = 1 - precision
        node_count = policy.get_node_count()
        policy_depth = policy.get_depth()
        results.append((policy,
                        reward,
                        mispredictions,
                        policy_depth,
                        node_count))
        log('Policy {0}: reward={1}; mispredictions={2}; node_count={3}; depth={4};'.
            format(policy_idx, reward, mispredictions,
                   node_count, policy_depth), INFO)

    # Sort first based on mispredictions, than based on depth,
    # and than based on node count.
    sorted_results = sorted(results, key=lambda x: (x[2], x[3], x[4]))
    best = sorted_results[0]
    best_policy_index = results.index(best)
    log('Choosing Policy {} as the best. With mispredictions: {}, depth: {} and nodes: {}'
        .format(best_policy_index, best[2], best[3], best[4]), INFO)
    return sorted_results[0][0]


def identify_best_policy_reward_mispredictions_harmonic(env,
                                                        policies,
                                                        teacher,
                                                        state_transformer,
                                                        n_test_rollouts,
                                                        min_episode_reward,
                                                        max_episode_reward):
    """Identifies best policy based on reward and mispredictions.

    Args:
      env: RL environment.
      policies: List of policies.
      state_transformer: Transformer of observation space.
      n_test_rollouts: Number of rollouts to perform for evaluation.
    """
    log('Identifying the best policy using mispredictions and reward', INFO)

    results = list()

    for policy_idx in range(0, len(policies)):
        log('Evaluating Policy {0}/{1}'.format(policy_idx, len(policies)), INFO)
        policy = policies[policy_idx][0]

        wrapped_policy = TransformerPolicy(policy, state_transformer)
        cmp_policy = ComparePolicy(wrapped_policy, teacher)
        reward = test_policy(env, cmp_policy, n_test_rollouts)

        if reward < min_episode_reward or reward > max_episode_reward:
            print('ERROR: reward {} is out of bounds.'.format(reward))

        reward_scaled = ((reward - min_episode_reward) /
                         (max_episode_reward - min_episode_reward))
        mispredictions = cmp_policy.mispredictions_ratio()
        precision = 1 - mispredictions
        # Harmonic mean of precision and reward.
        score = 2 * reward_scaled * precision / (reward_scaled + precision)

        results.append((policy, score))
        log('Policy {}: reward={:.2f}; mispredictions={:.2f}; score={:.2f}'.
            format(policy_idx, reward, mispredictions, score), INFO)

    sorted_results = sorted(results, key=lambda x: (-x[1]))
    best = sorted_results[0]
    best_policy_index = results.index(best)
    log('Choosing Policy {0} as the best.'.format(best_policy_index), INFO)
    return sorted_results[0][0]


def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i + seq_len])
    return seqs


def get_action_sequences(env, policy, seq_len, n_rollouts):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, False)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])

    return seqs_sorted


def train_dagger(env,
                 teacher,
                 student,
                 state_transformer,
                 max_iters,
                 n_batch_rollouts,
                 max_samples,
                 train_frac,
                 is_reweight,
                 n_test_rollouts,
                 identify_best='reward_only',
                 min_episode_reward=None,
                 max_episode_reward=None):
    # TODO: Check why conversion to np.array is done.

    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    wrapped_student = TransformerPolicy(student, state_transformer)

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))
    qs.extend(teacher.predict_q(np.array([obs for obs, _, _ in trace])))

    start = time.time()
    # Step 2: Dagger outer loop
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(np.array(obss),
                                             np.array(acts),
                                             np.array(qs),
                                             max_samples,
                                             is_reweight)
        log('Training student with {} points'.format(len(cur_obss)), INFO)
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env,
                                     wrapped_student,
                                     False,
                                     n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]

        # Step 2c: Query the oracle for supervision
        # at the interface level, order matters,
        # since teacher.predict may run updates
        teacher_qs = teacher.predict_q(student_obss)
        teacher_acts = teacher.predict(student_obss)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((state_transformer(obs) for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)

        students.append((student.clone(), cur_rew))

    end = time.time()
    duration = end - start
    log('Dagger training student policies in: {0}'.format(
        seconds_to_hms_string(duration)), INFO)

    start = time.time()
    # Step 3: Identify the best policy.
    # TODO: Use strategy pattern for identify best functions.
    if identify_best == 'reward_only':
        max_student = identify_best_policy_reward_only(env,
                                                       students,
                                                       state_transformer,
                                                       n_test_rollouts)
    elif identify_best == 'reward_and_mispredictions':
        max_student = identify_best_policy_reward_mispredictions(env,
                                                                 students,
                                                                 state_transformer,
                                                                 n_test_rollouts,
                                                                 np.array(obss),
                                                                 np.array(acts))
    elif identify_best == 'mispredictions_and_nodes':
        max_student = identify_best_policy_mispredictions_nodes(
            env, students, state_transformer, n_test_rollouts,
            np.array(obss), np.array(acts))
    elif identify_best == 'reward_and_mispredictions_harmonic':
        max_student = identify_best_policy_reward_mispredictions_harmonic(
            env=env,
            policies=students,
            teacher=teacher,
            state_transformer=state_transformer,
            n_test_rollouts=n_test_rollouts,
            min_episode_reward=min_episode_reward,
            max_episode_reward=max_episode_reward)
    else:
        raise ValueError(
            'Incorrect value {0} for identify_best argument'.format(
                identify_best))

    end = time.time()
    duration = end - start
    log('Dagger identifying the best policy in: {0}'.format(
        seconds_to_hms_string(duration)), INFO)

    return max_student


def _sample_uniform(obss, acts, max_pts):
    idx = np.random.choice(len(obss), size=min(max_pts, len(obss)),
                           replace=False)
    return obss[idx], acts[idx]


def train_dagger_noq(env,
                     teacher,
                     student,
                     state_transformer,
                     max_iters,
                     n_batch_rollouts,
                     max_samples,
                     train_frac,
                     n_test_rollouts):
    """
    This is a version of Dagger algorithm when q values are not
    available, and thus we perform uniform sampling instead of
    sampling based on q-values.
    """
    # Step 0: Setup
    obss, acts = [], []
    students = []
    wrapped_student = TransformerPolicy(student, state_transformer)

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts = _sample_uniform(np.array(obss), np.array(acts),
                                             max_samples)
        log('Training student with {} points'.format(len(cur_obss)), INFO)
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, wrapped_student, False,
                                     n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]

        # Step 2c: Query the oracle for supervision
        teacher_acts = teacher.predict(student_obss)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((state_transformer(obs) for obs in student_obss))
        acts.extend(teacher_acts)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)

        students.append((student.clone(), cur_rew))

    max_student = identify_best_policy_reward_only(env, students, state_transformer,
                                                   n_test_rollouts)

    return max_student
