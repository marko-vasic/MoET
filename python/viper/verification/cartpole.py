from viper.util.log import *
from viper.util.serialization import load_policy

import numpy as np
from z3 import *
import time
import copy

A = np.array([[0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, -7.1707, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 7.8878, 0.0]])

B = np.array([0.0, 1.5743, 0.0, -0.7317])

force = 10.0
dt = 0.02

t_max = 10


def f(z, a):
    u = -force if a == 0 else force
    y = z + (np.matmul(A, z) + u * B) * dt
    return y


def f_symb(z, a):
    u = If(a == 0, -force, force)
    y = copy.copy(z)
    for i in range(4):
        for j in range(4):
            y[i] += A[i, j] * z[j] * dt
        y[i] += u * B[i] * dt
    return y


def get_vars(t):
    return [Real('x' + str(t)), Real('v' + str(t)), Real('t' + str(t)),
            Real('w' + str(t))]


def get_init(z):
    return And(z[0] >= -0.05, z[0] <= 0.05,
               z[1] >= -0.05, z[1] <= 0.05,
               z[2] >= -0.05, z[2] <= 0.05,
               z[3] >= -0.05, z[3] <= 0.05)


def get_safety(z):
    return And(z[2] >= -0.1, z[2] <= 0.1)


def solve_policy(policy):
    # Step 0: Solver
    s = Solver()

    # Step 1: Get initial variables
    z = get_vars(0)

    # Step 2: Build initial constraints
    s.add(get_init(z))

    # Step 3: Run dynamics
    safety = []
    for t in range(t_max):
        # dynamics
        z = f_symb(z, policy(z))

        # safety
        safety.append(get_safety(z))

        # logging
        zt = get_vars(t + 1)
        s.add([z[i] == zt[i] for i in range(4)])

    # Step 4: Safety constraint
    s.append(Not(And(safety)))

    # Step 5: Run solver
    r = s.check()
    log('Result: {}'.format(r), INFO)
    if r == sat:
        for t in range(t_max + 1):
            values = [_convert(s.model()[var]) for var in get_vars(t)]
            action = s.model().evaluate(policy(get_vars(t)))
            log('State {}: {}; action {}'.format(t, values, action), INFO)


def _convert(val):
    if val.__class__ == RatNumRef:
        return float(val.numerator_as_long()) / float(val.denominator_as_long())
    elif val.__class__ == IntNumRef:
        return val.as_long()
    else:
        raise Exception()


def make_policy_func(policy, z):
    return _make_policy_func_helper(policy, z, 0)


def _make_policy_func_helper(policy, z, nid):
    # Step 0: Base case (leaf node)
    if policy.tree_.children_left[nid] == policy.tree_.children_right[nid]:
        return int(np.argmax(policy.tree_.value[nid]))

    # Step 1: Feature
    s = z[policy.tree_.feature[nid]]

    # Step 2: Threshold
    t = policy.tree_.threshold[nid]

    # Step 3: Recursive calls
    v_true = _make_policy_func_helper(policy, z,
                                      policy.tree_.children_left[nid])
    v_false = _make_policy_func_helper(policy, z,
                                       policy.tree_.children_right[nid])

    # Step 4: Construct if statement
    return If(s <= t, v_true, v_false)


def make_policy_func_moe(moe_policy, z):
    experts = moe_policy.dtc_list
    E = len(experts)
    F = moe_policy.tetag.shape[0] / E
    gating = moe_policy.tetag.reshape(E, F).T

    z_copy = copy.copy(z)
    # Normalize values.
    z_copy -= moe_policy.scaler.mean_
    z_copy /= moe_policy.scaler.scale_

    # Append bias term.
    z_copy = np.append(z_copy, [1])

    return _make_moe_policy_func_helper(experts, gating, z_copy)


def _make_moe_policy_func_helper(experts, gating, z):
    """
    Args:
        experts: A list of decision tree experts.
        gating: Gating parameters; dimension is feature_num x experts_num
        z: Feature values.
    """
    experts_num = len(experts)
    features_num = len(z)

    option_values = [0] * experts_num
    for i in range(experts_num):
        for j in range(features_num):
            option_values[i] += z[j] * gating[j][i]

    guard_conditions = []
    for i in range(len(option_values)):
        max_conditions = []
        for j in range(len(option_values)):
            if i == j:
                continue
            max_conditions.append(option_values[i] >= option_values[j])
        guard_conditions.append(And(max_conditions))

    if_condition = make_policy_func(experts[experts_num - 1], z)
    for j in range(experts_num - 2, -1, -1):
        # TODO: Maybe this can be speeded up.
        if_condition = If(guard_conditions[j],
                          make_policy_func(experts[j], z),
                          if_condition)

    return if_condition


def solve_viper(depth):
    dirname = '../data/cartpole/best/ViperPlus'
    fname = 'dt_policy_d{}.pk'.format(depth)

    policy = load_policy(dirname, fname).tree
    policy_func = lambda z: make_policy_func(policy, z)

    start = time.time()
    solve_policy(policy_func)
    end = time.time()
    log('Time to verify {}: {:.2f}s'.format(fname, end - start), INFO)


def solve_viper_all():
    for depth in [1, 2, 3, 4, 5]:
        log('########', INFO)
        solve_viper(depth)


def solve_moe(experts, depth):
    dirname = '../data/cartpole/best/MOEHard'
    fname = 'moe_policy_e{}_d{}.pk'.format(experts, depth)

    log('Verifying policy {}'.format(fname), INFO)

    moe_policy = load_policy(dirname, fname)
    policy_func = lambda z: make_policy_func_moe(moe_policy.moe, z)

    start = time.time()
    solve_policy(policy_func)
    end = time.time()
    duration = end - start
    log('Time to verify {}: {:.2f}s'.format(fname, duration), INFO)
    return duration


def solve_moe_all(experts, depths):
    verification_times = list()
    for depth in depths:
        verification_times_for_depth = list()
        for experts_num in experts:
            log('#########', INFO)
            time = solve_moe(experts_num, depth)
            verification_times_for_depth.append(time)
        verification_times.append(verification_times_for_depth)
    return verification_times


# +---------+---------+---------+--------+--------+--------+
# | Experts |   D1    |   D2    |   D3   |   D4   |   D5   |
# +---------+---------+---------+--------+--------+--------+
# |       2 | 5.99s   | 9.74s   | 2.64s  | 6.66s  | 8.04s  |
# |       4 | 124.77s | 19.77s  | 15.64s | 41.55s | 13.72s |
# |       8 | 319.42s | 171.28s | 98.54s | 52.37s | 60.60s |
# +---------+---------+---------+--------+--------+--------+
def plot():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({'font.size': 14})

    x = [2, 4, 8]
    ys = [[5.99, 124.77, 319.42],
          [9.74, 19.77, 171.28],
          [2.64, 15.64, 98.54],
          [6.66, 41.55, 52.37],
          [8.04, 13.72, 60.60]]

    for depth in [1, 2, 3, 4, 5]:
        # plotting the line 1 points
        plt.plot(x, ys[depth - 1], label="d{}".format(depth))

    # naming the x axis
    plt.xlabel('#experts')
    # naming the y axis
    plt.ylabel('time [s]')
    # giving a title to my graph
    # plt.title('Two lines on same graph!')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.savefig('/home/UNK/Downloads/vvvv.pdf')
    plt.show()


def plot_new(times, experts, depths):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({'font.size': 14})

    for depth in depths:
        # plotting the line 1 points
        plt.plot(experts, times[depth - 1], label="d{}".format(depth))

    # naming the x axis
    plt.xlabel('#experts')
    # naming the y axis
    plt.ylabel('time [s]')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.savefig('/home/UNK/Downloads/vvvv.pdf')
    plt.show()


if __name__ == '__main__':
    # solve_viper_all()

    # solve_viper(1)
    # solve_moe(2, 2)

    depths = [1, 2, 3, 4, 5]
    experts = [2, 3, 4, 5, 6, 7, 8]

    times = solve_moe_all(experts, depths)
    log('verification times: {}'.format(times), INFO)
    plot_new(times, experts, depths)

    # plot()
