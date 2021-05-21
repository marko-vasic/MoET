import os
from .rl import get_rollouts
from .hybrid import HybridAgent


def tuple_to_int(tuple):
    x1, x2, x3, x4, x5, x6, x7 = tuple
    return int(x1), int(x2), int(x3), int(x4), int(x5), int(x6), int(x7)


def add_daikon_trace(state, action, invocation_number, daikon_files_dir,
                     daikon_output_dir):
    # Type of vars is numpy.float64
    x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(state)

    entry_file = os.path.join(daikon_files_dir, 'entry_point.txt')
    with open(entry_file, 'r') as f:
        entry_text = f.read()
    entry_text = entry_text.format(invocation_nonce=str(invocation_number),
                                   x1=str(x1),
                                   x2=str(x2),
                                   x3=str(x3),
                                   x4=str(x4),
                                   x5=str(x5),
                                   x6=str(x6),
                                   x7=str(x7))

    # TODO: Change
    exit_file = os.path.join(daikon_files_dir, 'exit.txt')

    with open(exit_file, 'r') as f:
        exit_text = f.read()
    exit_text = exit_text.format(invocation_nonce=str(invocation_number),
                                 x1=str(x1),
                                 x2=str(x2),
                                 x3=str(x3),
                                 x4=str(x4),
                                 x5=str(x5),
                                 x6=str(x6),
                                 x7=str(x7),
                                 exit_number=str(action))

    daikon_dtrace_file = os.path.join(daikon_output_dir, 'pong.dtrace.txt')
    with open(daikon_dtrace_file, "a") as f:
        f.write('\n')
        f.write(entry_text)
        f.write('\n')
        f.write(exit_text)


def generate_daikon_trace(env, teacher, state_transformer, n_batch_rollouts,
                          daikon_files_dir, daikon_output_dir):
    obss, acts, qs = [], [], []

    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))

    daikon_dtrace_file = os.path.join(daikon_output_dir, 'pong.dtrace.txt')
    # Truncate or create dtrace file.
    open(daikon_dtrace_file, 'w').close()

    for i in range(len(obss)):
        add_daikon_trace(obss[i], acts[i], i, daikon_files_dir,
                         daikon_output_dir)


class HybridAgentDaikon(HybridAgent):
    """
    Created using Daikon invariants from data/daikon_output/pong_1.dtrace.txt trace file.
    """

    def __init__(self, rl_agent, state_transformer):
        super(HybridAgentDaikon, self).__init__(rl_agent)
        self.state_transformer = state_transformer

    def exit_0_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (x1 != 0 and x1 <= 71 and
                x3 <= 79 and x3 >= -49 and
                x4 >= -45 and
                x5 <= 12 and x5 >= -12 and
                x6 <= 12 and x6 >= -11 and
                x7 <= 14 and
                x1 != x2)

    def exit_1_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (
            x1 <= 73 and x3 <= 43 and x3 >= -46 and x4 <= 41 and x4 >= -14 and x5 <= 12 and x5 >= -12
            and x6 <= 10 and x6 >= -12 and x7 <= 15 and x7 >= -15 and x1 != x2 and x1 != x6)

    def exit_2_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (
            x2 <= 73 and x4 <= 39 and x4 >= -56 and x5 <= 12 and x7 <= 16 and x7 >= -15 and x1 != x5)

    def exit_3_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (
            x1 != 0 and x1 <= 46 and x1 >= -82 and x2 <= 71 and x3 <= 7 and x3 >= -6 and x4 <= 10 and x4 >= -8
            and x5 <= 12 and x5 >= -12 and x6 <= 12 and x6 >= -6 and x7 <= 11 and x7 >= -9 and x1 != x2 and x1 != x3
            and x1 != x4 and x1 != x5 and x1 != x7 and x2 >= x3 and x2 != x5 and x2 >= x6 and x2 != x7 and x3 != x5
            and x4 != x7)

    def exit_4_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (
            x1 <= 37 and x1 >= -83 and x3 <= 44 and x4 <= 41 and x4 >= -42 and x5 >= -12 and x6 <= 12 and x6 >= -12
            and x7 <= 14 and x7 >= -19 and x1 != x2)

    def exit_5_cond(self, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)

        return (
            x1 <= 68 and x2 <= 73 and x3 <= 79 and x6 <= 14 and x6 >= -9 and x7 >= -15 and x1 != x2)

    def modify_action(self, obs, action):
        self.num_predictions += 1

        symbolic_obs = self.state_transformer(obs)
        exits_satisfied = list()
        if self.exit_0_cond(symbolic_obs):
            exits_satisfied.append(0)
        if self.exit_1_cond(symbolic_obs):
            exits_satisfied.append(1)
        if self.exit_2_cond(symbolic_obs):
            exits_satisfied.append(2)
        if self.exit_3_cond(symbolic_obs):
            exits_satisfied.append(3)
        if self.exit_4_cond(symbolic_obs):
            exits_satisfied.append(4)
        if self.exit_5_cond(symbolic_obs):
            exits_satisfied.append(5)
        if len(exits_satisfied) == 1:
            # If only one exit satisfies condition choose that action
            return exits_satisfied[0]
        self.num_predictions_rl_agent += 1
        return action
