import os
from .rl import get_rollouts
from .hybrid import HybridAgent
import re


def tuple_to_int(tuple):
    x1, x2, x3, x4, x5, x6, x7 = tuple
    return int(x1), int(x2), int(x3), int(x4), int(x5), int(x6), int(x7)


def add_dig_trace(state, action, invocation_number, out_dir):
    # Type of vars is numpy.float64
    x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(state)

    dig_trace_file = os.path.join(out_dir, 'dig' + str(action) + '.tcs')
    with open(dig_trace_file, "a") as f:
        f.write(
            str(x1) + ' ' + str(x2) + ' ' + str(x3) + ' ' + str(x4) + ' ' + str(
                x5)
            + ' ' + str(x6) + ' ' + str(x7) + '\n')


def generate_dig_trace(env, teacher, state_transformer, n_batch_rollouts,
                       out_dir):
    obss, acts, qs = [], [], []

    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))

    for action_id in range(6):
        dig_file = os.path.join(out_dir, 'dig' + str(action_id) + '.tcs')
        with open(dig_file, "w") as f:
            f.write('x1 x2 x3 x4 x5 x6 x7\n')

    for i in range(len(obss)):
        add_dig_trace(obss[i], acts[i], i, out_dir)


class HybridAgentDig(HybridAgent):

    def __init__(self, rl_agent, state_transformer):
        super(HybridAgentDig, self).__init__(rl_agent)
        self.state_transformer = state_transformer
        self.action_lambdas = self.load_lambdas()

    def load_lambdas(self):
        """Load lambda functions representing invariants for different actions."""
        result = list()
        for action_id in range(6):
            dig_file = '../data/dig_output/dig' + str(action_id) + '.inv.txt'
            content = open(dig_file).read()
            lambdas = eval(content)
            result.append(lambdas)
        return result

    def evaluate_lambdas(self, lambdas, symbolic_obs):
        x1, x2, x3, x4, x5, x6, x7 = tuple_to_int(symbolic_obs)
        arg_name_to_val = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                           'x6': x6, 'x7': x7}

        for i in xrange(len(lambdas)):
            # Get lambda function (in string format).
            l = lambdas[i]
            # Extract argument names of the lambda function.
            arg_names = re.sub(r'lambda (.*):.*$', r'\1', l).split(',')
            # Eval lambda string to get actual lambda function.
            f = eval(l)
            # Construct tuple with arguments.
            arg_tuple = ()
            for arg_name in arg_names:
                arg_tuple = arg_tuple + (arg_name_to_val[arg_name],)
            if not f(*arg_tuple):
                # If lambda is not sattisfied then invariant does not hold for input example.
                return False
        return True

    def modify_action(self, obs, action):
        self.num_predictions += 1

        symbolic_obs = self.state_transformer(obs)

        plausible_actions = list()
        for action_id in range(6):
            if self.evaluate_lambdas(self.action_lambdas[action_id],
                                     symbolic_obs):
                plausible_actions.append(action_id)

        if len(plausible_actions) == 1:
            return plausible_actions[0]

        self.num_predictions_rl_agent += 1
        return action
