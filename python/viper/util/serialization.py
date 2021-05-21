import os
import pickle as pk
import sys


def save_policy(policy, dirname, fname, protocol=2):
    """Saves a policy as a pickle file."""
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    path = os.path.join(dirname, fname)
    f = open(path, 'wb')
    pk.dump(policy, f, protocol=protocol)
    f.close()


def load_policy(dirname, fname):
    """Loads a policy saved as a pickle file."""
    path = os.path.join(dirname, fname)

    f = open(path, 'rb')
    if sys.version_info[0] < 3:
        policy = pk.load(f)
    else:
        policy = pk.load(f, encoding='latin1')
    f.close()
    return policy


if __name__ == '__main__':
    policy = load_policy('/home/UNK/Downloads/test', 'moe_policy_e16_d21.pk')
    for protocol in range(0, 5):
        save_policy(policy,
                    '/home/UNK/Downloads/test',
                    'v{}.pk'.format(protocol),
                    protocol)
