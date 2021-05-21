"""
Converts all files in a given directory to a new serialization protocol.
"""
import os
from viper.util.serialization import load_policy, save_policy
from viper.util.util import get_files


def convert(path, protocol):
    """
    Converts file containing serialized object
    to the new serialization protocol.
    :param path: Path to the file.
    :param protocol: Serialization protocol to be used.
    """
    dirname, fname = os.path.split(path)
    policy = load_policy(dirname, fname)
    save_policy(policy, dirname, fname, protocol)


def main():
    files = get_files(
        '/home/UNK/projects/explainableRL/viper/data/experiments',
        'pk')
    converted = 0
    for file in files:
        try:
            convert(file, protocol=2)
            converted += 1
        except Exception:
            print('File cannot be converted: {}'.format(file))
    print('Converted {} out of {} files'.format(converted, len(files)))


if __name__ == '__main__':
    main()
