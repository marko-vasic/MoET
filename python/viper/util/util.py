import glob
import os
import errno


def truncate_file(file_name):
    with open(file_name, 'w') as f:
        pass


def ensure_parent_exists(file_path):
    """Ensure that directory exists."""
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    directory = os.path.dirname(file_path)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def ensure_dir_exists(directory):
    """Ensure that directory exists."""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def delete_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def seconds_to_hms(seconds):
    """Convert seconds to hours, minutes and seconds.
    Args:
      seconds: Number representing number of seconds.
    Returns:
      A tuple (h, m, s) of time divided into hours, minutes and
      seconds.
    """
    hours = seconds // (60 * 60)
    seconds %= (60 * 60)
    minutes = seconds // 60
    seconds %= 60
    return (hours, minutes, seconds)


def seconds_to_hms_string(seconds):
    h, m, s = seconds_to_hms(seconds)
    return '%dh:%02dm:%02ds' % (h, m, s)


def read_all_lines(path):
    """
    Reads all lines in a given file.
    :param path: Path to the file.
    :return: A list containing file lines.
    """
    result = list()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            result.append(line.rstrip())
    return result


def get_subdirectories(directory):
    """
    Returns all subdirectories of a directory.
    :param directory: Path to the directory.
    :return: List of directory paths.
    """
    paths = [directory]
    for rootdir, dirs, files in os.walk(directory):
        for subdir in dirs:
            paths.append(os.path.join(rootdir, subdir))
    return paths


def get_files(directory, extension):
    """
    Returns all files under directory with specified extension.
    :param directory: String representing path to the directory.
    :param extension: String representing file extension.
    :return: List of file paths.
    """
    # Find all directories under given directory
    # Run glob.glob on each subdirectory.
    paths = []
    subdirectories = get_subdirectories(directory)
    for dir in subdirectories:
        paths.extend(
            glob.glob(
                os.path.join(dir, '*.{}'.format(extension))
            )
        )
    return paths
