# This file provides functionality for creating evaluation results database.

from viper.util.util import delete_if_exists
from viper.util.util import read_all_lines
from viper.util.latex import unpack_latex_macro
from viper.evaluation.constants import *
from viper.util.serialization import load_policy
import glob
import re
import sqlite3
import os
import numpy as np
import warnings


def get_db_path():
    return os.path.join(RESULTS_DIR, 'results.db')


def get_result_files(subject):
    """
    Discovers files containing evaluation results for a given subject.

    Args:
        subject (str): Name of the subject.

    Returns:
        list(str): Result file paths.
    """
    subject_result_dir = os.path.join(RESULTS_DIR, subject)
    patterns = ['journal/*',
                'sweeps_2021_3/train_gating_1x_per_epoch/*',
                'sweeps_2021_3_broad / train_gating_1x_per_epoch/*',
                'sweeps_2021_4/*']
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(subject_result_dir, pattern)))
    result = []
    for path in paths:
        if os.path.isdir(path):
            result_files = glob.glob(path + '/*/*{}'.format(EVALUATION_FILE_SUFFIX))
            result.extend(result_files)
    return result


def extract_configs_used(results):
    """
    Extract all different configurations present in results.

    Args:
        results (list(str)): A list of results (latex DefMacros).

    Returns:
        list(str): A list of configuration strings.
    """
    configs = list()
    for line in results:
        key, _ = unpack_latex_macro(line)
        if not key:
            continue
        m = re.match(
            '(?P<subject>[^_]+)_(?P<model>[^_]+)_(?P<config>.*)_reward',
            key)
        if m:
            configs.append(m.group('config'))
    return configs


def filter_results(results, config):
    """

    Args:
        results (list(str)): A list of results (latex DefMacros).
        config (str): Configuration.

    Returns:
        list(str): Input list of results filtered by a given configuration.
    """
    filtered = list()
    for line in results:
        key, _ = unpack_latex_macro(line)
        if not key:
            continue
        m = re.match(
            '[^_]+_[^_]+_' + config + '.*',
            key)
        if m:
            filtered.append(line)
    return filtered


def get_value(results, config, type):
    """
    Extract a value from results.

    Args:
        results (list(str)): A list of results (latex DefMacros).
        config (str): Configuration.
        type (str): Type of value we are searching for.
            Informs about name of a macro we are searching for.

    Returns:
        str: Value of searched macro.
    """
    for result in results:
        key, value = unpack_latex_macro(result)
        if not key:
            continue
        m = re.match(
            '[^_]+_[^_]+_{}_{}'.format(config, type),
            key)
        if m:
            return value
    return None


def extract_all_configurations(db):
    """
    Extracts all configurations present in database.

    Args:
        db (Database): Database.

    Returns:
        list(tuple(str, str, str)): Returns tuples (subject, model, config).

    """
    triples = list()
    for subject in SUBJECTS:
        for model in ['MOE', 'MOEHard', 'ViperPlus']:
            # Note that ORDER will not work always as expected, e.g., D12, D8.
            query = '''
              SELECT DISTINCT config FROM results 
                WHERE subject='{}' and model='{}' 
                  ORDER BY config ASC;
            '''.format(subject, model)
            c = db.execute(query)
            for row in c.fetchall():
                config = row[0]
                triples.append((subject, model, config))
    return triples


def pick_best_result(results):
    """
    Finds best results in a set of provided results.

    Args:
        results (list(DBResults)): List of results.
        reward_name (str): Name of the field in DBResults named tuple.
        mispredictions_name (str): Name of the field in DBResults named tuple.

    Returns:
        DBResults: Entry with best results.

    """
    results_sorted = sorted(
        results,
        key=lambda element: (-element.reward,
                             element.mispredictions))
    return results_sorted[0]


ResultsTable = collections.namedtuple('ResultsTable',
                                      [
                                          'id',
                                          'file_path',
                                          'subject',
                                          'model',
                                          'config',
                                          'reward',
                                          'mispredictions',
                                          'nodes',
                                          'choose_best_student_strategy',
                                          'moe_max_epoch',
                                          'moe_init_learning_rate',
                                          'moe_learning_rate_decay',
                                          'moe_log_frequency',
                                          'moe_stop_count',
                                          'moe_regularization_mode',
                                          'moe_use_new_formula',
                                          'moe_use_adam_optimizer',
                                          'unix_timestamp',
                                          'max_iters'
                                      ])

SummaryTable = collections.namedtuple('SummaryTable',
                                      [
                                          'id',
                                          'subject',
                                          'model',
                                          'config',
                                          'reward_best',
                                          'reward_avg',
                                          'reward_std',
                                          'mispredictions_best',
                                          'mispredictions_avg',
                                          'mispredictions_std',
                                          'nodes',
                                          'num_runs'
                                      ])


class Database(object):

    CREATE_RESULTS_TABLE_SQL = '''
    CREATE TABLE results (id INTEGER PRIMARY KEY,
                          file_path TEXT NOT NULL,
                          subject TEXT NOT NULL,
                          model TEXT NOT NULL,
                          config TEXT NOT NULL,
                          reward REAL NOT NULL,
                          mispredictions REAL NOT NULL,
                          nodes INTEGER NOT NULL,
                          choose_best_student_strategy TEXT,
                          moe_max_epoch INTEGER,
                          moe_init_learning_rate INTEGER,
                          moe_learning_rate_decay REAL,
                          moe_log_frequency INTEGER,
                          moe_stop_count INTEGER,
                          moe_regularization_mode INTEGER,
                          moe_use_new_formula INTEGER,
                          moe_use_adam_optimizer INTEGER,
                          unix_timestamp TIMESTAMP,
                          max_iters INTEGER) 
    '''

    CREATE_SUMMARY_TABLE_SQL = '''
    CREATE TABLE summary (id INTEGER PRIMARY KEY,
                          subject TEXT NOT NULL,
                          model TEXT NOT NULL,
                          config TEXT NOT NULL,
                          reward_best REAL NOT NULL,
                          reward_avg REAL NOT NULL,
                          reward_std REAL NOT NULL,
                          mispredictions_best REAL NOT NULL,
                          mispredictions_avg REAL NOT NULL,
                          mispredictions_std REAL NOT NULL,
                          nodes INTEGER NOT NULL,
                          num_runs INTEGER NOT NULL)
    '''

    INSERT_RESULTS_SQL = '''INSERT INTO results(
                     file_path, subject, model, config, reward, mispredictions, nodes,
                     choose_best_student_strategy, moe_max_epoch,
                     moe_init_learning_rate, moe_learning_rate_decay,
                     moe_log_frequency, moe_stop_count, moe_regularization_mode,
                     moe_use_new_formula, moe_use_adam_optimizer, 
                     unix_timestamp, max_iters) 
                     VALUES ('{}', '{}', '{}','{}','{}', '{}', '{}', '{}', '{}',
                             '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
    '''

    INSERT_SUMMARY_SQL = '''INSERT INTO summary(
        subject, model, config, reward_best, reward_avg, reward_std,
        mispredictions_best, mispredictions_avg, mispredictions_std, 
        nodes, num_runs)
        VALUES ('{}', '{}', '{}', '{}', '{}', 
                '{}', '{}', '{}', '{}', '{}', '{}')'''

    def __init__(self, path):
        self.path = path
        self.conn = None

    def execute(self, command):
        c = self.conn.cursor()
        c.execute(command)
        return c

    def create_schema(self):
        self.conn = sqlite3.connect(self.path)
        self.execute(self.CREATE_RESULTS_TABLE_SQL)
        self.execute(self.CREATE_SUMMARY_TABLE_SQL)

    def close(self):
        self.conn.commit()
        self.conn.close()

    def open(self):
        self.conn = sqlite3.connect(self.path)

    def add_to_results_table(self, file_path, subject,
                             model, config, data, nodes):
        reward = get_value(data, config, 'reward')
        mispredictions = get_value(data, config,
                                   'mispredictions_student_playing')
        choose_best_student_strategy = get_value(data, config,
                                                 'choose_best_student_strategy')
        moe_max_epoch = get_value(data, config, 'moe_max_epoch')
        moe_init_learning_rate = get_value(data, config,
                                           'moe_init_learning_rate')
        moe_learning_rate_decay = get_value(data, config,
                                            'moe_learning_rate_decay')
        moe_log_frequency = get_value(data, config,
                                      'moe_log_frequency')
        moe_stop_count = get_value(data, config,
                                   'moe_stop_count')
        moe_regularization_mode = get_value(data, config,
                                            'moe_regularization_mode')
        moe_use_new_formula = (
            1 if get_value(data, config, 'moe_use_new_formula') == 'True'
            else 0)
        moe_use_adam_optimizer = (
            1 if get_value(data, config, 'moe_use_adam_optimizer') == 'True'
            else 0)
        unix_timestamp = get_value(data, config,
                                   'unix_timestamp')
        max_iters = get_value(data, config,
                              'max_iters')

        c = self.conn.cursor()
        c.execute(self.INSERT_RESULTS_SQL.format(
            file_path, subject, model, config, reward, mispredictions, nodes,
            choose_best_student_strategy, moe_max_epoch, moe_init_learning_rate,
            moe_learning_rate_decay, moe_log_frequency, moe_stop_count,
            moe_regularization_mode, moe_use_new_formula, moe_use_adam_optimizer,
            unix_timestamp, max_iters))

    def add_to_summary_table(self, subject, model, config,
                             reward_best, reward_avg, reward_std,
                             mispredictions_best, mispredictions_avg,
                             mispredictions_std, nodes, num_runs):
        c = self.conn.cursor()
        c.execute(self.INSERT_SUMMARY_SQL.format(
            subject, model, config, reward_best, reward_avg, reward_std,
            mispredictions_best, mispredictions_avg, mispredictions_std,
            nodes, num_runs))

    def select_from_summary_table(self, subject, model, config):
        columns = ', '.join(SummaryTable._fields)
        query = '''SELECT {}
                     FROM summary
                     WHERE subject='{}' and model='{}' and config='{}';
                     '''.format(columns, subject, model, config)

        result = self.execute(query).fetchall()
        if result:
            assert len(result) == 1
            return SummaryTable(*result[0])
        else:
            return None

    def get_node_count(self, dirname, model, config):
        if model == 'ViperPlus':
            fname = 'dt_policy_{}.pk'.format(config)
        else:
            fname = 'moe_policy_{}.pk'.format(config)
        try:
            policy = load_policy(dirname, fname)
            return policy.get_node_count()
        except IOError:
            # TODO: Handle these cases differently
            print('no file: ' + dirname)
            return -1
        except ImportError as e:
            print('WARNING: Pickle file not in the right format. ' +
                  'dir: {}, file: {}'
                  .format(dirname, fname))
            print(e)
            return -1

    def process_result_file(self, subject, path):
        """
        Process result file.
        :param path: Path to a result file.
        """
        lines = read_all_lines(path)

        basename = os.path.basename(path)
        m = re.match('(?P<model>.*){}'.format(EVALUATION_FILE_SUFFIX),
                     basename)
        model = m.group('model')

        configs = extract_configs_used(lines)
        for config in configs:
            filtered = filter_results(lines, config)
            node_count = self.get_node_count(os.path.dirname(path),
                                             model,
                                             config)
            self.add_to_results_table(path, subject, model, config, filtered,
                                      node_count)

    def create_results_table(self):
        # for each subject
        # get result directories
        for subject in SUBJECTS:
            paths = get_result_files(subject)
            for path in paths:
                self.process_result_file(subject, path)

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        result = []
        for i in range(0, len(lst), n):
            result.append(lst[i:i + n])
        return result

    def add_config_to_summary(self, subject, model, config):
        columns = ', '.join(ResultsTable._fields)
        var = globals()['{}ViperPlusParams'.format(subject)]
        max_iters = var.max_iters
        where_clause = "subject='{}' and model='{}' and config='{}' and max_iters='{}'".format(
            subject, model, config, max_iters
        )
        if 'MOE' in model:
            where_clause = "{} and moe_use_adam_optimizer='{}'".format(
                where_clause, SHOW_ADAM_OPTIMIZER_RESULTS[subject]
            )
            # Takes last k results.
            query = ("""SELECT {} FROM results 
                        WHERE {}
                        ORDER BY unix_timestamp DESC
                        LIMIT {}""".format(
                columns, where_clause, RUNS_FOR_SUMMARY[subject]))

            results = []
            for row in self.execute(query).fetchall():
                results.append(ResultsTable(*row))

            if len(results) != RUNS_FOR_SUMMARY[subject]:
                warnings.warn('Not enough experiments for {}: {}_{}'.format(
                    subject, model, config
                ))

            if len(results) == 0:
                return

            best_result = pick_best_result(results)

            rewards = np.array([x.reward for x in results])
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            mispredictions = np.array([x.mispredictions for x in results])
            avg_mispredictions = np.mean(mispredictions)
            std_mispredictions = np.std(mispredictions)

            self.add_to_summary_table(subject, model, config,
                                      best_result.reward, avg_reward, std_reward,
                                      best_result.mispredictions,
                                      avg_mispredictions,
                                      std_mispredictions, best_result.nodes,
                                      num_runs=len(results))
        else:
            # Splits Viper results into groups.
            # e.g., let's say there's 30 runs of Viper split across 3 groups.
            # Each group contains 10 results.
            # Average of each group is computed, and the group that has the
            # best average is selected as a result.
            # TODO: A more correct thing to do might be to add different
            # chunks of viper as different rows in summary table and then
            # do choosing based on the summary table.

            # ViperPlus
            copies = VIPER_RUN_COPIES[subject][config]
            runs = RUNS_FOR_SUMMARY[subject]

            query = ("""SELECT {} FROM results 
                                    WHERE {}
                                    ORDER BY unix_timestamp DESC
                                    LIMIT {}""".format(
                columns, where_clause, copies * runs))

            all_results = []
            for row in self.execute(query).fetchall():
                all_results.append(ResultsTable(*row))

            if len(all_results) == 0:
                return

            if len(all_results) != copies * runs:
                warnings.warn('Not enough experiments for {}_{}_{}'.format(
                    subject, model, config
                ))

            result_chunks = self.chunk_list(all_results, runs)
            chunk_perf = []

            for chunk_id in range(len(result_chunks)):
                chunk_results = result_chunks[chunk_id]
                chunk_rewards = np.array([x.reward for x in chunk_results])
                chunk_mispredictions = np.array([x.mispredictions
                                                 for x in chunk_results])
                chunk_perf.append((np.mean(chunk_rewards),
                                   np.mean(chunk_mispredictions),
                                   chunk_id))

            chunk_results_sorted = sorted(
                chunk_perf,
                key=lambda element: (-element[0], element[1]))
            best_chunk_id = chunk_results_sorted[0][2]

            results = result_chunks[best_chunk_id]
            best_result = pick_best_result(results)

            rewards = np.array([x.reward for x in results])
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            mispredictions = np.array([x.mispredictions for x in results])
            avg_mispredictions = np.mean(mispredictions)
            std_mispredictions = np.std(mispredictions)

            self.add_to_summary_table(subject, model, config,
                                      best_result.reward, avg_reward,
                                      std_reward,
                                      best_result.mispredictions,
                                      avg_mispredictions,
                                      std_mispredictions, best_result.nodes,
                                      num_runs=len(results))

    def create_summary_table(self):
        triples = extract_all_configurations(self)

        for triple in triples:
            subject, model, config = triple
            self.add_config_to_summary(subject, model, config)


def create_database():
    db_path = get_db_path()
    delete_if_exists(db_path)

    db = Database(db_path)
    db.create_schema()
    db.create_results_table()
    db.create_summary_table()

    return db


def main():
    db = create_database()
    db.close()


if __name__ == '__main__':
    main()
