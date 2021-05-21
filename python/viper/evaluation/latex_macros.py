from viper.evaluation.database import extract_all_configurations
from viper.evaluation.database import get_db_path
from viper.evaluation.database import Database
from viper.evaluation.constants import *
from viper.util.latex import create_latex_macro
from viper.util.util import delete_if_exists
from viper.evaluation.util import get_config_names_for_depth
from viper.evaluation.util import get_experts_from_config_name
from viper.evaluation.util import get_depth_from_config_name
import math
import os

CHOOSE_BEST_CRITERIA = 'best'
CHOOSE_AVG_CRITERIA = 'avg'

# Macro for maximum achieved reward for a given depth.
MACRO_MAX_REWARD_DEPTH = 'MaxRewardDepth_Pick-{}'
# Macro for maximum achieved reward for a given effective depth.
MACRO_MAX_REWARD_EFFECTIVE_DEPTH = 'MaxRewardEffectiveDepth_Pick-{}'
MACRO_AVG_ACROSS_CONFIGS = 'AvgRewardAcrossConfigs{}'

PAPER_DATA_DIR = '../../paper/icml2020/data'
MACROS_FILE = os.path.join(PAPER_DATA_DIR, 'database.tex')


def create_macros(db):
    delete_if_exists(MACROS_FILE)

    # For each configuration (subject, model, parameters) create performance
    # macros.
    create_macros_all_configurations(db)

    for criteria in [CHOOSE_BEST_CRITERIA, CHOOSE_AVG_CRITERIA]:
        # For each configuration (subject, model, depth) create perf. macros.
        create_macros_depths(db, criteria)

        # For each configuration (subject, model, effective depth) create
        # perf. macros.
        create_macros_effective_depth(db, criteria)

    # Average results across different configuration with same effective depth.
    create_macros_average_across_effective_depth(db)

    # For each configuration (subject, model) create perf. macros.
    create_macros_models(db)


def format_value(value):
    return '{0:.2f}'.format(value)


def write_macros(results, subject, model, config, prefix=''):
    best_reward = results.reward_best
    avg_reward = results.reward_avg
    std_reward = results.reward_std
    best_mispredictions = results.mispredictions_best
    avg_mispredictions = results.mispredictions_avg
    std_mispredictions = results.mispredictions_std
    nodes = results.nodes

    # TODO: Rename reward to reward_best.
    with open(MACROS_FILE, 'a') as f:
        # Best reward across different models trained with the same configuration.
        # Reward is still averaged across multiple trials.
        f.write(create_latex_macro('{}_{}_{}_{}_reward_best'
                                   .format(prefix, subject, model, config),
                                   format_value(best_reward)))
        f.write('\n')
        f.write(create_latex_macro('{}_{}_{}_{}_mispredictions_student_playing_best'
                                   .format(prefix, subject, model, config),
                                   format_value(best_mispredictions)))
        f.write('\n')
        f.write(create_latex_macro('{}_{}_{}_{}_reward_avg'
                                   .format(prefix, subject, model, config),
                                   format_value(avg_reward)))
        f.write('\n')
        f.write(create_latex_macro('{}_{}_{}_{}_reward_std'
                                   .format(prefix, subject, model, config),
                                   format_value(std_reward)))
        f.write('\n')
        f.write(
            create_latex_macro('{}_{}_{}_{}_mispredictions_student_playing_avg'
                               .format(prefix, subject, model, config),
                               format_value(avg_mispredictions)))
        f.write('\n')
        f.write(
            create_latex_macro('{}_{}_{}_{}_mispredictions_student_playing_std'
                               .format(prefix, subject, model, config),
                               format_value(std_mispredictions)))
        f.write('\n')
        f.write(
            create_latex_macro('{}_{}_{}_{}_nodes'
                               .format(prefix, subject, model, config),
                               str(nodes)))
        f.write('\n')


def create_macros_all_configurations(db):
    triples = extract_all_configurations(db)

    for triple in triples:
        subject, model, config = triple
        results = db.select_from_summary_table(subject, model, config)
        if results:
            write_macros(results, subject, model, config)


def find_best_for_depth(depth, subject, model, db, criteria):
    all_results = []
    for config_name in get_config_names_for_depth(subject=subject,
                                                  model=model,
                                                  depth=depth):
        results = db.select_from_summary_table(subject,
                                               model,
                                               config_name)
        if not results:
            continue
        all_results.append(results)

    if len(all_results) > 0:
        return pick_from_summaries(all_results, criteria)
    else:
        return None


def create_macros_depths(db, criteria):
    for subject in SUBJECTS:
        for depth in DEPTHS[subject]:
            config = 'd{}'.format(depth)
            for model in ['ViperPlus', 'MOE', 'MOEHard']:
                results = find_best_for_depth(depth,
                                              subject,
                                              model,
                                              db,
                                              criteria)
                if results:
                    prefix = MACRO_MAX_REWARD_DEPTH.format(criteria)
                    write_macros(results, subject, model, config, prefix)

                    with open(MACROS_FILE, 'a') as f:
                        key_name = '{}_{}_{}_{}_experts'.format(prefix,
                                                               subject,
                                                               model,
                                                               config)
                        experts = get_experts_from_config_name(results.config)
                        f.write(create_latex_macro(key_name, str(experts)))
                        f.write('\n')


def pick_from_summaries(results, criteria):
    """
    Picks the result from summaries.

    Args:
        results list(SummaryTable): A list of results from SummaryTable.
        criteria str: Whether to pick the best ('best')
          or average result ('avg').

    Returns:
        SummaryTable: Best result out of input entries.

    """
    if criteria == CHOOSE_BEST_CRITERIA:
        results_sorted = sorted(results, key=lambda element: (
            -element.reward_best, element.mispredictions_best))
    elif criteria == CHOOSE_AVG_CRITERIA:
        results_sorted = sorted(results, key=lambda element: (
            -element.reward_avg, element.mispredictions_avg))
    else:
        raise Exception('Unrecognized criteria.')

    return results_sorted[0]


def get_moe_results_for_effective_depth(effective_depth, subject, model, db):
    all_results = list()
    for experts in EXPERTS[subject]:
        for depth in EXPERT_DEPTHS[subject]:
            real_depth = int(math.ceil(math.log(experts, 2))) + depth
            if real_depth == effective_depth:
                config = 'e{}_d{}'.format(experts, depth)
                results = db.select_from_summary_table(subject,
                                                       model,
                                                       config)
                if not results:
                    continue
                # results += (experts, depth)
                all_results.append(results)
    return all_results


def find_best_moe_for_effective_depth(effective_depth, subject, model, db,
                                      criteria):
    """
    Finds the best moe config for given effective depth.
    Experts and depth of the winning candidate are appended to the results.
    :param effective_depth:
    :param subject:
    :param model:
    :param db:
    :return:
    """
    all_results = get_moe_results_for_effective_depth(effective_depth,
                                                      subject,
                                                      model,
                                                      db)
    if all_results:
        return pick_from_summaries(all_results, criteria)
    else:
        return None


def create_macros_effective_depth(db, criteria):
    for subject in SUBJECTS:
        for effective_depth in DEPTHS[subject]:
            config = 'd{}'.format(effective_depth)

            for model in ['MOE', 'MOEHard']:
                results_moe = find_best_moe_for_effective_depth(effective_depth,
                                                                subject,
                                                                model,
                                                                db,
                                                                criteria)
                if results_moe:
                    prefix = MACRO_MAX_REWARD_EFFECTIVE_DEPTH.format(criteria)
                    write_macros(results_moe, subject, model, config, prefix)

                    with open(MACROS_FILE, 'a') as f:
                        key = '{}_{}_{}_{}_best-config'.format(prefix,
                                                              subject,
                                                              model,
                                                              config)
                        experts_used = get_experts_from_config_name(
                            results_moe.config)
                        depth_used = get_depth_from_config_name(
                            results_moe.config)
                        f.write(create_latex_macro(key, 'E{}:D{}'.format(
                            experts_used, depth_used
                        )))
                        f.write('\n')


def create_macros_average_across_effective_depth(db):
    for subject in SUBJECTS:
        for effective_depth in DEPTHS[subject]:
            config = 'd{}'.format(effective_depth)

            for model in ['MOE', 'MOEHard']:
                all_results = get_moe_results_for_effective_depth(
                    effective_depth, subject, model, db)
                if all_results is not None and len(all_results) > 0:
                    rewards = [result.reward_avg for result in all_results]
                    mispredictions = [result.mispredictions_avg for result in all_results]
                    rewards_avg = sum(rewards) / len(rewards)
                    mispredictions_avg = sum(mispredictions) / len(mispredictions)

                    prefix = MACRO_AVG_ACROSS_CONFIGS.format('avg')
                    reward_macro_name = '{}_{}_{}_{}_reward_avg'.format(prefix,
                                                                        subject,
                                                                        model,
                                                                        config)
                    mispredictions_macro_name = '{}_{}_{}_{}_mispredictions_student_playing_avg'.format(
                        prefix, subject, model, config)

                    with open(MACROS_FILE, 'a') as f:
                        f.write(create_latex_macro(reward_macro_name,
                                                   format_value(rewards_avg)))
                        f.write('\n')
                        f.write(create_latex_macro(mispredictions_macro_name,
                                                   format_value(mispredictions_avg)))
                        f.write('\n')


def create_macros_models(db):
    # raise NotImplementedError
    pass


def main():
    db_path = get_db_path()
    db = Database(db_path)
    db.open()
    create_macros(db)


if __name__ == '__main__':
    main()
