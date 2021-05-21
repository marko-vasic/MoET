from viper.evaluation.database import ResultsTable
from viper.evaluation.database import get_db_path
from viper.evaluation.database import Database
from viper.evaluation.constants import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import pandas as pd
import warnings
import os
import re

from viper.evaluation.util import identify_pareto
from viper.util.util import ensure_parent_exists
from shutil import copyfile


DEPTHS_FOR_VIPER_TREND_PLOT = {
    'cartpole': range(1, 16),
    'pong': range(1, 31),
    'acrobot': range(1, 21),
    'mountaincar': range(1, 21),
    'lunarlander': range(1, 26)
}


def get_from_result_table(db, columns, where_clause, num_results,
                          print_warnings=True):
    # TODO: change
    # I put a large number that ensures getting all results for a given config.
    num_results = 500
    print_warnings = False

    query = ("""SELECT {}   FROM results 
                            WHERE {}
                            ORDER BY unix_timestamp DESC
                            LIMIT {}""".format(
        columns, where_clause, num_results))

    results = []
    for row in db.execute(query).fetchall():
        results.append(ResultsTable(*row))

    if print_warnings:
        if len(results) != num_results:
            warnings.warn('Not enough experiments (missing {}) for {}.'.format(
                num_results - len(results), where_clause
            ))

    return results


def extract_results_for_config(db, subject, model, config, num_runs,
                            print_warnings=True):
    columns = ', '.join(ResultsTable._fields)

    viper_max_iters = globals()['{}ViperPlusParams'.format(subject)].max_iters
    moe_max_iters = globals()['{}MOEParams'.format(subject)].max_iters
    moeh_max_iters = globals()['{}MOEHardParams'.format(subject)].max_iters
    assert viper_max_iters == moe_max_iters == moeh_max_iters

    if model == 'ViperPlus':
        viper_where_clause = "subject='{}' and model='{}' and config='{}' and max_iters='{}'".format(
            subject, model, config, viper_max_iters
        )
        results = get_from_result_table(db,
                                        columns,
                                        viper_where_clause,
                                        num_runs,
                                        print_warnings)
    else:
        where_clause = "subject='{}' and model='{}' and config='{}' and max_iters='{}' and moe_use_adam_optimizer='{}'".format(
            subject, model, config, moe_max_iters,
            SHOW_ADAM_OPTIMIZER_RESULTS[subject]
        )
        results = get_from_result_table(
            db, columns, where_clause, num_runs, print_warnings)

    return results


def extract_results_for_ED(db, subject, effective_depth):
    """
    Extract model results for given effective depth.
    """
    results = []

    viper_model = 'ViperPlus'
    viper_config = 'd{}'.format(effective_depth)

    num_repetitions = VIPER_RUN_COPIES[subject][viper_config]
    if num_repetitions == 0:
        num_repetitions = 1
    viper_num_runs = num_repetitions * RUNS_FOR_SUMMARY[subject]

    results.extend(extract_results_for_config(db,
                                           subject,
                                           viper_model,
                                           viper_config,
                                           viper_num_runs))

    for model in [MOE_MODEL, MOEHARD_MODEL]:
        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                real_depth = int(math.ceil(math.log(experts, 2))) + depth
                if real_depth == effective_depth:
                    config = 'e{}_d{}'.format(experts, depth)

                    num_runs = RUNS_FOR_SUMMARY[subject]
                    results.extend(extract_results_for_config(db,
                                                           subject,
                                                           model,
                                                           config,
                                                           num_runs))

    return results


def create_box_plot(data_dict, plot_label, output_file):
    # Code taken from:
    # https://matplotlib.org/3.1.1/gallery/statistics/boxplot_demo.html

    data = []
    models = []
    for label in data_dict.keys():
        data.append(data_dict[label])
        models.append(label)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(
        'Comparison of Viper, MOET and MOETh')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel(plot_label)

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 21
    # bottom = 0
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(models, rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .97, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color=box_colors[k])

    # # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    ensure_parent_exists(output_file)
    plt.savefig(output_file)
    plt.close()


def create_box_plots(db, output_dir):
    for subject in SUBJECTS:
        subject_output_dir = os.path.join(output_dir, subject, 'box_plots')
        for effective_depth in DEPTHS[subject]:
            results = extract_results_for_ED(
                db, subject, effective_depth)

            rewards_dict = collections.OrderedDict()
            mispredictions_dict = collections.OrderedDict()
            for result in results:
                key = '{}-{}'.format(result.model, result.config)
                if not key in rewards_dict:
                    rewards_dict[key] = []
                rewards_dict[key].append(result.reward)
                if not key in mispredictions_dict:
                    mispredictions_dict[key] = []
                mispredictions_dict[key].append(result.mispredictions)
            for key in rewards_dict.keys():
                rewards_dict[key] = np.array(rewards_dict[key])
                mispredictions_dict[key] = np.array(mispredictions_dict[key])

            file_name = '{}_depth{}_{}.png'.format(subject,
                                                   effective_depth,
                                                   'reward')
            create_box_plot(rewards_dict,
                            'reward',
                            os.path.join(subject_output_dir, file_name))

            file_name = '{}_depth{}_{}.png'.format(subject,
                                                   effective_depth,
                                                   'mispredictions')
            create_box_plot(mispredictions_dict,
                            'mispredictions',
                            os.path.join(subject_output_dir,
                                         file_name))


def plot_Pareto_fronts(scores,
                       labels,
                       plot_only_pareto_points,
                       output_file,
                       plot_params,
                       show_legend=True):
    dict_marker = plot_params['dict_marker']
    dict_color = plot_params['dict_color']

    scores = np.array(scores)
    pareto_ids = identify_pareto(scores)
    pareto_scores = [scores[i] for i in pareto_ids]
    pareto_labels = [labels[i] for i in pareto_ids]

    pareto_df = pd.DataFrame(pareto_scores, columns=['reward', 'fidelity'])
    pareto_df['labels'] = pareto_labels
    # Sort pareto front based on the first coordinate (to aid plotting).
    pareto_df.sort_values('reward', inplace=True)

    if show_legend:
        fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]})
    else:
        fig, (ax) = plt.subplots(ncols=1)

    for label, pareto_group_df in pareto_df.groupby('labels'):
        legend_label = label if plot_only_pareto_points else None

        ax.scatter(pareto_group_df['reward'].values,
                   pareto_group_df['fidelity'].values,
                   label=legend_label,
                   c=dict_color[label],
                   marker=dict_marker[label],
                   linewidths=plot_params['pareto_linewidths'],
                   s=plot_params['pareto_marker_size'])

    # Plots the Pareto front line.
    ax.plot(pareto_df['reward'].values,
            pareto_df['fidelity'].values,
            linestyle=':',
            color='black')

    if not plot_only_pareto_points:
        df = pd.DataFrame(scores.tolist(), columns=['reward', 'fidelity'])
        df['labels'] = labels

        for label, group_df in df.groupby('labels'):
            ax.scatter(group_df['reward'].values,
                       group_df['fidelity'].values,
                       label=label,
                       c=dict_color[label],
                       marker=dict_marker[label],
                       linewidths=plot_params['linewidths'],
                       s=plot_params['marker_size'])

    # ax1.set_xlim(lower_bound, upper_bound)
    ax.set_xlabel('reward', fontweight="bold")
    ax.set_ylabel('fidelity', fontweight="bold")
    ax.grid()

    # The logic for placing plot outside of the figure is taken from:
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    if show_legend:
        h, l = ax.get_legend_handles_labels()
        lax.legend(h, l, borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()

    # plt.setp([a.get_xticklabels() for a in [ax1]], visible=False)
    ensure_parent_exists(output_file)
    plt.savefig(output_file)
    plt.close()


def filter_top_k_best(all_scores, all_labels, k):
    """
    Given a list of scores for Viper, MOE and MOEHard
    extract top k best results for each configuration and merge it into return
    list.
    """

    scores = {
        'viper': [],
        'moe': [],
        'moehard': []
    }

    labels = {
        'viper': [],
        'moe': [],
        'moehard': []
    }

    for i in range(len(all_scores)):
        if 'ViperPlus' in all_labels[i]:
            scores['viper'].append(all_scores[i])
            labels['viper'].append(all_labels[i])
        elif 'MOE' in all_labels[i] and 'MOEHard' not in all_labels[i]:
            scores['moe'].append(all_scores[i])
            labels['moe'].append(all_labels[i])
        else:
            scores['moehard'].append(all_scores[i])
            labels['moehard'].append(all_labels[i])

    if (len(scores['viper']) != len(scores['moe'])
        or len(scores['viper']) != len(scores['moehard'])):
        warnings.warn('Number of total runs not equal')

    filtered_scores = {
        'viper': [],
        'moe': [],
        'moehard': []
    }
    filtered_labels = {
        'viper': [],
        'moe': [],
        'moehard': []
    }

    for model in ['viper', 'moe', 'moehard']:
        reward_sorted_idx = np.argsort(
            np.array(scores[model],
                     dtype=[('reward', np.float64),
                            ('fidelity', np.float64)]),
            order=['reward', 'fidelity'])
        fidelity_sorted_idx = np.argsort(
            np.array(scores[model],
                     dtype=[('reward', np.float64),
                            ('fidelity', np.float64)]),
            order=['fidelity', 'reward'])
        reward_sorted_idx = reward_sorted_idx[::-1]
        fidelity_sorted_idx = fidelity_sorted_idx[::-1]

        i = 0
        j = 0
        visited = set()
        order = True
        while (len(filtered_scores[model]) < k
               and (i < len(scores[model]) and j < len(scores[model]))):
            if order:
                idx = reward_sorted_idx[i]
                if idx not in visited:
                    visited.add(idx)
                    order = False
                    filtered_scores[model].append(scores[model][idx])
                    filtered_labels[model].append(labels[model][idx])
                i += 1
            else:
                idx = fidelity_sorted_idx[j]
                if idx not in visited:
                    visited.add(idx)
                    order = True
                    filtered_scores[model].append(scores[model][idx])
                    filtered_labels[model].append(labels[model][idx])
                j += 1

        if len(filtered_scores[model]) < k:
            warnings.warn('Not enough runs for {}'.format(model))

    merged_scores = []
    merged_labels = []

    for model in ['viper', 'moe', 'moehard']:
        merged_scores.extend(filtered_scores[model])
        merged_labels.extend(filtered_labels[model])

    return merged_scores, merged_labels


def filter_top_k_best_pareto(results, k,
                             filter=None):
    """
    Given a list of scores for Viper, MOE and MOEHard
    extract top k best results for each configuration and merge it into return
    list.
    """
    models = [ViperPlus_MODEL, MOE_MODEL, MOEHARD_MODEL]
    scores = {}
    labels = {}
    results_by_model = {}
    filtered_scores = {}
    filtered_labels = {}
    filtered_results_by_model = {}
    for model in models:
        scores[model] = []
        labels[model] = []
        results_by_model[model] = []
        filtered_scores[model] = []
        filtered_labels[model] = []
        filtered_results_by_model[model] = []

    for result in results:
        scores[result.model].append(
            [result.reward,
             (100. - result.mispredictions) / 100.])
        labels[result.model].append(
            '{}-{}'.format(result.model, result.config)
        )
        results_by_model[result.model].append(result)
    for model in models:
        if filter and not filter(model):
            continue
        while len(scores[model]) > 0:
            finished = False
            pareto_ids = identify_pareto(np.array(scores[model]))
            for idx in pareto_ids:
                filtered_scores[model].append(scores[model][idx])
                filtered_labels[model].append(labels[model][idx])
                filtered_results_by_model[model].append(
                    results_by_model[model][idx])
                if len(filtered_scores[model]) == k:
                    finished = True
                    break
            if finished:
                break
            for idx in sorted(pareto_ids, reverse=True):
                # It's important to iterate list in reverse order.
                del scores[model][idx]

        if len(filtered_scores[model]) < k:
            warnings.warn('Not enough runs for {}'.format(model))

    merged_scores = []
    merged_labels = []

    for model in models:
        merged_scores.extend(filtered_scores[model])
        merged_labels.extend(filtered_labels[model])

    # TODO: Move this elsewhere
    copy = True
    for model in models:
        if len(filtered_results_by_model[model]) == 0:
            # Ensures to copy only when we contain results of all models.
            copy = False
    if copy:
        dest_path = '/home/UNK/projects/explainableRL/viper/data/filtered/{}'.format(
            results[0].subject
        )

        commands_path = os.path.join(dest_path, 'commands.sh')
        ensure_parent_exists(commands_path)
        open(commands_path, 'w').close()

        for model in models:
            for i in range(len(filtered_results_by_model[model])):
                dest_model_dir = os.path.join(dest_path,
                                              str(i),
                                              model)
                result = filtered_results_by_model[model][i]
                model_dir = os.path.dirname(result.file_path)
                model_dir = os.path.join('/home/UNK/projects/explainableRL/viper/data',
                                         model_dir)
                if result.model == ViperPlus_MODEL:
                    m = re.match(
                        'd(?P<depth>.*)',
                        result.config)
                    max_depth = int(m.group('depth'))
                    file_name = 'dt_policy_{}.pk'.format(result.config)

                    model_path = os.path.join(model_dir, file_name)
                    dest_model_path = os.path.join(dest_model_dir, file_name)
                    ensure_parent_exists(dest_model_path)
                    copyfile(model_path, dest_model_path)

                    eval_cmd = (
                    "python -m viper.evaluation.main --subject_name={} --config_type={} --function=evaluate --max_depth={} --out_dir_name={} --choose_best_student_strategy={} --max_iters={}"
                        .format(result.subject,
                                result.model,
                                max_depth,
                                os.path.abspath(os.path.join(dest_model_dir,
                                                             os.pardir)),
                                result.choose_best_student_strategy,
                                result.max_iters
                                ))
                else:
                    m = re.match(
                        'e(?P<experts>.*)_d(?P<depth>.*)',
                        result.config)
                    experts_no = int(m.group('experts'))
                    experts_depths = int(m.group('depth'))
                    file_name = 'moe_policy_{}.pk'.format(result.config)

                    model_path = os.path.join(model_dir, file_name)
                    dest_model_path = os.path.join(dest_model_dir, file_name)
                    ensure_parent_exists(dest_model_path)
                    copyfile(model_path, dest_model_path)

                    eval_cmd = (
                    "python -m viper.evaluation.main --subject_name={} --config_type={} --function=evaluate --experts_no={} --experts_depths={} --out_dir_name={} --use_new_formula={} --choose_best_student_strategy={} --moe_max_epoch={} --moe_init_learning_rate={} --moe_learning_rate_decay={} --max_iters={} --use_adam_optimizer={}"
                        .format(result.subject,
                                result.model,
                                experts_no,
                                experts_depths,
                                os.path.abspath(os.path.join(dest_model_dir,
                                                             os.pardir)),
                                result.moe_use_new_formula,
                                result.choose_best_student_strategy,
                                result.moe_max_epoch,
                                result.moe_init_learning_rate,
                                result.moe_learning_rate_decay,
                                result.max_iters,
                                result.moe_use_adam_optimizer
                                ))
                with open(commands_path, 'a') as file:
                    file.write(eval_cmd + '\n')

    return merged_scores, merged_labels


def create_pareto_front_plots_for_subject(subject,
                                          db,
                                          output_dir,
                                          plot_params,
                                          filter=None):
    all_scores = []
    all_labels = []
    all_results = []
    for effective_depth in DEPTHS[subject]:
        results = extract_results_for_ED(db, subject, effective_depth)
        scores = []
        labels = []
        for result in results:
            if not filter or (filter and filter(result.model)):
                labels.append('{}-{}'.format(result.model, result.config))
                scores.append([result.reward,
                               (100. - result.mispredictions) / 100.])
                all_results.append(result)

        for plot_only_pareto in [True, False]:
            file_name = 'depth{}_pareto-{}.png'.format(
                effective_depth,
                'only' if plot_only_pareto else 'all')
            file_path = os.path.join(output_dir, file_name)
            plot_Pareto_fronts(scores,
                               labels,
                               plot_only_pareto_points=plot_only_pareto,
                               output_file=file_path,
                               plot_params=plot_params)
        all_scores.extend(scores)
        all_labels.extend(labels)

    for plot_only_pareto in [True, False]:
        file_name = 'all-depth_pareto-{}.png'.format(
            'only' if plot_only_pareto else 'all')
        file_path = os.path.join(output_dir, file_name)

        plot_Pareto_fronts(all_scores,
                           all_labels,
                           plot_only_pareto_points=plot_only_pareto,
                           output_file=file_path,
                           plot_params=plot_params,
                           show_legend=plot_only_pareto)

    file_name = 'all-depth_pareto-all_filtered.png'
    file_path = os.path.join(output_dir, file_name)
    filtered_scores, filtered_labels = filter_top_k_best_pareto(
        all_results, k=plot_params['filter_k'], filter=filter)
    plot_Pareto_fronts(filtered_scores,
                       filtered_labels,
                       plot_only_pareto_points=False,
                       output_file=file_path,
                       plot_params=plot_params,
                       show_legend=False)


def initialize_plot_dicts(subject, model):
    # List of available markers:
    # https://matplotlib.org/api/markers_api.html
    plot_params = {}
    dict_marker = {}
    dict_color = {}

    if model == ViperPlus_MODEL:
        plot_params['pareto_marker_size'] = 75
        plot_params['pareto_linewidths'] = 0.75
        plot_params['marker_size'] = 20
        plot_params['linewidths'] = 0.25
        plot_params['filter_k'] = 50

        for ed in DEPTHS[subject]:
            config = (ViperPlus_MODEL + '-d{}').format(ed)
            dict_marker[config] = '${}$'.format(ed)
            dict_color[config] = 'r'
    elif model == MOE_MODEL:
        plot_params['pareto_marker_size'] = 250
        plot_params['pareto_linewidths'] = 1.
        plot_params['marker_size'] = 100
        plot_params['linewidths'] = 0.75
        plot_params['filter_k'] = 50

        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                config = (MOE_MODEL + '-e{}_d{}').format(experts, depth)
                dict_marker[config] = '${}{}$'.format(experts, depth)
                dict_color[config] = 'g'
    elif model == MOEHARD_MODEL:
        plot_params['pareto_marker_size'] = 250
        plot_params['pareto_linewidths'] = 1.
        plot_params['marker_size'] = 100
        plot_params['linewidths'] = 0.75
        plot_params['filter_k'] = 50

        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                config = (MOEHARD_MODEL + '-e{}_d{}').format(experts, depth)
                dict_marker[config] = '${}{}$'.format(experts, depth)
                dict_color[config] = 'b'
    elif model == 'all':
        plot_params['pareto_marker_size'] = 70
        plot_params['pareto_linewidths'] = 1.5
        plot_params['marker_size'] = 10
        plot_params['linewidths'] = 1.5
        plot_params['filter_k'] = 25

        for ed in DEPTHS[subject]:
            config = (ViperPlus_MODEL + '-d{}').format(ed)
            dict_marker[config] = 'o'
            dict_color[config] = 'r'
        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                config = (MOE_MODEL + '-e{}_d{}').format(experts, depth)
                dict_marker[config] = 'h'
                dict_color[config] = 'g'
        for experts in EXPERTS[subject]:
            for depth in EXPERT_DEPTHS[subject]:
                config = (MOEHARD_MODEL + '-e{}_d{}').format(experts, depth)
                dict_marker[config] = '+'
                dict_color[config] = 'b'
    plot_params['dict_marker'] = dict_marker
    plot_params['dict_color'] = dict_color
    return plot_params


def create_pareto_front_plots(db, output_dir):
    for subject in SUBJECTS:
        subject_output_dir = os.path.join(output_dir, subject)
        pareto_output_dir = os.path.join(subject_output_dir, 'pareto')
        plot_params = initialize_plot_dicts(subject, 'all')
        create_pareto_front_plots_for_subject(
            subject,
            db,
            os.path.join(pareto_output_dir, 'all'),
            plot_params
        )
        for model in [ViperPlus_MODEL, MOE_MODEL, MOEHARD_MODEL]:
            filter = lambda c : model in c
            if model == MOE_MODEL:
                filter = lambda c : model in c and not MOEHARD_MODEL in c
            plot_params = initialize_plot_dicts(subject, model)
            create_pareto_front_plots_for_subject(
                subject,
                db,
                os.path.join(pareto_output_dir, model),
                plot_params,
                filter
            )


def create_trend_plot(depths,
                      rewards_mean,
                      rewards_std,
                      fidelity_mean,
                      fidelity_std,
                      file_name,
                      plot_title):
    # Plots inspired by the following posts
    # Plotting std range of line:
    # https://matplotlib.org/3.1.1/gallery/recipes/fill_between_alpha.html
    # Having two scales on y axis left and right:
    # https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots(1)
    rewards_color = 'blue'
    fidelity_color = 'orange'

    ax1.set_xlabel('depth')
    ax1.set_ylabel('reward', color=rewards_color)
    ax1.tick_params(axis='y', labelcolor=rewards_color)
    lns1 = ax1.plot(depths,
                    rewards_mean,
                    lw=2,
                    label='reward',
                    color=rewards_color)
    ax1.fill_between(depths,
                     rewards_mean + rewards_std,
                     rewards_mean - rewards_std,
                     facecolor=rewards_color,
                     alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('fidelity', color=fidelity_color)
    ax2.tick_params(axis='y', labelcolor=fidelity_color)
    lns2 = ax2.plot(depths,
                    fidelity_mean,
                    lw=2,
                    label='fidelity',
                    color=fidelity_color)
    ax2.fill_between(depths,
                     fidelity_mean + fidelity_std,
                     fidelity_mean - fidelity_std,
                     facecolor=fidelity_color,
                     alpha=0.5)
    ax1.set_title(plot_title)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.grid()

    ensure_parent_exists(file_name)
    plt.savefig(file_name)
    plt.close()


def create_Viper_trend_plot(subject, db, output_dir):
    rewards = []
    mispredictions = []
    depths = []
    for depth in DEPTHS_FOR_VIPER_TREND_PLOT[subject]:
        # If we don't care about number of runs being equal,
        # we can set number of runs to high number.
        results = extract_results_for_config(
            db, subject, 'ViperPlus', 'd{}'.format(depth),
            num_runs=RUNS_FOR_SUMMARY[subject], print_warnings=True)
        if len(results) > 0:
            rewards.append(
                np.array([result.reward for result in results]))
            mispredictions.append(
                np.array([result.mispredictions for result in results]))
            depths.append(depth)
    rewards = np.array(rewards)
    rewards_mean = np.array([np.mean(rewards[i]) for i in range(rewards.shape[0])])
    rewards_std = np.array([np.std(rewards[i]) for i in range(rewards.shape[0])])

    fidelity = (100. - np.array(mispredictions)) / 100.
    fidelity_mean = np.array(
        [np.mean(fidelity[i]) for i in range(fidelity.shape[0])])
    fidelity_std = np.array(
        [np.std(fidelity[i]) for i in range(fidelity.shape[0])])

    file_name = os.path.join(output_dir,
                             'viper_reward-fidelity-trend.png'.format(subject))
    plot_title = (
        r'{}: Viper reward and fidelity $\mu$ and $\pm \sigma$ interval'
        .format(subject))
    create_trend_plot(depths=depths,
                      rewards_mean=rewards_mean,
                      rewards_std=rewards_std,
                      fidelity_mean=fidelity_mean,
                      fidelity_std=fidelity_std,
                      file_name=file_name,
                      plot_title=plot_title)


def create_MOET_trend_plot(subject, model, experts, db, output_dir):
    rewards = []
    mispredictions = []
    depths = []
    for depth in EXPERT_DEPTHS[subject]:
        config = 'e{}_d{}'.format(experts, depth)
        num_runs = RUNS_FOR_SUMMARY[subject]
        results = extract_results_for_config(
            db, subject, model, config, num_runs)
        if len(results) > 0:
            rewards.append(
                np.array([result.reward for result in results]))
            mispredictions.append(
                np.array([result.mispredictions for result in results]))
            depths.append(depth)

    rewards = np.array(rewards)
    rewards_mean = np.array(
        [np.mean(rewards[i]) for i in range(rewards.shape[0])])
    rewards_std = np.array(
        [np.std(rewards[i]) for i in range(rewards.shape[0])])

    fidelity = (100. - np.array(mispredictions)) / 100.
    fidelity_mean = np.array(
        [np.mean(fidelity[i]) for i in range(fidelity.shape[0])])
    fidelity_std = np.array(
        [np.std(fidelity[i]) for i in range(fidelity.shape[0])])

    file_name = os.path.join(
        output_dir,
        '{}_e{}_reward-fidelity-trend.png'.format(model, experts))
    plot_title = (
        r'{}: {}_e{} reward and fidelity $\mu$ and $\pm \sigma$ interval'
            .format(subject, model, experts))
    create_trend_plot(depths=depths,
                      rewards_mean=rewards_mean,
                      rewards_std=rewards_std,
                      fidelity_mean=fidelity_mean,
                      fidelity_std=fidelity_std,
                      file_name=file_name,
                      plot_title=plot_title)


def create_trend_plots(db, output_dir):
    for subject in SUBJECTS:
        subject_output_dir = os.path.join(output_dir, subject, 'trend_plots')
        create_Viper_trend_plot(subject,
                                db,
                                output_dir=subject_output_dir)
        for experts in EXPERTS[subject]:
            for model in [MOE_MODEL, MOEHARD_MODEL]:
                create_MOET_trend_plot(subject,
                                       model,
                                       experts,
                                       db,
                                       output_dir=subject_output_dir)


OUTPUT_DIR = '/home/UNK/Downloads/plots/'

def main():
    db_path = get_db_path()
    db = Database(db_path)
    db.open()

    create_trend_plots(db, output_dir=OUTPUT_DIR)
    create_pareto_front_plots(db, output_dir=OUTPUT_DIR)
    create_box_plots(db, output_dir=OUTPUT_DIR)

    db.close()


if __name__ == '__main__':
    main()
