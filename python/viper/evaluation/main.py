from viper.util.latex import create_latex_macro
from viper.evaluation.constants import *
from viper.evaluation.util import *
from viper.core.dt import DTPolicy
from viper.core.rl import train_dagger
from viper.evaluation.constants import get_params
from viper.evaluation.constants import ConfigType
from viper.util.serialization import save_policy, load_policy
from viper.core.dt import save_dt_policy_viz
from viper.core.dt import save_dt_viz
from viper.core.rl import test_policy
from viper.core.rl import TransformerPolicy
from viper.core.moe_policy import MOEPolicy
from viper.util.log import *
import viper.util.util as util
from viper.core.compare_policy import ComparePolicy

import os
import argparse
import time


def get_results_dirname(params):
    """Returns a directory where results are saved."""

    config_dirname = params.config_type.name

    if not params.out_dir_name:
        return os.path.join(RESULTS_DIR,
                            params.subject_name,
                            config_dirname)
    else:
        return os.path.join(RESULTS_DIR,
                            params.subject_name,
                            params.out_dir_name,
                            config_dirname)


def get_policy_filename(params):
    """Returns a name of the policy file."""

    if (params.config_type == ConfigType.Viper
        or params.config_type == ConfigType.ViperPlus):
        return 'dt_policy_d{0}.pk'.format(params.max_depth)
    elif (params.config_type == ConfigType.MOE
          or params.config_type == ConfigType.MOEHard):
        return 'moe_policy_e{0}_d{1}.pk'.format(params.experts_no,
                                                params.experts_depths)


def get_evaluation_file(params):
    """Returns a path to the file for evaluation results."""

    dirname = get_results_dirname(params)
    fname = '{0}_evaluation.tex'.format(params.config_type.name)
    return os.path.join(dirname, fname)


def get_config_string(params):
    if (params.config_type == ConfigType.Viper
            or params.config_type == ConfigType.ViperPlus):
        return '{0}_{1}_d{2}'.format(params.subject_name,
                                     params.config_type.name,
                                     params.max_depth)
    elif (params.config_type == ConfigType.MOE
              or params.config_type == ConfigType.MOEHard):
        return '{0}_{1}_e{2}_d{3}'.format(params.subject_name,
                                          params.config_type.name,
                                          params.experts_no,
                                          params.experts_depths)
    elif params.config_type == ConfigType.DRL:
        return '{0}_{1}'.format(params.subject_name,
                                params.config_type.name)


def set_log_file(params, command_name):
    file_dir = get_results_dirname(params)
    file_name = '{0}_{1}.txt'.format(command_name,
                                     get_config_string(params))
    file_path = os.path.join(file_dir, 'logs', file_name)
    util.ensure_parent_exists(file_path)
    # Delete old log file if it exists.
    util.delete_if_exists(file_path)
    set_file(file_path)


def learn(params):
    set_log_file(params, 'learn')
    log('Learning with configuration: {0}'.format(get_config_string(params)),
        INFO)
    log('Params values: {0}'.format(params), INFO)

    if params.config_type == ConfigType.DRL:
        raise Exception('Learning DRL agent not supported.')

    save_dirname = get_results_dirname(params)
    save_fname = get_policy_filename(params)

    env = create_gym_env(params.subject_name)
    teacher = load_rl_policy(params.subject_name)

    if (params.config_type == ConfigType.Viper
            or params.config_type == ConfigType.ViperPlus):
        student = DTPolicy(
            max_depth=(params.max_depth if params.max_depth != -1 else None))
    elif (params.config_type == ConfigType.MOE
          or params.config_type == ConfigType.MOEHard):
        hard_prediction = params.config_type == ConfigType.MOEHard
        student = MOEPolicy(experts_no=params.experts_no,
                            dts_depth=params.experts_depths,
                            num_classes=params.action_space,
                            hard_prediction=hard_prediction,
                            max_epoch=params.moe_max_epoch,
                            init_learning_rate=params.moe_init_learning_rate,
                            learning_rate_decay=params.moe_learning_rate_decay,
                            log_frequency=params.moe_log_frequency,
                            stop_count=params.moe_stop_count,
                            regularization_mode=params.moe_regularization_mode,
                            use_new_formula=params.moe_use_new_formula,
                            use_adam_optimizer=params.moe_use_adam_optimizer)

    state_transformer = get_state_transformer(params.subject_name)
    student = train_dagger(env=env,
                           teacher=teacher,
                           student=student,
                           state_transformer=state_transformer,
                           max_iters=params.max_iters,
                           n_batch_rollouts=params.n_batch_rollouts,
                           max_samples=params.max_samples,
                           train_frac=params.train_frac,
                           is_reweight=params.is_reweight,
                           n_test_rollouts=params.n_test_rollouts,
                           identify_best=params.choose_best_student_strategy,
                           min_episode_reward=params.min_episode_reward,
                           max_episode_reward=params.max_episode_reward)

    save_policy(student, save_dirname, save_fname)
    if (params.config_type == ConfigType.Viper
            or params.config_type == ConfigType.ViperPlus):
        save_viz_fname, _ = os.path.splitext(save_fname)
        save_viz_fname = save_viz_fname + '.dot'
        save_dt_policy_viz(student, save_dirname, save_viz_fname,
                           params.feature_names, params.action_names)
        log('Number of nodes in DT: {}'.format(student.tree.tree_.node_count),
            INFO)
        log('Depth of DT: {}'.format(student.tree.tree_.max_depth), INFO)
    elif (params.config_type == ConfigType.MOE
              or params.config_type == ConfigType.MOEHard):
        basename, _ = os.path.splitext(save_fname)
        for i in range(len(student.moe.dtc_list)):
            save_viz_fname = basename + '_DT{0}'.format(i) + '.dot'
            save_dt_viz(student.moe.dtc_list[i], save_dirname, save_viz_fname,
                        params.feature_names, params.action_names)

    current_time = time.time()
    key_header = get_config_string(params)
    evaluation_file = get_evaluation_file(params)
    tuples = [(evaluation_file, key_header)]

    # if params.config_type == ConfigType.MOE:
    #     # TODO: Rewrite nicer
    #     # Ensure that we add info for MOEHard as well.
    #     new_params = params._replace(config_type=ConfigType.MOEHard)
    #     evaluation_file_hard = get_evaluation_file(new_params)
    #     key_header_hard = get_config_string(new_params)
    #     tuples.append((evaluation_file_hard, key_header_hard))

    for file, key_header in tuples:
        with open(file, 'a') as f:
            f.write(create_latex_macro(
                key=key_header + '_unix_timestamp',
                value=str(current_time)))
            f.write(os.linesep)

            f.write(create_latex_macro(
                key=key_header + '_choose_best_student_strategy',
                value=params.choose_best_student_strategy))
            f.write(os.linesep)

            f.write(create_latex_macro(
                key=key_header + '_max_iters',
                value=params.max_iters))
            f.write(os.linesep)


def evaluate_policy(params, policy):
    """Note that the policy plays with raw observation space input; no
    state transformation is done; thus wrap your policy with state
    transformer if it accepts transformed observation space."""
    n_test_rollouts = params.n_test_rollouts
    env = create_gym_env(params.subject_name)
    return test_policy(env, policy, n_test_rollouts)


def evaluate(params):
    set_log_file(params, 'evaluate')
    log('Evaluating with configuration: {0}'.format(get_config_string(params)),
        INFO)
    log('Params values: {0}'.format(params), INFO)

    if params.config_type != ConfigType.DRL:
        dirname = get_results_dirname(params)
        fname = get_policy_filename(params)
        student_policy = load_policy(dirname, fname)
    else:
        student_policy = load_rl_policy(params.subject_name)

    # TODO: Rewrite nicer
    if (params.config_type == ConfigType.MOE
            or params.config_type == ConfigType.MOEHard):
        student_policy.hard_prediction = params.config_type == ConfigType.MOEHard

    teacher_policy = load_rl_policy(params.subject_name)

    log('### Evaluating Policy', INFO)

    if params.config_type != ConfigType.DRL:
        state_transformer = get_state_transformer(params.subject_name)
        wrapped_student = TransformerPolicy(student_policy,
                                            state_transformer)
    else:
        wrapped_student = student_policy

    cmp_policy = ComparePolicy(wrapped_student, teacher_policy)
    reward = evaluate_policy(params, cmp_policy)
    mispredictions_student_playing = cmp_policy.mispredictions_ratio() * 100

    cmp_policy = ComparePolicy(teacher_policy, wrapped_student)
    evaluate_policy(params, cmp_policy)
    mispredictions_teacher_playing = cmp_policy.mispredictions_ratio() * 100

    log('Reward: {0}'.format(str(reward)), INFO)
    log('Mispredictions (student playing): {0}%'.format(
        mispredictions_student_playing), INFO)
    log('Mispredictions (teacher playing): {0}%'.format(
        mispredictions_teacher_playing), INFO)

    with open(get_evaluation_file(params), 'a') as f:
        key_header = get_config_string(params)

        f.write(create_latex_macro(key=key_header + '_reward',
                                   value='{0:.2f}'.format(
                                   reward)))
        f.write(os.linesep)

        f.write(create_latex_macro(
            key=key_header + '_mispredictions_student_playing',
            value='{0:.2f}'.format(mispredictions_student_playing)))
        f.write(os.linesep)

        f.write(create_latex_macro(
            key=key_header + '_mispredictions_teacher_playing',
            value='{0:.2f}'.format(mispredictions_teacher_playing)))
        f.write(os.linesep)

        if isinstance(student_policy, MOEPolicy):
            moe_policy = student_policy

            f.write(create_latex_macro(
                key=key_header + '_moe_max_epoch',
                value=moe_policy.max_epoch))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_init_learning_rate',
                value=moe_policy.init_learning_rate))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_learning_rate_decay',
                value=moe_policy.learning_rate_decay))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_log_frequency',
                value=moe_policy.log_frequency))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_stop_count',
                value=moe_policy.stop_count))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_regularization_mode',
                value=moe_policy.regularization_mode))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_use_new_formula',
                value=moe_policy.use_new_formula))
            f.write(os.linesep)
            f.write(create_latex_macro(
                key=key_header + '_moe_use_adam_optimizer',
                value=moe_policy.use_adam_optimizer))
            f.write(os.linesep)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_name',
                        type=str,
                        help='Name of the subject.')
    parser.add_argument('--config_type',
                        type=str,
                        help='Type of a model to use.')
    parser.add_argument('--function',
                        type=str,
                        help='Name of the function to invoke.')
    parser.add_argument('--max_depth',
                        type=int,
                        help='Depth of DT, unlimited if -1. '
                             'Used if global DT is trained.')
    parser.add_argument('--experts_no',
                        type=int,
                        help='Number of experts (used only for MOE)')
    parser.add_argument('--experts_depths',
                        type=int,
                        help='Depths of experts (used only for MOE)')
    parser.add_argument('--choose_best_student_strategy',
                        type=str,
                        default=None,
                        help='Strategy to use when choosing the best student '
                             '(during DAGGER training).')
    parser.add_argument('--use_new_formula',
                        type=str2bool,
                        default=None,
                        help='Whether to use new formula for MOET.')
    parser.add_argument('--use_adam_optimizer',
                        type=str2bool,
                        default=None,
                        help='Whether to use adam optimizer for MOET.')
    parser.add_argument('--moe_max_epoch',
                        type=int,
                        default=None,
                        help='Maximum number of training epochs in MOET.')
    parser.add_argument('--moe_init_learning_rate',
                        type=float,
                        default=None,
                        help='Initial learning rate in MOET..')
    parser.add_argument('--moe_learning_rate_decay',
                        type=float,
                        default=None)
    parser.add_argument('--moe_log_frequency',
                        type=int,
                        default=None)
    parser.add_argument('--moe_stop_count',
                        type=int,
                        default=None)
    parser.add_argument('--moe_regularization_mode',
                        type=int,
                        default=None)
    parser.add_argument('--max_iters',
                        type=int,
                        default=None)
    parser.add_argument('--is_reweight',
                        type=str2bool,
                        default=None,
                        help='Whether to use sampling based on q-value'
                             'when picking the points to train students on.')
    parser.add_argument('--out_dir_name',
                        type=str,
                        default=None,
                        help='Name of the output directory, '
                             'if not provided default directory will be used.')

    args = parser.parse_args()
    args.config_type = ConfigType[args.config_type]
    return args


def main(args):
    params = get_params(args.subject_name, args.config_type)
    params = params._replace(max_depth=args.max_depth)
    params = params._replace(experts_no=args.experts_no)
    params = params._replace(experts_depths=args.experts_depths)
    params = params._replace(out_dir_name=args.out_dir_name)
    if args.choose_best_student_strategy:
        params = params._replace(
            choose_best_student_strategy=args.choose_best_student_strategy)
    if args.use_new_formula is not None:
        params = params._replace(
            moe_use_new_formula=args.use_new_formula
        )
    if args.use_adam_optimizer is not None:
        params = params._replace(
            moe_use_adam_optimizer=args.use_adam_optimizer
        )
    if args.moe_max_epoch is not None:
        params = params._replace(
            moe_max_epoch=args.moe_max_epoch
        )
    if args.moe_init_learning_rate is not None:
        params = params._replace(
            moe_init_learning_rate=args.moe_init_learning_rate
        )
    if args.moe_learning_rate_decay is not None:
        params = params._replace(
            moe_learning_rate_decay=args.moe_learning_rate_decay
        )
    if args.moe_log_frequency is not None:
        params = params._replace(
            moe_log_frequency=args.moe_log_frequency
        )
    if args.moe_stop_count is not None:
        params = params._replace(
            moe_stop_count=args.moe_stop_count
        )
    if args.moe_regularization_mode is not None:
        params = params._replace(
            moe_regularization_mode=args.moe_regularization_mode
        )
    if args.max_iters is not None:
        params = params._replace(
            max_iters=args.max_iters
        )
    if args.is_reweight is not None:
        params = params._replace(
            is_reweight=args.is_reweight
        )

    func = globals()[args.function]
    func(params)


if __name__ == '__main__':
    main(build_cmd_args())
