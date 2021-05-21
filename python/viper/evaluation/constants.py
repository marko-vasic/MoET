import collections
import os
from enum import Enum
import math


class ConfigType(Enum):
    Viper = 1
    ViperPlus = 2
    MOE = 3
    # MOE using hard prediction.
    MOEHard = 4
    # DRL agent.
    DRL = 5


DATA_DIR = '../data'
RESULTS_DIR = os.path.join(DATA_DIR, 'experiments')
EVALUATION_FILE_SUFFIX = '_evaluation.tex'

# Whether to show MOET results with Adam optimizer (1) or without (0).
SHOW_ADAM_OPTIMIZER_RESULTS = collections.defaultdict(lambda: 1)

SUBJECTS = [
    'cartpole',
    'pong',
    'acrobot',
    'mountaincar',
    'lunarlander'
]

ViperPlus_MODEL = 'ViperPlus'
MOE_MODEL = 'MOE'
MOEHARD_MODEL = 'MOEHard'

DEPTHS = {
    'cartpole': range(1, 9),
    'pong': [5, 10, 15, 20, 25],
    'acrobot': range(1, 16, 1),
    'mountaincar': range(1, 13),
    'lunarlander': range(1, 21)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
}
EXPERTS = {
    'cartpole': range(2, 9),
    'pong': [2, 4, 8, 16, 32],
    'acrobot': [2, 3, 4, 5, 6, 7, 8, 15, 16],
    'mountaincar': [2, 3, 4, 5, 6, 7, 8, 12, 16],
    'lunarlander': range(2, 9),
}
EXPERT_DEPTHS = {
    'cartpole': range(0, 9),
    'pong': range(0, 26),
    'acrobot': range(0, 16),
    'mountaincar': range(0, 13),
    'lunarlander': range(0, 21)
}


# How many experiment results should be taken when creating
# a summary of results.
# Ensures that we use k runs for all configurations
# even when we have more runs for some configuration.
RUNS_FOR_SUMMARY = collections.defaultdict(lambda: 5)
# RUNS_FOR_SUMMARY['pong'] = 5
# RUNS_FOR_SUMMARY['acrobot'] = 7

# How many MOET configurations match a given Viper configuration.
# This is used to match number of experiments for Viper and MOET for same
# effective depth
VIPER_RUN_COPIES = dict()


def set_viper_run_copies():
    for subject in SUBJECTS:
        VIPER_RUN_COPIES[subject] = collections.defaultdict(lambda: 0)
        for depth in DEPTHS[subject]:
            number_moet_configs = 0
            for experts in EXPERTS[subject]:
                for expert_depth in EXPERT_DEPTHS[subject]:
                    if (int(math.ceil(math.log(experts, 2))) + expert_depth
                         == depth):
                        number_moet_configs += 1
            VIPER_RUN_COPIES[subject]['d{}'.format(depth)] = number_moet_configs


set_viper_run_copies()


"""
Parameters for the evaluation.
"""
Params = collections.namedtuple('Params', [
    # str: Name of the subject (e.g., cartpole).
    'subject_name',
    # Type of the configuration (e.g, Viper).
    'config_type',
    # int: Number of plausible actions.
    'action_space',
    # float: Minimum possible reward in an episode.
    'min_episode_reward',
    # float: Maximum possible reward in an episode.
    'max_episode_reward',
    # int: Maximum depth of DT in a case of Viper.
    'max_depth',
    # int: Number of experts used in MOE.
    'experts_no',
    # int: Maximum depth of expert DTs of MOE.
    'experts_depths',
    # int: Number of game rollouts used during training.
    'n_batch_rollouts',
    # int: Number of game rollouts used for testing.
    'n_test_rollouts',
    # int: Maximum number of samples with which to train.
    'max_samples',
    # int: Number of iterations of Viper (DAGGER) algorithm.
    'max_iters',
    # Fraction of datapoints used for training purposes.
    'train_frac',
    # If True, then Q values are used to sample points for
    # training.
    'is_reweight',
    # Algorithm using which to identify the best student policy.
    'choose_best_student_strategy',
    # Labels of features in environment's observation space;
    # this is used for nicer visualization (specify None if you do not
    # want to use).
    'feature_names',
    # Similar to feature labels but for actions.
    'action_names',
    # Maximum number of epochs in MOE training.
    'moe_max_epoch',
    # Initial learning rate in MOE training.
    'moe_init_learning_rate',
    # Learning rate decay in MOE training.
    'moe_learning_rate_decay',
    # Log frequency in MOE training.
    'moe_log_frequency',
    # If validation accuracy does not improve for this many times, than
    # training is stopped.
    "moe_stop_count",
    # Regularization model to be used.
    "moe_regularization_mode",
    # Whether to use new formula for MOET.
    "moe_use_new_formula",
    # Whether to use adam optimizer for MOET.
    "moe_use_adam_optimizer",
    # Name of the output directory where results will be saved.
    'out_dir_name'
])

##################
## CARTPOLE Params
##################

cartpoleViperParams = Params(subject_name='cartpole',
                             config_type=None,
                             action_space=2,
                             min_episode_reward=0.,
                             max_episode_reward=200.,
                             max_depth=None,
                             experts_no=None,
                             experts_depths=None,
                             n_batch_rollouts=10,
                             # n_test_rollouts was 100 in Viper
                             n_test_rollouts=250,
                             max_samples=200000,
                             # Max iters were 80 in Viper.
                             max_iters=40,
                             train_frac=0.8,
                             is_reweight=True,
                             choose_best_student_strategy='reward_only',
                             feature_names=['cp', 'cv', 'pa', 'pv'],
                             action_names=['l', 'r'],
                             moe_max_epoch=None,
                             moe_init_learning_rate=None,
                             moe_learning_rate_decay=None,
                             moe_log_frequency=None,
                             moe_stop_count=None,
                             moe_regularization_mode=None,
                             moe_use_new_formula=None,
                             moe_use_adam_optimizer=None,
                             out_dir_name=None)

cartpoleViperPlusParams = cartpoleViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

cartpoleDRLParams = cartpoleViperPlusParams

cartpoleMOEParams = cartpoleViperPlusParams._replace(
    # TODO: Remove this when bug with adding bias to DT in MOE is fixed.
    feature_names=['cp', 'cv', 'pa', 'pv', 'bias'],
    moe_max_epoch=50,
    moe_init_learning_rate=0.3,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=None,
    moe_stop_count=None,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

cartpoleMOEHardParams = cartpoleMOEParams

##################
## Pong Params
##################

pongViperParams = Params(subject_name='pong',
                         config_type=None,
                         action_space=6,
                         min_episode_reward=-21.,
                         max_episode_reward=21.,
                         max_depth=None,
                         experts_no=None,
                         experts_depths=None,
                         n_batch_rollouts=10,
                         # Test rollouts was 50 in Viper
                         n_test_rollouts=100,
                         max_samples=200000,
                         # Max iters were 80 in Viper.
                         max_iters=40,
                         train_frac=0.8,
                         is_reweight=True,
                         choose_best_student_strategy='reward_only',
                         feature_names=None,
                         action_names=None,
                         moe_max_epoch=None,
                         moe_init_learning_rate=None,
                         moe_learning_rate_decay=None,
                         moe_log_frequency=None,
                         moe_stop_count=None,
                         moe_regularization_mode=None,
                         moe_use_new_formula=None,
                         moe_use_adam_optimizer=None,
                         out_dir_name=None)

pongViperPlusParams = pongViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

pongDRLParams = pongViperPlusParams

pongMOEParams = pongViperPlusParams._replace(
    moe_max_epoch=50,
    moe_init_learning_rate=0.3,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=None,
    moe_stop_count=None,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

pongMOEHardParams = pongMOEParams

##################
## Acrobot Params
##################

acrobotViperParams = Params(subject_name='acrobot',
                            config_type=None,
                            action_space=3,
                            min_episode_reward=-500.,
                            max_episode_reward=0.,
                            max_depth=None,
                            experts_no=None,
                            experts_depths=None,
                            n_batch_rollouts=10,
                            n_test_rollouts=100,
                            max_samples=200000,
                            # Max iters were 80 in Viper.
                            max_iters=40,
                            train_frac=0.8,
                            is_reweight=True,
                            choose_best_student_strategy='reward_only',
                            feature_names=['cosT1', 'sinT1', 'cosT2', 'sinT2',
                                           'Tdot1', 'Tdot2'],
                            action_names=['+1', '0', '-1'],
                            moe_max_epoch=None,
                            moe_init_learning_rate=None,
                            moe_learning_rate_decay=None,
                            moe_log_frequency=None,
                            moe_stop_count=None,
                            moe_regularization_mode=None,
                            moe_use_new_formula=None,
                            moe_use_adam_optimizer=None,
                            out_dir_name=None)

acrobotViperPlusParams = acrobotViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

acrobotDRLParams = acrobotViperPlusParams

acrobotMOEParams = acrobotViperPlusParams._replace(
    # TODO: Remove 'bias' when bug with adding bias to DT in MOE is fixed.
    feature_names=['cosT1', 'sinT1', 'cosT2', 'sinT2', 'Tdot1', 'Tdot2',
                   'bias'],
    moe_max_epoch=50,
    moe_init_learning_rate=0.3,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=None,
    moe_stop_count=None,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

acrobotMOEHardParams = acrobotMOEParams

##################
## Mountaincar Params
##################

mountaincarViperParams = Params(subject_name='mountaincar',
                                config_type=None,
                                action_space=3,
                                min_episode_reward=-200.,
                                max_episode_reward=0.,
                                max_depth=None,
                                experts_no=None,
                                experts_depths=None,
                                n_batch_rollouts=10,
                                n_test_rollouts=100,
                                max_samples=200000,
                                # Max iters were 80 in Viper.
                                max_iters=40,
                                train_frac=0.8,
                                is_reweight=True,
                                choose_best_student_strategy='reward_only',
                                feature_names=['position', 'velocity'],
                                action_names=['-1', '0', '+1'],
                                moe_max_epoch=None,
                                moe_init_learning_rate=None,
                                moe_learning_rate_decay=None,
                                moe_log_frequency=None,
                                moe_stop_count=None,
                                moe_regularization_mode=None,
                                moe_use_new_formula=None,
                                moe_use_adam_optimizer=None,
                                out_dir_name=None)

mountaincarViperPlusParams = mountaincarViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

mountaincarDRLParams = mountaincarViperPlusParams


mountaincarMOEParams = mountaincarViperPlusParams._replace(
    # TODO: Remove 'bias' when bug with adding bias to DT in MOE is fixed.
    feature_names=['position', 'velocity', 'bias'],
    moe_max_epoch=50,
    moe_init_learning_rate=0.3,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=None,
    moe_stop_count=None,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

mountaincarMOEHardParams = mountaincarMOEParams


gridworldViperParams = Params(subject_name='gridworld',
                              config_type=None,
                                action_space=4,
                                min_episode_reward=-200.,
                                max_episode_reward=0.,
                                max_depth=None,
                                experts_no=None,
                                experts_depths=None,
                                n_batch_rollouts=10,
                                n_test_rollouts=100,
                                max_samples=200000,
                                max_iters=40,
                                train_frac=0.8,
                                is_reweight=False,
                                choose_best_student_strategy='reward_only',
                                feature_names=['x', 'y'],
                                action_names=['left', 'right', 'up', 'down'],
                                moe_max_epoch=None,
                                moe_init_learning_rate=None,
                                moe_learning_rate_decay=None,
                                moe_log_frequency=None,
                                moe_stop_count=None,
                                moe_regularization_mode=None,
                                moe_use_new_formula=None,
                                moe_use_adam_optimizer=None,
                                out_dir_name=None)

gridworldViperPlusParams = gridworldViperParams._replace(
    choose_best_student_strategy='mispredictions_and_nodes')

gridworldMOEParams = gridworldViperPlusParams._replace(
    # TODO: Remove 'bias' when bug with adding bias to DT in MOE is fixed.
    feature_names=['x', 'y', 'bias'],
    moe_max_epoch=100, # 50, # 80,
    moe_init_learning_rate=2, # 2, # 1,
    moe_learning_rate_decay=0.97, # 0.95, # 0.97,
    moe_log_frequency=30, # 45, #3,
    moe_stop_count=3,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=False,
)

gridworldMOEHardParams = gridworldMOEParams

##################
## frozenLakeV0 Params
##################

frozenLakeV0ViperParams = Params(subject_name='frozenLakeV0',
                             config_type=None,
                             action_space=4,
                             min_episode_reward=0.,
                             max_episode_reward=200.,
                             max_depth=None,
                             experts_no=None,
                             experts_depths=None,
                             n_batch_rollouts=10,
                             # n_test_rollouts was 100 in Viper
                             n_test_rollouts=250,
                             max_samples=200000,
                             # Max iters were 80 in Viper.
                             max_iters=40,
                             train_frac=0.8,
                             is_reweight=True,
                             choose_best_student_strategy='reward_only',
                             feature_names=['position'],
                             action_names=['left', 'down', 'right', 'up'],
                             moe_max_epoch=None,
                             moe_init_learning_rate=None,
                             moe_learning_rate_decay=None,
                             moe_log_frequency=None,
                             moe_stop_count=None,
                             moe_regularization_mode=None,
                             moe_use_new_formula=None,
                             moe_use_adam_optimizer=None,
                             out_dir_name=None)

frozenLakeV0ViperPlusParams = frozenLakeV0ViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

frozenLakeV0DRLParams = frozenLakeV0ViperPlusParams

frozenLakeV0MOEParams = frozenLakeV0ViperPlusParams._replace(
    # TODO: Remove this when bug with adding bias to DT in MOE is fixed.
    feature_names=['position', 'bias'],
    moe_max_epoch=80,
    moe_init_learning_rate=1,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=3,
    moe_stop_count=3,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

frozenLakeV0MOEHardParams = cartpoleMOEParams

##################
## LunarLander Params
##################

lunarlanderViperParams = Params(subject_name='lunarlander',
                                config_type=None,
                                action_space=4,
                                min_episode_reward=None, # TODO: specify
                                max_episode_reward=None, # TODO: specify
                                max_depth=None,
                                experts_no=None,
                                experts_depths=None,
                                n_batch_rollouts=10,
                                n_test_rollouts=100,
                                max_samples=200000,
                                max_iters=40,
                                train_frac=0.8,
                                is_reweight=True,
                                choose_best_student_strategy='reward_only',
                                feature_names=['X',  # X coordinate
                                               # Y coordinate
                                               'Y',
                                               # velocity in x direction.
                                               'VEL_X',
                                               # velocity in y direction.
                                               'VEL_Y',
                                               # angle of lunar lander.
                                               'ANGLE',
                                               # angular velocity.
                                               'ANG_VEL',
                                               # left leg touches ground.
                                               'LLEG_TOUCH',
                                               # right leg touches ground.
                                               'RLEG_TOUCH'
                                               ],
                                action_names=['NOP',
                                              # fire left engine.
                                              'FIRE_L',
                                              # fire main engine.
                                              'FIRE_M',
                                              # fire right engine.
                                              'FIRE_R'],
                                moe_max_epoch=None,
                                moe_init_learning_rate=None,
                                moe_learning_rate_decay=None,
                                moe_log_frequency=None,
                                moe_stop_count=None,
                                moe_regularization_mode=None,
                                moe_use_new_formula=None,
                                moe_use_adam_optimizer=None,
                                out_dir_name=None)

lunarlanderViperPlusParams = lunarlanderViperParams._replace(
    choose_best_student_strategy='reward_and_mispredictions')

lunarlanderDRLParams = lunarlanderViperPlusParams

lunarlanderMOEParams = lunarlanderViperPlusParams._replace(
    feature_names=['X',  # X coordinate
                   #  Y coordinate
                   'Y',
                   # velocity in x direction.
                   'VEL_X',
                   # velocity in y direction.
                   'VEL_Y',
                   # angle of lunar lander.
                   'ANGLE',
                   # angular velocity.
                   'ANG_VEL',
                   # left leg touches ground.
                   'LLEG_TOUCH',
                   # right leg touches ground.
                   'RLEG_TOUCH',
                   'bias'
                   ],
    moe_max_epoch=50,
    moe_init_learning_rate=0.3,
    moe_learning_rate_decay=0.97,
    moe_log_frequency=None,
    moe_stop_count=None,
    moe_regularization_mode=0,
    moe_use_new_formula=True,
    moe_use_adam_optimizer=True,
)

lunarlanderMOEHardParams = lunarlanderMOEParams


def get_params(subject_name, config_type):
    params = globals()['{0}{1}Params'
        .format(subject_name, config_type.name)]
    return params._replace(config_type=config_type)
