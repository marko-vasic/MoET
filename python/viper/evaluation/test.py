from viper.evaluation.main import *


def test_cartpole():
    """Testing evaluation code."""
    subject_name = 'cartpole'

    config_type = ConfigType.ViperPlus
    params = get_params(subject_name, config_type)
    params = params._replace(max_depth=2)
    params = params._replace(max_iters=4)
    params = params._replace(n_test_rollouts=10)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOE
    params = get_params(subject_name, config_type)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    params = params._replace(max_iters=4)
    params = params._replace(n_test_rollouts=10)
    learn(params)
    evaluate(params)


def test_pong():
    """Testing evaluation code."""
    subject_name = 'pong'

    config_type = ConfigType.ViperPlus
    params = get_params(subject_name, config_type)
    params = params._replace(max_depth=2)
    params = params._replace(max_iters=2)
    params = params._replace(n_batch_rollouts=1)
    params = params._replace(n_test_rollouts=1)
    params = params._replace(max_samples=20000)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOE
    params = get_params(subject_name, config_type)
    params = params._replace(config_type=ConfigType.MOE)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    learn(params)
    evaluate(params)


def test_acrobot():
    """Testing evaluation code."""
    subject_name = 'acrobot'

    config_type = ConfigType.ViperPlus
    params = get_params(subject_name, config_type)
    params = params._replace(max_depth=2)
    params = params._replace(max_iters=4)
    params = params._replace(n_test_rollouts=10)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOE
    params = get_params(subject_name, config_type)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    params = params._replace(max_iters=4)
    params = params._replace(n_test_rollouts=10)
    params = params._replace(moe_max_epoch=20)
    params = params._replace(moe_use_adam_optimizer=True)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOEHard
    params = get_params(subject_name, config_type)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    params = params._replace(max_iters=4)
    params = params._replace(n_test_rollouts=10)
    params = params._replace(moe_max_epoch=20)
    params = params._replace(moe_use_adam_optimizer=True)
    learn(params)
    evaluate(params)


def test_mountaincar():
    """Testing evaluation code."""
    subject_name = 'mountaincar'

    config_type = ConfigType.ViperPlus
    params = get_params(subject_name, config_type)
    params = params._replace(max_depth=2)
    params = params._replace(max_iters=4)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOE
    params = get_params(subject_name, config_type)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    params = params._replace(max_iters=4)
    learn(params)
    evaluate(params)


def test_frozenLakeV0():
    """Testing evaluation code."""
    subject_name = 'frozenLakeV0'

    config_type = ConfigType.ViperPlus
    params = get_params(subject_name, config_type)
    params = params._replace(max_depth=2)
    params = params._replace(max_iters=4)
    learn(params)
    evaluate(params)

    config_type = ConfigType.MOE
    params = get_params(subject_name, config_type)
    params = params._replace(experts_no=2)
    params = params._replace(experts_depths=2)
    params = params._replace(max_iters=4)
    learn(params)
    evaluate(params)

    
def build_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True, type=str,
                        help='Name of the subject.')
    return parser.parse_args()


def main(args):
    func_name = 'test_{0}'.format(args.subject)
    func = globals()[func_name]
    func()


if __name__ == '__main__':
    main(build_cmd_args())
