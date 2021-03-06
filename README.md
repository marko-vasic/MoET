# MoËT: Mixture of Expert Trees and its Application to Verifiable Reinforcement Learning
In this project we introduce Mixture of Expert Trees (MoËT), which enhances Mixture of Experts (MoE) framework to support non-differentiable experts.

Next, we provide instructions on how to use the code and how to reproduce paper results.

## Relevant publication
Paper that presents work of this project is:
```
@article{vasicETAL22MoET,
  title = {MoËT: Mixture of Expert Trees and its application to verifiable reinforcement learning},
  journal = {Neural Networks},
  volume = {151},
  pages = {34-47},
  year = {2022},
  author = {Marko Vasić and Andrija Petrović and Kaiyuan Wang and Mladen Nikolić and Rishabh Singh and Sarfraz Khurshid}
}
```
If you would like to reference this work in an academic publication please cite the previous paper.

## Running code

### Install requirements
- Create a virtualenv. We created virtualenv with Python 2 (Python
  2.7.17) by running: `virtualenv ${OPT}/v_env`
- Activate the environment: `source ${OPT}/v_env/bin/activate`
- Install necessary requirements: `pip install -r requirements.txt`

### Basic Workflow
**Training a model.** To train a model, do the following from the project root directory:

``` shell
cd python
python -m viper.evaluation.main \
    --subject_name=cartpole \
    --config_type=ViperPlus \
    --function=learn \
    --max_depth=2 \
    --max_iters=5 \
    --out_dir_name=test \
    --use_new_formula=True \
    --use_adam_optimizer=True \
    --choose_best_student_strategy=reward_and_mispredictions
```
This trains a Viper model with maximum depth 2, for 5 iterations of Dagger (we used 40 in our work). 
Output will be stored in `${PROJECT_ROOT}/data/experiments/cartpole/test/ViperPlus/`.
In that folder you can convert dot file representing student policy into a pdf file to visualize it, e.g., using: `dot -Tpdf dt_policy_d2.dot -o dt_policy_d2.pdf`.
Example of the learned policy:
![DT Policy](data/experiments/cartpole/example.svg)
Features of the observation space in cartpole are represented by *cp* (cart position), *cv* (cart velocity), *pa* (pole angle) and *pv* (pole angular velocity).

**Evaluating a model.** You can evaluate the trained model with a following command:

``` shell
python -m viper.evaluation.main \
    --subject_name=cartpole \
    --config_type=ViperPlus \
    --function=evaluate \
    --max_depth=2 \
    --max_iters=5 \
    --out_dir_name=test \
    --use_new_formula=True \
    --use_adam_optimizer=True \
    --choose_best_student_strategy=reward_and_mispredictions
```
The experiment results will be stored in `${PROJECT_ROOT}/data/experiments/cartpole/test/ViperPlus/ViperPlus_evaluation.tex`

**MoËT.** Similarly, you can train a MOET model:

``` shell
python -m viper.evaluation.main \
    --subject_name=cartpole \
    --config_type=MOE \
    --function=learn \
    --experts_no=2 \
    --experts_depths=0 \
    --max_iters=5 \
    --out_dir_name=test \
    --use_new_formula=True \
    --use_adam_optimizer=True \
    --choose_best_student_strategy=reward_and_mispredictions
```
**MoËTHard.** Or, a MoËTHard model:

``` shell
python -m viper.evaluation.main \
    --subject_name=cartpole \
    --config_type=MOEHard \
    --function=learn \
    --experts_no=2 \
    --experts_depths=0 \
    --max_iters=5 \
    --out_dir_name=test \
    --use_new_formula=True \
    --use_adam_optimizer=True \
    --choose_best_student_strategy=reward_and_mispredictions
```

### Running multiple configurations
To create different configurations to train various models as we did in our evaluation, do the following:

``` shell
cd ${PROJECT_ROOT}/scripts/experiments
./create_configurations.sh --subject=cartpole
```
This will create configurations for Cartpole environment, in a similar way you can create it for the other environments
(currently supported subjects are: *cartpole*, *lunarlander*, *acrobot*, *mountaincar* and *pong*).
Inside of the `paramfiles/paramlist_cartpole` you will find a list of bash commands one experiment per line.
You can execute the commands in a way you like (in a single thread, multiple threads, distributed), and in that way train models for different configurations.

Additionally, when we trained MOET models in the previous way we examined couple of the best configurations and performed parameter sweeps on them.
Note that the previous trains all MOET models with the same initial learning rate and other hyperparameters.
To create configurations with parameter sweeps you can run the following:

``` shell
cd ${PROJECT_ROOT}/scripts/experiments
./create_parameter_sweeps.sh --subject=cartpole
```
The parameter file will be saved in `paramfiles/paramlist_cartpole_parameter_sweeps`.

You can do the same for other subjects. You can also encode yourself
which configurations you want to do experiments with by controlling
value of variable `EXPERT_DEPTH_PAIRS` in
`create_parameter_sweeps.sh`.

### Verification
We have included an cartpole MOET_h model that is verified under
`${PROJECT_ROOT}/data/experiments/verified`. You can run its
verification by executing:
``` shell
cd python
python -m viper.verification.cartpole
```
