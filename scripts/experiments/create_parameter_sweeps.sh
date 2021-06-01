#!/bin/bash

# This scripts creates commands with different values of parameters.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly MAX_EPOCHS=(50 100 500)
readonly INIT_LEARNING_RATES=(1. 0.3 0.1 0.01 0.001 0.0001 0.00001)
readonly LEARNING_RATE_DECAYS=(1.)
readonly LOG_FREQUENCIES=(-1)
readonly MAX_ITERS=(40)

readonly EXPERIMENT_NAME="sweeps"

# Returns a command for running evaluation.
function cmd_run_evaluation() {
    local subject_name="${1}"; shift
    local function_name="${1}"; shift
    local config_type="${1}"; shift
    local max_depth="${1}"; shift
    local experts_no="${1}"; shift
    local experts_depths="${1}"; shift
    local out_dir_name="${1}"; shift
    local use_new_formula="${1}"; shift
    local choose_student_strategy="${1}"; shift
    local max_epoch="${1}"; shift
    local init_learning_rate="${1}"; shift
    local learning_rate_decay="${1}"; shift
    local log_frequency="${1}"; shift
    local max_iters="${1}"; shift
    local use_adam_optimizer="${1}"; shift

    echo "python -m viper.evaluation.main" \
	"--subject_name=${subject_name}" \
	"--config_type=${config_type}" \
        "--function=${function_name}" \
        "--max_depth=${max_depth}" \
	"--experts_no=${experts_no}" \
	"--experts_depths=${experts_depths}" \
	"--out_dir_name=${out_dir_name}" \
	"--use_new_formula=${use_new_formula}" \
	"--choose_best_student_strategy=${choose_student_strategy}" \
	"--moe_max_epoch=${max_epoch}" \
	"--moe_init_learning_rate=${init_learning_rate}" \
	"--moe_learning_rate_decay=${learning_rate_decay}" \
	"--moe_log_frequency=${log_frequency}" \
	"--max_iters=${max_iters}" \
	"--use_adam_optimizer=${use_adam_optimizer}"
}

function generate_params() {
    local subject_name="${1}"; shift
    local config="${1}"; shift
    local use_new_formula="${1}"; shift
    local choose_student_strategy="${1}"; shift
    local use_adam_optimizer="${1}"; shift

    local uppercase_subject_name=$(printf '%s\n' "${subject_name}" | awk '{ print toupper($0) }')

    if [ "${config}" = "all" ] || [ "${config}" = "MOE" ]; then
	max_depth=0

	for((i=0;i<${#EXPERT_DEPTH_PAIRS[@]};i+=2)); do
	    experts_no=${EXPERT_DEPTH_PAIRS[$i]}
	    experts_depths=${EXPERT_DEPTH_PAIRS[$i+1]}
            for config_type in "MOE" "MOEHard"; do
    	    for max_epoch in ${MAX_EPOCHS[@]}; do
    	        for init_learning_rate in ${INIT_LEARNING_RATES[@]}; do
    		    for learning_rate_decay in ${LEARNING_RATE_DECAYS[@]}; do
    		        for log_frequency in ${LOG_FREQUENCIES[@]}; do
    			    for max_iters in ${MAX_ITERS[@]}; do
    			        # Create random name for output directory.
    			        out_dir_name="${EXPERIMENT_NAME}/$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)"
    			        learn_cmd=$(cmd_run_evaluation "${subject_name}" "learn" "${config_type}" "${max_depth}" \
                                                                   "${experts_no}" "${experts_depths}" "${out_dir_name}" \
                                                                   "${use_new_formula}" "${choose_student_strategy}" \
    				                               "${max_epoch}" "${init_learning_rate}" "${learning_rate_decay}" \
                                                                   "${log_frequency}" "${max_iters}" "${use_adam_optimizer}")
    			        evaluate_cmd=$(cmd_run_evaluation "${subject_name}" "evaluate" "${config_type}" "${max_depth}" \
                                                                      "${experts_no}" "${experts_depths}" "${out_dir_name}" \
                                                                      "${use_new_formula}" "${choose_student_strategy}" \
    			    	                                  "${max_epoch}" "${init_learning_rate}" "${learning_rate_decay}" \
                                                                      "${log_frequency}" "${max_iters}" "${use_adam_optimizer}")
    			        echo "(cd ../../python && ${learn_cmd} && ${evaluate_cmd})"
                            done
    			done
    		    done
    		done
    	    done
    	    done
	done
    fi
}

function create_param_file() {
    local out_file_name="${1}"; shift
    local subject_name="${1}"; shift
    local config="${1}"; shift
    # How many times to schedule experiment.
    local repetitions="${1}"; shift
    local use_new_formula="${1}"; shift
    local choose_student_strategy="${1}"; shift
    local use_adam_optimizer="${1}"; shift

    local paramdir="${SCRIPT_DIR}/paramfiles"
    # Ensure directory exist.
    mkdir -p "${paramdir}"
    if [ -z ${out_file_name} ]; then
	local paramfile="${paramdir}/paramlist_${subject_name}"
    else
	local paramfile="${paramdir}/${out_file_name}"
    fi
    # Delete old file if it exists.
    rm -f "${paramfile}"

    local i
    for ((i = 0; i < ${repetitions}; i++)); do
	local params=$(generate_params "${subject_name}" "${config}" "${use_new_formula}" "${choose_student_strategy}" "${use_adam_optimizer}")
	echo "${params}" >> "${paramfile}"
    done
    echo "${paramfile}"
}

# Default values.
config="all"
repetitions=1 # How many times to repeat experiment.
use_new_formula="True"
choose_student_strategy="reward_and_mispredictions"
# choose_student_strategy="reward_and_mispredictions_harmonic"
use_adam_optimizer="True"

while [ $# -gt 0 ]; do
    case "$1" in
	--subject=*)
	    subject="${1#*=}"
	    ;;
	--config=*)
	    # configuration: Viper or MOE.
	    config="${1#*=}"
	    ;;
	--repetitions=*)
	    repetitions="${1#*=}"
	    ;;
	--use_new_formula=*)
	    use_new_formula="${1#*=}"
	    ;;
	--choose_student_strategy=*)
	    choose_student_strategy="${1#*=}"
	    ;;
	--use_adam_optimizer=*)
	    use_adam_optimizer="${1#*=}"
	    ;;
	*)
	    printf "***************************\n"
	    printf "* Error: Invalid argument.*\n"
	    printf "***************************\n"
	    exit 1
    esac
    shift
done

if [ -z "${subject}" ]; then
    echo "You must specify the subject."
    exit 1
fi

if [ "${subject}" = "lunarlander" ]; then
    readonly EXPERT_DEPTH_PAIRS=(8 17 6 3 6 17 7 0)
elif [ "${subject}" = "cartpole" ]; then
    readonly EXPERT_DEPTH_PAIRS=(2 0 4 0 8 0)
elif [ "${subject}" = "acrobot" ]; then
    readonly EXPERT_DEPTH_PAIRS=(16 11 2 2 15 11)
elif [ "${subject}" = "mountaincar" ]; then
    readonly EXPERT_DEPTH_PAIRS=(2 3 2 5 2 8 3 6 3 7 4 1 4 2 4 4 4 5 5 1 6 1 7 2 7 6 8 2 8 5)
elif [ "${subject}" = "pong" ]; then
    readonly EXPERT_DEPTH_PAIRS=(16 21)
else
    echo "Unrecognized subject"
    exit 1
fi

out_file_name="paramlist_${subject}_parameter_${EXPERIMENT_NAME}"
paramfile=$(create_param_file "${out_file_name}" "${subject}" "${config}" "${repetitions}" \
    "${use_new_formula}" "${choose_student_strategy}" "${use_adam_optimizer}")
