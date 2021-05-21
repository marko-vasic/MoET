#!/bin/bash

# This script creates commands with different configurations.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly PONG_VIPERPLUS_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
readonly PONG_MOE_EXPERTS=(2 4 8 16 32)
readonly PONG_MOE_EXPERT_DEPTHS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)

readonly CARTPOLE_VIPERPLUS_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
readonly CARTPOLE_MOE_EXPERTS=(2 3 4 5 6 7 8)
readonly CARTPOLE_MOE_EXPERT_DEPTHS=(0 1 2 3 4 5 6 7)

readonly ACROBOT_VIPERPLUS_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
readonly ACROBOT_MOE_EXPERTS=(2 3 4 5 6 7 8 15 16)
readonly ACROBOT_MOE_EXPERT_DEPTHS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

readonly MOUNTAINCAR_VIPERPLUS_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
readonly MOUNTAINCAR_MOE_EXPERTS=(2 3 4 5 6 7 8)
readonly MOUNTAINCAR_MOE_EXPERT_DEPTHS=(0 1 2 3 4 5 6 7 8 9 10 11)

readonly LUNARLANDER_VIPERPLUS_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
readonly LUNARLANDER_MOE_EXPERTS=(2 3 4 5 6 7 8)
readonly LUNARLANDER_MOE_EXPERT_DEPTHS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

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
    local use_adam_otpimizer="${1}"; shift
    local choose_student_strategy="${1}"; shift

    echo "python -m viper.evaluation.main" \
	"--subject_name=${subject_name}" \
	"--config_type=${config_type}" \
        "--function=${function_name}" \
        "--max_depth=${max_depth}" \
	"--experts_no=${experts_no}" \
	"--experts_depths=${experts_depths}" \
	"--out_dir_name=${out_dir_name}" \
	"--use_new_formula=${use_new_formula}" \
	"--use_adam_optimizer=${use_adam_optimizer}" \
	"--choose_best_student_strategy=${choose_student_strategy}"
}

function generate_params() {
    local subject_name="${1}"; shift
    local config="${1}"; shift
    local use_new_formula="${1}"; shift
    local use_adam_optimizer="${1}"; shift
    local choose_student_strategy="${1}"; shift
    local results_dir_prefix="${1}"; shift
    local uppercase_subject_name=$(printf '%s\n' "${subject_name}" | awk '{ print toupper($0) }')

    # Create name for output directory.
    if [ ! -z ${results_dir_prefix} ]; then
	out_dir_name="${results_dir_prefix}/$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)"
    else
	out_dir_name=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    fi

    if [ "${config}" = "all" ] || [ "${config}" = "MOE" ]; then
	# MOE Evaluation
	declare -a 'moe_experts=("${'"${uppercase_subject_name}_MOE_EXPERTS"'[@]}")'
	declare -a 'moe_expert_depths=("${'"${uppercase_subject_name}_MOE_EXPERT_DEPTHS"'[@]}")'
	declare -a 'viperplus_depths=("${'"${uppercase_subject_name}_VIPERPLUS_DEPTHS"'[@]}")'
	max_depth=0

	for config_type in "MOE" "MOEHard"; do
	    for experts_no in "${moe_experts[@]}"; do
		for experts_depths in "${moe_expert_depths[@]}"; do
		    effective_depth=$(python -c "import math; print(int(math.ceil(math.log(${experts_no}, 2))) + ${experts_depths})")
		    if [[ ! " ${viperplus_depths[@]} " =~ " ${effective_depth} " ]]; then
			# There's no corresponding Viper depth
			continue
		    fi
    		    learn_cmd=$(cmd_run_evaluation "${subject_name}" "learn" "${config_type}" ${max_depth} ${experts_no} ${experts_depths} ${out_dir_name} "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}")
    		    evaluate_cmd=$(cmd_run_evaluation "${subject_name}" "evaluate" "${config_type}" ${max_depth} ${experts_no} ${experts_depths} ${out_dir_name} "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}")
		    echo "(cd ../../python && ${learn_cmd} && ${evaluate_cmd})"
		done
	    done
	done
    fi

    if [ "${config}" = "all" ] || [ "${config}" = "Viper" ]; then
	# Viper Evaluation
	declare -a 'viperplus_depths=("${'"${uppercase_subject_name}_VIPERPLUS_DEPTHS"'[@]}")'
	config_type="ViperPlus"
	experts_no=0
	experts_depths=0
	for max_depth in "${viperplus_depths[@]}"; do
            learn_cmd=$(cmd_run_evaluation "${subject_name}" "learn" "${config_type}" ${max_depth} ${experts_no} ${experts_depths} ${out_dir_name} "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}")
            evaluate_cmd=$(cmd_run_evaluation "${subject_name}" "evaluate" "${config_type}" ${max_depth} ${experts_no} ${experts_depths} ${out_dir_name} "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}")
	    # Wrap the command in subshell so it doesn't affect other
	    # processes. Previously there were issues maybe because of
	    # directory change.
            echo "(cd ../../python && ${learn_cmd} && ${evaluate_cmd})"
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
    local use_adam_optimizer="${1}"; shift
    local choose_student_strategy="${1}"; shift
    local results_dir_prefix="${1}"; shift

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
	local params=$(generate_params "${subject_name}" "${config}" "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}" "${results_dir_prefix}")
	echo "${params}" >> "${paramfile}"
    done
    echo "${paramfile}"
}

function schedule_runs() {
    paramfile="${1}"; shift
    subject_name="${1}"; shift

    ## Stampede2 queues:
    ## https://portal.tacc.utexas.edu/user-guides/stampede2#table5
    ## SKX are faster then KNL
    queue_name="normal"

    # for Pong might need to have higher N/n ratio because of memory.
    # maybe 10n per N.
    sbatch \
	-N 2 -n 48 \
	-t 12:00:00 \
	-p "${queue_name}" \
	-J "${subject_name}" \
	launcher.slurm "${paramfile}"
}

# Default values.
config="all"
repetitions=1 # How many times to repeat experiment.
use_new_formula="True"
use_adam_optimizer="False"
# choose_student_strategy="reward_and_mispredictions_harmonic"
choose_student_strategy="reward_and_mispredictions"
out_file_name=""

while [ $# -gt 0 ]; do
    case "$1" in
	--subject=*)
	    subject="${1#*=}"
	    ;;
	--config=*)
	    # configuration: Viper or MOE or all.
	    config="${1#*=}"
	    ;;
	--repetitions=*)
	    repetitions="${1#*=}"
	    ;;
	--use_new_formula=*)
	    use_new_formula="${1#*=}"
	    ;;
	--use_adam_optimizer=*)
	    use_adam_optimizer="${1#*=}"
	    ;;
	--choose_student_strategy=*)
	    choose_student_strategy="${1#*=}"
	    ;;
	--out_file_name=*)
	    out_file_name="${1#*=}"
	    ;;
	--results_dir_prefix=*)
	    # Prefix for evaluation results directory name.
	    results_dir_prefix="${1#*=}"
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

paramfile=$(create_param_file "${out_file_name}" "${subject}" "${config}" "${repetitions}" \
    "${use_new_formula}" "${use_adam_optimizer}" "${choose_student_strategy}" "${results_dir_prefix}")
