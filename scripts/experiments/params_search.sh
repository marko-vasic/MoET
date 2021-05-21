#!/bin/bash

# Script that runs experiments searching for best hyperparams.

readonly MOE_EXPERTS=2
readonly EXPERTS_DEPTHS=1
readonly CONFIG_TYPE="MOEHard"
readonly SUMMARY_FILE="summary_acrobot_experts${MOE_EXPERTS}_depth${EXPERTS_DEPTHS}_${CONFIG_TYPE}.csv"

function run_experiments() {
    for learning_rate in "0.01" "0.03" "0.1" "0.3" "1"; do
        for learning_rate_decay in "1" "0.97"; do
            for moe_max_epoch in 100 200 300; do
                for stop_count in 3 1000; do
                    (
                        cd ../../python &&  \
                            python -m viper.evaluation.main \
                                   --subject_name=acrobot \
                                   --config_type="${CONFIG_TYPE}" \
                                   --function=learn \
                                   --max_depth=0 \
                                   --experts_no="${MOE_EXPERTS}" \
                                   --experts_depths="${EXPERTS_DEPTHS}" \
                                   --out_dir_name="test/LR${learning_rate}-DEC${learning_rate_decay}-ME${moe_max_epoch}-SC${stop_count}" \
                                   --use_new_formula=True \
                                   --use_adam_optimizer=True \
                                   --choose_best_student_strategy=reward_and_mispredictions \
                                   --moe_init_learning_rate="${learning_rate}" \
                                   --moe_learning_rate_decay="${learning_rate_decay}" \
                                   --moe_max_epoch="${moe_max_epoch}" \
                                   --moe_stop_count="${stop_count}"
                    )
                    (
                        cd ../../python && \
                            python -m viper.evaluation.main \
                                   --subject_name=acrobot \
                                   --config_type="${CONFIG_TYPE}" \
                                   --function=evaluate \
                                   --max_depth=0 \
                                   --experts_no="${MOE_EXPERTS}" \
                                   --experts_depths="${EXPERTS_DEPTHS}" \
                                   --out_dir_name="test/LR${learning_rate}-DEC${learning_rate_decay}-ME${moe_max_epoch}-SC${stop_count}" \
                                   --use_new_formula=True \
                                   --use_adam_optimizer=True \
                                   --choose_best_student_strategy=reward_and_mispredictions \
                                   --moe_init_learning_rate="${learning_rate}" \
                                   --moe_learning_rate_decay="${learning_rate_decay}" \
                                   --moe_max_epoch="${moe_max_epoch}" \
                                   --moe_stop_count="${stop_count}"
                    )
                done
            done
        done
    done
}

function extract_results() {
    echo "LR,LRDecay,MaxEpoch,StopCount,Reward,Mispredictions"
    for learning_rate in "0.01" "0.03" "0.1" "0.3" "1"; do
        for learning_rate_decay in "1" "0.97"; do
            for moe_max_epoch in 100 200 300; do
                for stop_count in 3 1000; do
                    out_dir_name="../../data/experiments/acrobot/test/LR${learning_rate}-DEC${learning_rate_decay}-ME${moe_max_epoch}-SC${stop_count}/${CONFIG_TYPE}"
                    out_file="${out_dir_name}/${CONFIG_TYPE}_evaluation.tex"
                    reward=$(cat "${out_file}" | grep '_reward' | sed -r 's!\DefMacro\{.*\}\{(.*)\}!\1!g')
                    reward="${reward:1}"
                    mispredictions=$(cat "${out_file}" | grep '_mispredictions_student_playing' | sed -r 's!\DefMacro\{.*\}\{(.*)\}!\1!g')
                    mispredictions="${mispredictions:1}"
                    echo "${learning_rate},${learning_rate_decay},${moe_max_epoch},${stop_count},${reward},${mispredictions}"
                done
            done
        done
    done
}

# run_experiments
extract_results > "${SUMMARY_FILE}"
