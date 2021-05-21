#!/bin/bash

for use_new_formula in "True" "False"; do
    for choose_student_strategy in "reward_and_mispredictions_harmonic" "reward_and_mispredictions"; do
	out_file_name="paramlist_acrobot_${use_new_formula}_${choose_student_strategy}"
	out_file="files/${out_file_name}"
	rm -f "${out_file}"

	./create_configurations.sh \
	    --subject=acrobot \
	    --use_new_formula="${use_new_formula}" \
	    --choose_student_strategy="${choose_student_strategy}" \
	    --out_file_name="${out_file_name}" \
	    --results_dir_prefix="test_strategy_and_formula" \
	    --repetitions=5

	./schedule_jobs.sh \
	    --cluster=hikari \
	    --subject=acrobot \
	    --param_file="${out_file}"
    done
done
