#!/bin/bash

USE_ADAM="True"
EXPERIMENT_NAME="journal"
CONFIGS="all"
REPETITIONS=5

function run() {
    local subject="${1}"; shift
    local cluster="${1}"; shift
    local A="${1}"; shift
    param_file_name="${EXPERIMENT_NAME}_paramlist_${subject}"

    if [ "${subject}" = "pong" ]; then
	# Pong is memory consuming, thus assigning less tasks per
	# node.
	N="10"
	n="30"
	t="28:00:00"
    elif [ "${subject}" = "mountaincar" ]; then
	# Previously finished in 2h 23min on Hikari.
	N="7"
	n="196"
	t="10:00:00"
    else
	N="7"
	n="196"
	t="15:00:00"
    fi

    ./create_configurations.sh \
    	--subject="${subject}" \
    	--results_dir_prefix="${EXPERIMENT_NAME}" \
    	--repetitions="${REPETITIONS}" \
    	--use_adam_optimizer="${USE_ADAM}" \
    	--config="${CONFIGS}" \
    	--out_file_name="${param_file_name}"
    
    ./schedule_jobs.sh \
    	--cluster="${cluster}" \
    	--subject="${subject}" \
    	--param_file=paramfiles/${param_file_name} \
    	--N="${N}" \
    	--n="${n}" \
    	--t="${t}" \
    	--A="${A}"
}


while [ $# -gt 0 ]; do
    case "$1" in
	--subject=*)
	    subject="${1#*=}"
	    ;;
	--cluster=*)
	    cluster="${1#*=}"
	    ;;
	--A=*)
	    A="${1#*=}"
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
    echo "You must specify subject."
    exit 1
fi

if [ -z "${cluster}" ]; then
    echo "You must specify cluster."
    exit 1
fi

if [ -z "${A}" ]; then
    echo "You must specify TACC project."
    exit 1
fi

run "${subject}" "${cluster}" "${A}"
