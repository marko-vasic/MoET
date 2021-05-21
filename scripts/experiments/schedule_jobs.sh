# # Testing on development node
# sbatch -N 1 -n 48 -t 00:05:00 -p development -J test launcher.slurm files/paramlist_acrobot

function schedule_runs() {
    local cluster="${1}"; shift
    local subject="${1}"; shift
    local param_file="${1}"; shift
    local N="${1}"; shift
    local n="${1}"; shift
    local t="${1}"; shift
    # Project
    local A="${1}"; shift

    local launcher_script=""
    if [ "${cluster}" = "ls5" ]; then
	# launcher_script="launcher.slurm"
	# launcher_script="pylauncher_jobscript"
	launcher_script="pylauncher_jobscript_new"
    elif [ "${cluster}" = "stampede2" ]; then
	launcher_script="launcher-stampede2.slurm"
    elif [ "${cluster}" = "hikari" ]; then
	launcher_script="launcher-hikari.slurm"
    else
	echo "Do not know how to schedule for this configuration!"
	exit 1
    fi

    sbatch \
	-A "${A}" \
	-N "${N}" \
	-n "${n}" \
	-t "${t}" \
	-p normal \
	-J "${subject}" \
	"${launcher_script}" "${param_file}"
}

N=1
n=40
t="10:00:00"

while [ $# -gt 0 ]; do
    case "$1" in
	--cluster=*)
	    cluster="${1#*=}"
	    ;;
	--subject=*)
	    subject="${1#*=}"
	    ;;
	--param_file=*)
	    param_file="${1#*=}"
	    ;;
	--N=*)
	    N="${1#*=}"
	    ;;
	--n=*)
	    n="${1#*=}"
	    ;;
	--t=*)
	    t="${1#*=}"
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

if [ -z "${cluster}" ]; then
    echo "You must specify cluster."
    exit 1
fi

if [ -z "${A}" ]; then
    echo "You must specify TACC project."
    exit 1
fi

if [ -z "${subject}" ]; then
    echo "You must specify subject."
    exit 1
fi

if [ -z "${param_file}" ]; then
    echo "You must specify file with run configurations."
    exit 1
fi

schedule_runs "${cluster}" "${subject}" "${param_file}" "${N}" "${n}" "${t}" "${A}"
