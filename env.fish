# source this file (. env.fish N)

set -l N $argv[1]


if not [ "$N" -eq "$N" ] 2>/dev/null
    echo "usage: . env.fish N"
    echo " where N is a valid CUDA device"
    exit 2
end


echo

echo "activate conda environment"
conda activate irt2m

echo "setting environment variables:"

echo "  CUDA_VISIBLE_DEVICES"
set -x CUDA_VISIBLE_DEVICES $N

echo "  IRT2M_LOG_FILE"
set -x IRT2M_LOG_FILE data/irt2m.$N.log

echo "  PYTHONBREAKPOINT"
set -x PYTHONBREAKPOINT ipdb.set_trace

echo
