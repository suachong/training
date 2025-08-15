# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

docker run -it --rm \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    --volume=/data/mlperf_llama31_8b/data:/data \
    --volume=/data/mlperf_llama31_8b/model:/model \
    --volume=/data2/mlperf_llama31_8b/outputs:/outputs \
    --volume $(pwd):/workspace/code/ \
    --name smal-llm-training-`whoami` rocm/mlperf:llama31_8b_training_5.1_gfx942_v2