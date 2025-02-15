export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1


export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1


if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=localhost
fi
if [ -z "$MASTER_PORT" ]; then
    export MASTER_PORT=8000
fi
if [ -z "$RANK" ]; then
    export RANK=0
fi
if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=1
fi

GPU_IDS="0,1,2,3,4,5,6,7"
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"

accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_machines $WORLD_SIZE \
    --num_processes $((WORLD_SIZE * 8)) \
    --gpu_ids $GPU_IDS \
    training/cogvideox_text_to_video_sft.py --config configs/sft_config.py