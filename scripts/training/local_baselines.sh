#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"
export OUTPUT_DIR=/science_data/jiangdazhi/research/Tina/outputs/baseline
echo "START TIME: ${OUTPUT_DIR}"
# source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=6,7 # make sure all evaluation run on 2 GPUs
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: $GPU_COUNT, make sure using 2 GPUs."
echo ""

# 只保留你要测试的本地模型路径
MODEL_LIST=("/science_data/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B")

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    # Define an array of tasks to evaluate
    tasks=("math_500" "gpqa:diamond" "aime25" "amc23" "minerva" "aime24")

    for TASK in "${tasks[@]}"; do
      echo "Evaluating task: $TASK"
      lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
          --custom-tasks ./scripts/training/run_post_train_eval.py \
          --use-chat-template \
          --output-dir "${OUTPUT_DIR}/${MODEL_NAME##*/}/${TASK}"
    done
done

echo "END TIME: $(date)"
echo "DONE"