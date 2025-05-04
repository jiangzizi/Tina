lighteval vllm \
    "pretrained=/science_data/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16" \
    "leaderboard|truthfulqa:mc|0|0"