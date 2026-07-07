#!/bin/bash
cd ~/homebot-wm
source .venv/bin/activate

echo "Waiting for block decoder training to complete..."
while true; do
    if ls checkpoints/block_decoder.pt 1> /dev/null 2>&1; then
        echo "Block decoder checkpoint found! Starting evaluation..."
        break
    fi
    sleep 60
done

echo "Running optimized closed-loop evaluation..."
MUJOCO_GL=egl python scripts/evaluate_optimized.py \
    --checkpoint checkpoints/world_model_ep100.pt \
    --block_decoder checkpoints/block_decoder.pt \
    --episodes 5 \
    --max_steps 100 \
    --diffusion_steps 10 \
    2>&1 | tee logs/evaluate_optimized.log

echo "Evaluation complete!"
