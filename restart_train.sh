#!/bin/bash
cd ~/homebot-wm
pkill -9 -f "train.py" 2>/dev/null
sleep 2
rm -f logs/train.log
screen -S train -X quit 2>/dev/null
sleep 1
source .venv/bin/activate
screen -dmS train bash -c "python scripts/train.py --data data/trajectories.h5 --epochs 100 2>&1 | tee logs/train.log"
echo "Training started in screen session"
sleep 20
tail -50 logs/train.log
