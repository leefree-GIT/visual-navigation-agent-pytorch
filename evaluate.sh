#!/bin/bash
checkpoints=("500000" "1000000")
len=${#checkpoints[@]}
for ((i=0;i<${len};i++))
do
    python -u eval.py -e EXPERIMENTS/ai2thor_env/param.json --csv_file EXPERIMENTS/ai2thor_env/eval${i}_train.csv --log_arg ${i} --checkpoint_path EXPERIMENTS/ai2thor_env/checkpoints/${checkpoints[$i]}.pth | tee EXPERIMENTS/ai2thor_env/eval_train${i}_back.log 
done