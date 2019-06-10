#!/bin/bash
checkpoints=("200000" "400000" "500000" "600000" "700000" "800000" "900000" "1000000" "1100000" "1200000" "1300001" "1500000" "1600000" "2000005")
len=${#checkpoints[@]}
for ((i=0;i<${len};i++))
do
    python -u eval.py -e EXPERIMENTS/all_env_keras_features_deepbox/param.json --csv_file EXPERIMENTS/all_env_keras_features_deepbox/eval${i}_train.csv --log_arg ${i} --checkpoint_path EXPERIMENTS/all_env_keras_features_deepbox/checkpoints/${checkpoints[$i]}.pth | tee EXPERIMENTS/all_env_keras_features_deepbox/eval_train${i}_back.log 
done