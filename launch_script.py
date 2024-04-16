import os
import argparse
import subprocess
from datetime import datetime
# script running over epochs, LRs, ratios data
script="""#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name=codellm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20

#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-44
#SBATCH --output=./out/out_finetune_deep-%A_%a.out
#SBATCH --error=./out/out_finetune_deep-%A_%a.out

module load python/3.11.5
conda deactivate
module purge
module load cuda/12.1.0
conda activate exllama

cd $WORK/evaluate_model  

epochs=(1 2 3)
lrs=(1e-5 5e-5 1e-6)
ratios=(0.2 0.4 0.6 0.8 1.0) # data

# Calculate indices
num_epochs=${#epochs[@]}
num_lrs=${#lrs[@]}
num_ratios=${#ratios[@]}

# Map the task array index to parameter indices
index=$SLURM_ARRAY_TASK_ID
ratio_index=$((index % num_ratios))
lr_index=$(((index / num_ratios) % num_lrs))
epoch_index=$(((index / (num_lrs * num_ratios)) % num_epochs))

# Extract the actual parameters based on indices
epoch=${epochs[$epoch_index]}
lr=${lrs[$lr_index]}
ratio=${ratios[$ratio_index]}

echo "Running job with epoch=$epoch, lr=$lr, ratio=$ratio"


python test_finetuned_quality_grid_search.py -z $WORK/evaluate_model/ --path_archive "run_saved/{name_archive}" -e $epoch -c 16 -b 2 -s "1" -t "train" --arg_gpu "v100" -a 2 --random {random} --lr=$lr --ratio=$ratio --description {description}
python test_finetuned_quality_grid_search.py -z $WORK/evaluate_model/ --path_archive "run_saved/{name_archive}" -e $epoch -c 16 -b 2 -s "1" -t "eval" --arg_gpu "v100" -a 2 --random {random} --lr=$lr --ratio=$ratio --description {description}
"""
list_archive=["/home/flowers/work/OpenELM/logs/archives/rd_gen_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/archives/elm_quality_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/archives/elm_nlp_quality_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/elm_nlp_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/aces_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/aces_quality_seed-1.json",
           "/gpfswork/rech/imi/uqv82bm/evaluate_model/aces_smart_quality_seed-1.json",
           ]

for random_mode in ["False","True"]:
    for archive in list_archive:
        for description_mode in ["False","True"]: 
            script_formated = script.format(name_archive=archive, random=random_mode,description=description_mode)