#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=sft7b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0,1,2,3,4      
#SBATCH --output=./out/out_finetune_llama7b-%A_%a.out
#SBATCH --error=./out/out_finetune_llama7b-%A.out

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.0.1
cd $SCRATCH/evaluate_model 

python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 1 -c 12 -b 12 -m "NousResearch/Llama-2-7b-hf" -s "21" -t "train"
python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 1 -c 12 -b 12 -m "NousResearch/Llama-2-7b-hf" -s "21" -t "eval"


