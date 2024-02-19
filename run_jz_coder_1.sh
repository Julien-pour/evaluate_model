#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=sft3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0,2,4
#SBATCH --output=./out/out_finetune_llama3b-%A_%a.out
#SBATCH --error=./out/out_finetune_llama3b-%A_%a.out
module load python/3.11.5
conda deactivate
module purge
module load cuda/12.1.0
conda activate exllama

cd $SCRATCH/evaluate_model 

python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 2 -c 16 -b 8 -s "1" -t "train" 
python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 2 -c 16 -b 8 -s "1" -t "eval" 

python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 3 -c 16 -b 8 -s "1" -t "train" 
python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 3 -c 16 -b 8 -s "1" -t "eval" 

python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 1 -c 16 -b 8 -s "1" -t "train"
python test_finetuned_rework.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 1 -c 16 -b 8 -s "1" -t "eval" 

# python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 2 -c 32 -b 12 -m "openlm-research/open_llama_3b_v2" -s "11" -t "train"
# python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 2 -c 32 -b 12 -m "openlm-research/open_llama_3b_v2" -s "11" -t "eval"

# python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 1 -c 32 -b 12 -m "openlm-research/open_llama_3b_v2" -s "22" -r False
# python test_finetuned.py -z $SCRATCH/evaluate_model/ -p ${SLURM_ARRAY_TASK_ID} -e 2 -c 32 -b 12 -m "openlm-research/open_llama_3b_v2" -s "22" -r False