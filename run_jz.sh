#!/bin/bash
module load pytorch-gpu/py3/2.0.1
pip install trl
python test_finetuned.py -a "IMGEP_smart_sol" -p $SCRATCH/evaluate_model/ "run_saved/imgep_smart/step_499_1/maps.json" -e 1 -c 128 -b 32
