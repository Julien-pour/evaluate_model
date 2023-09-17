python test_finetuned.py -a "IMGEP_smart_sol" -p $SCRATCH/evaluate_model/ "run_saved/imgep_smart/step_499_1/maps.json" -e 1 -c 64 -b 4
# python test_finetuned.py -a "ELM_sol" -p $SCRATCH/evaluate_model/ -b "run_saved/elm/step_499_1/maps.json" -e 1
# python test_finetuned.py -a "random_gen_sol" -p $SCRATCH/evaluate_model/ -b "run_saved/random_gen/step_499_1/maps.json" -e 1
# python test_finetuned.py -a "IMGEP_random_sol" -p $SCRATCH/evaluate_model/ -b "run_saved/imgep_random/step_499_1/maps.json" -e 1
# python test_finetuned.py -a "ELM_nlp_sol" -p $SCRATCH/evaluate_model/ -b "run_saved/elm_nlp/step_499_1/maps.json" -e 1

# python test_finetuned.py -a "ELM_nlp_sol" -b "/projets/flowers/julien/evaluate_model/run_saved/elm_nlp/step_499_1/maps.json" -c "cosine" -d 0.3 -e "True"
# # python test_model_batch.py -a "ELM_nlp_sol"
# # python test_finetuned.py -a "IMGEP_random_sol" -b "/projets/flowers/julien/evaluate_model/run_saved/imgep_random/step_499_1/maps.json" -c "cosine" -d 0.3  -e "True"
# python test_model_batch.py -a "IMGEP_random_sol"
# # python test_finetuned.py -a "random_gen_sol" -b "/projets/flowers/julien/evaluate_model/run_saved/random_gen/step_499_1/maps.json" -c "cosine" -d 0.3  -e "True"
# python test_model_batch.py -a "random_gen_sol"
# python test_model_batch.py -a "IMGEP_smart_sol"
# python test_model_batch.py -a "ELM_sol"
# python test_model_batch.py 
