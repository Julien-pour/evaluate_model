print("aaaaaaa")


import argparse
parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("-z", "--base_path", type=str, help="path to git project evaluate_model",default="/media/data/flowers/evaluate_model/")
parser.add_argument("--path_model_base", type=str, help="path where hf model are saved",default="/gpfsscratch/rech/imi/uqv82bm/hf/")

# parser.add_argument("-p", "--path_dir", type=str, help="path to evaluate_model")

parser.add_argument("-p", "--arg_path_idx", type=int, help="path baseline idx  (data to use as trainset)",default=0)
parser.add_argument("-c", "--arg_bs_test", type=int, help=" bs test",default=64)
parser.add_argument("-e", "--arg_epoch", type=int, help="number epoch",default=2)

parser.add_argument("-m", "--arg_model_id", type=str, help=" model",default="deepseek-coder-1.3b-instruct")
parser.add_argument("-s", "--arg_seed", type=str, help="seed ",default="1")
parser.add_argument("-k", "--arg_k", type=int, help="k in pass@k",default=10)
parser.add_argument("-g", "--arg_gpu", type=str, help="GPU use",default="a100")
parser.add_argument("--test_base_model", type=str, help="just test base model",default="False")
parser.add_argument("--test_base_model_on_train", type=str, help="just test_base_model_on_train",default="False")
parser.add_argument(
    "--dataset",  type=str, choices=["humaneval", "mbpp"], default="humaneval"
)
# parser.add_argument("--samples",  type=str, default="samples.jsonl")
parser.add_argument("--base-only", action="store_true")
parser.add_argument("--parallel", default=None, type=int)
parser.add_argument("--i-just-wanna-run", action="store_true")
parser.add_argument("--test-details", action="store_true")
parser.add_argument("--min-time-limit", default=1, type=float)
parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
parser.add_argument("--mini", action="store_true")

args = parser.parse_args()


import copy
import os
os.environ["HUMANEVAL_OVERRIDE_PATH"] = args.base_path+ "eval_plus_data/HumanEvalPlus-v0.1.9.jsonl"
os.environ["MBPP_OVERRIDE_PATH"] = args.base_path+ "eval_plus_data/MbppPlus-v0.1.0.jsonl"

# from key import wandb_key   
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import json
from utils_eval_plus import evaluate_plus

# from peft import LoraConfig
from utils_test import return_full_prompt,pass_at_k


if args.arg_gpu == "v100":
    type_use = torch.float16
    bf16=False
    fp16=True
else:
    type_use = torch.bfloat16
    bf16=True
    fp16=False

# /!\ set that
# limited_trainset=True # data generated by expe with 3 first example from trainset or full trainset



if args.arg_epoch:
    num_train_epochs= args.arg_epoch

seed = str(args.arg_seed)



# test model

os.environ["WANDB_DISABLED"] = "True"

os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['WANDB_MODE'] = "offline"
os.environ["WANDB_PROJECT"] = "codegpt finetuned"
# os.environ['WANDB_API_KEY'] = wandb_key
os.environ['WANDB_CACHE_DIR'] = args.base_path+"wandb_cache/"


# os.environ['HF_DATASETS_CACHE'] = args.base_path+"hf/datasets"
# os.environ['TRANSFORMERS_CACHE'] = args.base_path+"hf/models"

os.environ['TOKENIZERS_PARALLELISM'] = "True"



path_train_idx = args.arg_path_idx

# define all path

if seed[0]=="1" or seed[0]=="2" or seed[0]=="3":
    path_rd_gen= "run_saved/maps_"+seed[0]+"_rd_gen.json"
    path_elm="run_saved/maps_"+seed[0]+"_elm.json"
    path_elm_nlp="run_saved/maps_"+seed[0]+"_elm_NLP.json"
    path_img_rd="run_saved/maps_"+seed[0]+"_imgep_random.json"
    path_imgp_smart="run_saved/maps_"+seed[0]+"_imgep_smart.json"



list_name=["rd_gen","elm","elm_NLP","imgep_random","imgep_smart"]

list_all_path=[path_rd_gen, path_elm,path_elm_nlp,path_img_rd,path_imgp_smart]
path_train = args.base_path 

curr_train_path = list_all_path[path_train_idx]
path_train += curr_train_path
name = list_name[path_train_idx]

path_save=args.base_path+"save_results/"+name+".json"
print("\n=============\n ")

print("path train ",path_train)
print("\n=============\n ")

print(path_train)

# hf way to load json dataset



model_id =   args.arg_model_id

hf_dir=args.path_model_base

path_load_model=hf_dir+model_id
print("path_load_model",path_load_model)

if args.dataset == "humaneval":
    name_json_save_all = args.base_path+"save_results/passk_human_eval.json"#.split("/")[1]
elif args.dataset == "mbpp":
    name_json_save_all = args.base_path+"save_results/passk_mbpp.json"#.split("/")[1]
else:
    raise ValueError("Unknown dataset")
run_name = name+"e_"+str(num_train_epochs)+"_"+str(seed)


args.samples = "res_eval_plus/"+run_name+"samples.jsonl"

if args.test_base_model_on_train=="True":
    run_name = model_id+"_traine_"+str(num_train_epochs)+"_"+str(seed)

if not os.path.exists(name_json_save_all):
    # Create a new JSON file with some sample data
    sample_data = {}
    with open(name_json_save_all, 'w') as file:
        json.dump(sample_data, file, indent=4)


name_json = args.base_path+"save_results/"+name+model_id#.split("/")[1]
name_json_sol = args.base_path+"save_sol/"+name+model_id#.split("/")[1]


output_dir = hf_dir+model_id+run_name
# testing
if args.test_base_model=="True":
    output_dir = hf_dir+model_id
    run_name = model_id+"_base"
tokenizer = AutoTokenizer.from_pretrained(output_dir,local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=type_use,

    # quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True
)
tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token
model.eval()
model.config.use_cache = True
# model=torch.compile(model)



# testset= preprocessing_P3_no_test(split="test",n_token_max=1024,path=args.base_path,tokenizer=tokenizer)


curr_idx=0
correct_puzz=0

num_return_sequences=args.arg_k #n_try
list_all_passk=[[] for i in range(num_return_sequences)]
list_passk=[]

list_puzzle=[]
list_all_puzzle=[]


if args.dataset == "humaneval":

    from evalplus.data import get_human_eval_plus, write_jsonl
    dic_puzzles = get_human_eval_plus()
    print("=======================")
    print("/!\ testing on humaneval /!\ ")
    print("=======================")
elif args.dataset == "mbpp":
    from evalplus.data import get_mbpp_plus, write_jsonl
    dic_puzzles = get_mbpp_plus()
    print("=======================")
    print("/!\ testing on mbpp /!\ ")
    print("=======================")
else:
    raise ValueError("Unknown dataset")

list_keys= list(dic_puzzles.keys())
list_task_id = [dic_puzzles[key]["task_id"] for key in list(dic_puzzles.keys())]
list_testset = [dic_puzzles[key]["prompt"] for key in list(dic_puzzles.keys())]

# samples = [
#     dict(task_id=task_id, solution=problem["prompt"])
#     for task_id, problem in get_mbpp_plus().items()
# ]
# write_jsonl("samples.jsonl", samples)
samples = []
list_puzzle_correct=[]

bs = args.arg_bs_test
with torch.inference_mode():
    
    for idx in tqdm(range(curr_idx,len(list_testset),bs)): #len(dataset["test"])
        # idx=0
        print(f"\n\n============ idx {idx} ==================\n")
        flag=True
        attempt=0
        list_puzzle_idx=[]
        list_prompt=[]
        list_prompt_=[]
        subset_test = list_testset[idx:idx+bs]
        for idx_puz in range(len(subset_test)):
            prompt_ = subset_test[idx_puz]
            list_prompt_.append(prompt_)
            prompt = return_full_prompt(model_id=model_id,pb=prompt_,mode=args.dataset) # todo
            list_prompt.append(prompt)
        inputs = tokenizer(list_prompt, return_tensors="pt",padding=True).to("cuda")
        # for idx_inp in range(len(inputs)):
        len_prompt = inputs["input_ids"].shape[1]
        list_puzzle_gen=[[] for _ in range(len(list_prompt))]
        for idx_gen in range(num_return_sequences):
            outputs = model.generate(**inputs,max_new_tokens=512,do_sample=True, temperature=0.7)
            generated_texts = tokenizer.batch_decode(outputs[:,len_prompt:], skip_special_tokens=True)
            for idx_out_gen in range(len(outputs)):
                list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])

        list_generated_text = copy.deepcopy(list_puzzle_gen)

        for i in range(len(list_puzzle_gen)): # along the bs
            dic_save={}
            for j in range(len(list_puzzle_gen[i])): # along the n_try (pass_k ->k)
                prompt_ =list_prompt_[i]
                try:
                    #check if "```" is in list_puzzle_gen[i][j]
                    list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```python","```")
                    list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```Python","```")

                    if "```" in list_puzzle_gen[i][j]:
                        extract_sol=list_puzzle_gen[i][j].split("```")[1]
                    else:
                        extract_sol=list_puzzle_gen[i][j]
                except:
                    print("error extract sol")
                    print(list_puzzle_gen[i][j])
                test_sol= extract_sol 
                list_puzzle_gen[i][j] = test_sol

                list_puzzle.append(test_sol)
                samples.append(dict(task_id=list_task_id[idx + i], solution=test_sol))

            

    write_jsonl(args.samples, samples)
    pass_at_k, pass_at_k_plus = evaluate_plus(args)
    dic_pass={"pass_at_k":pass_at_k,"pass_at_k_plus":pass_at_k_plus}
    with open(name_json_save_all, "r") as outfile:
        json_content=json.load(outfile)
    json_content[run_name]= dic_pass
    with open(name_json_save_all, "w") as outfile:
        json.dump(json_content,outfile,indent=4)


