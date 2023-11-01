print("aaaaaaa")
import argparse
parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("-z", "--base_path", type=str, help="path to git project evaluate_model",default="/media/data/flowers/evaluate_model/")

# parser.add_argument("-p", "--path_dir", type=str, help="path to evaluate_model")

parser.add_argument("-p", "--arg_path_idx", type=int, help="path baseline maps.json (trainset)")
parser.add_argument("-e", "--arg_epoch", type=int, help="number epoch",default=2)
parser.add_argument("-n", "--arg_n_train", type=int, help="number of trainset")
parser.add_argument("-b", "--arg_bs", type=int, help=" bs",default=4)
parser.add_argument("-c", "--arg_bs_test", type=int, help=" bs test",default=64)

parser.add_argument("-m", "--arg_model_id", type=str, help=" model")
parser.add_argument("-s", "--arg_seed", type=str, help="seed ",default="00")
parser.add_argument("-r", "--arg_rm_ex", type=bool, help=" remove example in docstring",default=True)
parser.add_argument("-t", "--arg_mode", type=str, help=" model",default="train_eval")


args = parser.parse_args()
if args.arg_mode=="train_eval":
    train_model=True
    eval_model=True
if args.arg_mode=="train":
    train_model=True
    eval_model=False 
if args.arg_mode=="eval":
    train_model=False
    eval_model=True 

if args.arg_epoch:
    num_train_epochs= args.arg_epoch
rm_ex=args.arg_rm_ex

seed = str(args.arg_seed)



# test model
from utils_test import remove_example_line,just_remove_example_in_docstring
import os
os.environ["WANDB_DISABLED"] = "True"

os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['WANDB_MODE'] = "offline"
os.environ["WANDB_PROJECT"] = "codegpt finetuned"
from key import wandb_key   
os.environ['WANDB_API_KEY'] = wandb_key
os.environ['WANDB_CACHE_DIR'] = args.base_path+"wandb_cache/"


os.environ['HF_DATASETS_CACHE'] = args.base_path+"hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = args.base_path+"hf/models"

os.environ['TOKENIZERS_PARALLELISM'] = "True"
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,LlamaTokenizer

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset,concatenate_datasets
from tqdm import tqdm
import json

# from peft import LoraConfig
import numpy as np
from utils_test import pass_at_k,prompt_solve_puzzle,judge_parallel,preprocessing_P3_no_test

import gc


path_train_idx = args.arg_path_idx

# define all path
if seed[0]=="2":
    path_rd_gen= "run_saved/maps_2_rd_gen.json"
    path_elm="run_saved/maps_2_elm.json"
    path_elm_nlp="run_saved/maps_2_elm_NLP.json"
    path_img_rd="run_saved/maps_2_imgep_random.json"
    path_imgp_smart="run_saved/maps_2_imgep_smart.json"
if seed[0]=="1":
    path_rd_gen= "run_saved/maps_1_rd_gen.json"
    path_elm="run_saved/maps_1_elm.json"
    path_elm_nlp="run_saved/maps_1_elm_NLP.json"
    path_img_rd="run_saved/maps_1_imgep_random.json"
    path_imgp_smart="run_saved/maps_1_imgep_smart.json"
if seed[0]=="3":
    path_rd_gen= "run_saved/maps_3_rd_gen.json"
    path_elm="run_saved/maps_3_elm.json"
    path_elm_nlp="run_saved/maps_3_elm_NLP.json"
    path_img_rd="run_saved/maps_3_imgep_random.json"
    path_imgp_smart="run_saved/maps_3_imgep_smart.json"

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

run_name_wandb=path_train.split("run_saved/")[1].split("/")[0]
run_name_wandb = run_name_wandb.replace("/","_")
dataset = load_dataset("json", data_files=path_train, split="train")

path_P3_trainset = args.base_path +"preprocess_p3_emb.json"

dataset_r = load_dataset("json", data_files=path_P3_trainset, split="train")

# dataset_r = dataset_r.train_test_split(test_size=0.005)
cat_datasets=concatenate_datasets([dataset,dataset_r])
cat_datasets=cat_datasets.shuffle(seed=42)

# model_id="openlm-research/open_llama_3b_v2"#"codellama/CodeLlama-7b-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
if args.arg_model_id:
    model_id =   args.arg_model_id


name_json=args.base_path+"save_results/"+name+model_id.split("/")[1]
name_json_sol=args.base_path+"save_sol/"+name+model_id.split("/")[1]

if train_model:
    tokenizer = LlamaTokenizer.from_pretrained(model_id,local_files_only=True)

    tokenizer.padding_side='right'
    tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    #training
    def formatting_prompts_func(example,prompt_solve_puzzle=prompt_solve_puzzle):
        output_texts = []
        # print(len(example['program_str']))
        for i in range(len(example['program_str'])):
            if rm_ex:
                puzzle= just_remove_example_in_docstring(example['program_str'][i])
            else:
                puzzle= example['program_str'][i]
            try:
                prompt_f=puzzle.split("def g(")[0]
                prompt_g= "def g(" + puzzle.split("def g(")[1]
                full_prompt = prompt_solve_puzzle.format(pb=prompt_f,g_firstline=prompt_g)
                output_texts.append(full_prompt)
            except:
                print("error in formatting_prompts_func idx",i)
                print(example['program_str'][i])
                print("======================")
                print(puzzle)
        return output_texts

    lr_scheduler_type= "cosine"

    warmup_ratio=0.1


    # response_template = 'Problem 1:'#"Solution 1:"

    # if args.arg_sol:
    # response_template= "Solution 1:" # for llama
    response_template= "Solution 1:"
    # list_tok_response_template=tokenizer(response_template)["input_ids"][1:]

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)




    run_name_wandb += "Epoch"+str(num_train_epochs)
    run_name_wandb+= model_id.split("/")[1]

    learning_rate=1e-5

    training_arguments=TrainingArguments(
        per_device_train_batch_size=args.arg_bs,
        # per_device_eval_batch_size=4,
        # evaluation_strategy="steps",
        gradient_accumulation_steps=1,
        run_name= run_name_wandb,
        # warmup_steps=2,
        save_strategy="no",
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        # max_steps=500,
        num_train_epochs=num_train_epochs,
        # weight_decay=0.001,
        learning_rate=learning_rate,
        bf16=True,
        # bf16_full_eval=True,
        gradient_checkpointing=False,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch",#"paged_adamw_32bit",
        max_grad_norm=0.3,
        # group_by_length=True,
        # do_eval=True,
        # eval_steps=10,
        # torch_compile=True
        
    )

    config = {"lr":learning_rate, "batch_size": args.arg_bs,"warmup_ratio":warmup_ratio,"model_name":"model_id", "epoch":args.arg_epoch}
    # config.update({"architecture": "", "depth": 34})
    # wandb.init(#name=name_wb,
    #            config=config,project="finetune llama")


    trainer = SFTTrainer(
        model,#"EleutherAI/gpt-neo-125m",
        tokenizer=tokenizer,
        train_dataset=cat_datasets,
        # eval_dataset=dataset_r["test"],
        # dataset_text_field="program_str",

        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=1100,
        args=training_arguments

    )
    trainer.train()

    output_dir = model_id+name+"e_"+str(num_train_epochs)+"_"+str(seed) #args.base_path+"hf/datasets"+name # where to save model
    trainer.save_model(output_dir)

if eval_model:  # OOD
    output_dir = model_id+name+"e_"+str(num_train_epochs)+"_"+str(seed)
    # del model
    # del tokenizer
    # gc.collect()
    # torch.cuda.empty_cache()
    # for obj in gc.get_objects():
    #     if torch.is_tensor(obj):
    #         obj.cpu()            
    # gc.collect()
    # torch.cuda.empty_cache()


    # testing
    # tokenizer = AutoTokenizer.from_pretrained(output_dir)#model_id)
    # if model_id =="openlm-research/open_llama_3b_v2":
    tokenizer = LlamaTokenizer.from_pretrained(output_dir,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,

        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    model=torch.compile(model)


    torch._dynamo.config.suppress_errors = True

    testset= preprocessing_P3_no_test(split="test",n_token_max=1024,path=args.base_path,tokenizer=tokenizer)

        
    list_trainset= [[x["program_str"],x["g_firstline"]] for x in testset]
    list_puzzle_correct=[]
    correct_puzz=0
    curr_idx=0
    num_return_sequences=10 #n_try
    list_passk=[]
    list_passk_1=[]
    list_passk_2=[]
    list_passk_3=[]
    list_passk_4=[]
    list_passk_5=[]
    list_passk_6=[]
    list_passk_7=[]
    list_passk_8=[]
    list_passk_9=[]
    list_passk_10=[]
    list_puzzle=[]
    list_all_puzzle=[]

    bs = args.arg_bs_test
    with torch.no_grad():
        
        for idx in tqdm(range(curr_idx,len(list_trainset),bs)): #len(dataset["test"])
            curr_idx=idx
            # idx=0
            print(f"\n\n============ idx {idx} ==================\n")
            flag=True
            attempt=0
            list_puzzle_idx=[]
            list_prompt=[]
            list_prompt_f=[]
            list_prompt_g_firstline=[]
            subset_train = list_trainset[idx:idx+bs]
            for (puzzle,g_firstline) in subset_train:
                
                prompt_f = puzzle.split("def g(")[0]
                list_prompt_g_firstline.append(g_firstline)
                list_prompt_f.append(prompt_f)
                prompt = prompt_solve_puzzle.format(pb=prompt_f,g_firstline=g_firstline)
                list_prompt.append(prompt)
            inputs = tokenizer(list_prompt, return_tensors="pt",padding=True).to("cuda")
            # for idx_inp in range(len(inputs)):
            len_prompt = inputs["input_ids"].shape[1]
            list_puzzle_gen=[[] for _ in range(len(list_prompt))]
            for idx_gen in range(num_return_sequences):
                outputs = model.generate(**inputs,max_new_tokens=512,do_sample=True, temperature=0.8)
                generated_texts = tokenizer.batch_decode(outputs[:,len_prompt:], skip_special_tokens=True)
                for idx_out_gen in range(len(outputs)):
                    list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])


            for i in range(len(list_puzzle_gen)): # along the bs
                dic_save={}
                list_raw_puzzle = []
                list_proc_puzzle =[]
                for j in range(len(list_puzzle_gen[i])):
                    prompt_f =list_prompt_f[i]
                    g_firstline = list_prompt_g_firstline[i]

                    extract_g=list_puzzle_gen[i][j].split("```")[0].split("assert")[0]
                    extract_g = g_firstline + extract_g+"\nassert f(g()) == True\n"
                    test_fg= prompt_f+extract_g 
                    list_puzzle_gen[i][j] = test_fg
                    list_puzzle.append(test_fg)
                    list_proc_puzzle.append(test_fg)
                    list_raw_puzzle.append(prompt_f+list_puzzle_gen[i][j])
                dic_save["raw_puzzle"]=list_raw_puzzle
                dic_save["process_puzzle"]=list_proc_puzzle
                
                    # if j<1:
                    #     print("\n-------------------\n")
                    #     print(test_fg)
                    
                
                list_valid_puzzles = judge_parallel(list_puzzle_gen[i])
                dic_save["list_valid"]=list_valid_puzzles                 
                list_all_puzzle.append(dic_save)    

                cor_puz= np.sum(list_valid_puzzles)

                n_sample, n_correct=num_return_sequences,cor_puz
                pass_k = pass_at_k(n_sample, n_correct, k=num_return_sequences)
                list_passk.append(pass_k)
                list_passk_1.append(pass_at_k(n_sample, n_correct, k=1))
                list_passk_2.append(pass_at_k(n_sample, n_correct, k=2))
                list_passk_3.append(pass_at_k(n_sample, n_correct, k=3))
                list_passk_4.append(pass_at_k(n_sample, n_correct, k=4))
                list_passk_5.append(pass_at_k(n_sample, n_correct, k=5))
                list_passk_6.append(pass_at_k(n_sample, n_correct, k=6))
                list_passk_7.append(pass_at_k(n_sample, n_correct, k=7))
                list_passk_8.append(pass_at_k(n_sample, n_correct, k=8))
                list_passk_9.append(pass_at_k(n_sample, n_correct, k=9))
                list_passk_10.append(pass_at_k(n_sample, n_correct, k=10))
            print(f"correct puzzles: {int(np.sum(list_passk))}/{len(list_passk)}")
            # with open(name_json+".json", "w") as outfile:
            #     json.dump(list_passk,outfile)

        print(f"pass 1: {np.sum(list_passk_1)}/{len(list_passk)}")
        print(f"pass 3: {np.sum(list_passk_3)}/{len(list_passk)}")
        print(f"pass 5: {np.sum(list_passk_5)}/{len(list_passk)}")
        print(f"pass 7: {np.sum(list_passk_7)}/{len(list_passk)}")
        print(f"pass 10: {np.sum(list_passk_10)}/{len(list_passk)}")
        dic_passk= {"pass_1":float(np.sum(list_passk_1))}
        dic_passk["pass_2"]= float(np.sum(list_passk_2))
        dic_passk["pass_3"]= float(np.sum(list_passk_3))
        dic_passk["pass_4"]= float(np.sum(list_passk_4))
        dic_passk["pass_5"]= float(np.sum(list_passk_5))
        dic_passk["pass_6"]= float(np.sum(list_passk_6))
        dic_passk["pass_7"]= float(np.sum(list_passk_7))
        dic_passk["pass_8"]= float(np.sum(list_passk_8))
        dic_passk["pass_9"]= float(np.sum(list_passk_9))
        dic_passk["pass_10"]= float(np.sum(list_passk_10))

        json_content=[dic_passk]
        with open(name_json+"_e"+str(num_train_epochs)+"_seed_"+seed+".json", "w") as outfile:
            json.dump(json_content,outfile,indent=4)
        with open(name_json_sol+"_e"+str(num_train_epochs)+"_seed_"+seed+".json", "w") as outfile:
            json.dump(list_all_puzzle,outfile,indent=4)
        

