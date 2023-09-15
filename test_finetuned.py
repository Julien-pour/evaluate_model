
import argparse

parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("-a", "--arg_name", type=str, help="name baseline")
parser.add_argument("-p", "--path_dir", type=str, help="path to evaluate_model")

parser.add_argument("-b", "--arg_path", type=str, help="path baseline")
parser.add_argument("-c", "--arg_lr_sched", type=str, help="schedule lr")
parser.add_argument("-d", "--arg_warm", type=float, help="warmup")
parser.add_argument("-e", "--arg_epoch", type=int, help="number epoch")
parser.add_argument("-s", "--arg_step", type=int, help="number of step")
parser.add_argument("-q", "--arg_bs", type=int, help=" bs")



args = parser.parse_args()





# test model
from utils_test import remove_example_line
import os

os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"

os.environ['TOKENIZERS_PARALLELISM'] = "True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,LlamaTokenizer
from utils_test import pass_at_k,prompt_solve_puzzle

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset,concatenate_datasets
from tqdm import tqdm
import json

# from peft import LoraConfig
import numpy as np
from utils_test import pass_at_k,prompt_solve_puzzle,test_puzzle,judge_parallel,preprocessing_P3_no_test

import gc

name="IMGEP_smart"
if args.arg_name:
    name=args.arg_name
    
path_save="/projets/flowers/julien/ELM/"+name+".json"
name_json=name+"_llama_passK"

# path_train= "/projets/flowers/julien/evaluate_model/run_saved/imgep_smart/step_499_1/maps.json"#"/projets/flowers/julien/evaluate_model/run_saved/elm/step_499_1/maps.json" # put path here
if args.arg_path:
    path_train = args.arg_path

run_name_wandb=path_train.split("run_saved/")[1].split("/step")[0]

dataset = load_dataset("json", data_files=path_train, split="train")
path_real_trainset = "/projets/flowers/julien/evaluate_model/preprocess_p3_emb.json"
dataset_r = load_dataset("json", data_files=path_real_trainset, split="train")

dataset_r = dataset_r.train_test_split(test_size=0.005)
cat_datasets=concatenate_datasets([dataset,dataset_r["train"]])
# model_id="openlm-research/open_llama_3b_v2"#"codellama/CodeLlama-7b-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
model_id = "bigcode/tiny_starcoder_py"

tokenizer = AutoTokenizer.from_pretrained(model_id)


# tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,padding=True)
tokenizer.padding_side='right'
tokenizer.pad_token = tokenizer.eos_token

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="bfloat16"#torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,

    # quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
# peft_config = LoraConfig(
#     r=32,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules=["qkv_proj"],
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model.config.use_cache = False
# model=torch.compile(model)
# torch._dynamo.config.suppress_errors = True
# g_firstline = "def g("

#training


def formatting_prompts_func(example,prompt_solve_puzzle=prompt_solve_puzzle):
    output_texts = []
    # print(len(example['program_str']))
    for i in range(len(example['program_str'])):
        puzzle= remove_example_line(example['program_str'][i])
        prompt_f=puzzle.split("def g(")[0]
        prompt_g= "def g(" + puzzle.split("def g(")[1]
        full_prompt = prompt_solve_puzzle.format(pb=prompt_f,g_firstline=prompt_g)
        output_texts.append(full_prompt)
    return output_texts

lr_scheduler_type= "cosine"
if args.arg_lr_sched:
    lr_scheduler_type= args.arg_lr_sched
warmup_ratio=0.2


# response_template = 'Problem 1:'#"Solution 1:"

# if args.arg_sol:
# response_template= "Solution 1:" # for llama
response_template= " Solution 1:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)


num_train_epochs=5
if args.arg_epoch:
    num_train_epochs= args.arg_epoch
run_name_wandb += "Epoch"+str(num_train_epochs)
run_name_wandb+="starcode116m"

training_arguments=TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    gradient_accumulation_steps=4,
    run_name= run_name_wandb,
    # warmup_steps=2,
    save_strategy="no",
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    # max_steps=500,
    num_train_epochs=num_train_epochs,
    # weight_decay=0.001,
    learning_rate=3e-5,
    bf16=True,
    # bf16_full_eval=True,
    gradient_checkpointing=False,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    # group_by_length=True,
    do_eval=True,
    eval_steps=10,
    # torch_compile=True
    
)

trainer = SFTTrainer(
    model,#"EleutherAI/gpt-neo-125m",
    train_dataset=cat_datasets,
    eval_dataset=dataset_r["test"],
    # dataset_text_field="program_str",

    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=1200,
    args=training_arguments

)
trainer.train()

output_dir = "/projets/flowers/julien/models/"+name
trainer.save_model(output_dir)
if True:  # OOD

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # testing
    tokenizer = AutoTokenizer.from_pretrained(output_dir)#model_id)
    # tokenizer = LlamaTokenizer.from_pretrained(output_dir,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,

        # quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    model=torch.compile(model)


    torch._dynamo.config.suppress_errors = True

    testset= preprocessing_P3_no_test(split="test",n_token_max=1024)

        
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

    bs=64
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
                for j in range(len(list_puzzle_gen[i])):
                    prompt_f =list_prompt_f[i]
                    g_firstline = list_prompt_g_firstline[i]

                    extract_g=list_puzzle_gen[i][j].split("```")[0].split("assert")[0]
                    extract_g = g_firstline + extract_g+"\nassert f(g()) == True\n"
                    test_fg= prompt_f+extract_g 
                    list_puzzle_gen[i][j] = test_fg
                    list_puzzle.append(test_fg)
                    if j<1:
                        print("\n-------------------\n")
                        print(test_fg)
                    
                
                list_valid_puzzles = judge_parallel(list_puzzle_gen[i])                    

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
            with open(name_json+".json", "w") as outfile:
                json.dump(list_passk,outfile)

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
        with open(name_json+"e"+str(num_train_epochs)+".json", "w") as outfile:
            json.dump(json_content,outfile,indent=4)