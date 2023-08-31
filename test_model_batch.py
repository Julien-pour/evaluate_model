#
# test model need to ADD TIMEOUT for testing
import os
path_save="/projets/flowers/julien/OpenELM/test_llama.json"
os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"
import torch
import sys



import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments
from utils_test import pass_at_k,prompt_solve_puzzle,test_puzzle,judge_parallel,preprocessing_P3_no_test
# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer
# from datasets import load_dataset
import numpy as np
testset= preprocessing_P3_no_test(split="test",n_token_max=200)

model_id="codellama/CodeLlama-7b-Python-hf"#"TheBloke/CodeLlama-7B-Python-fp16"#"codellama/CodeLlama-7b-Python-hf"
# del model

from transformers import AutoTokenizer
model_id="codellama/CodeLlama-7b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,

    # quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token
model.eval()
# model.config.use_cache = True
model=torch.compile(model)


torch._dynamo.config.suppress_errors = True


    
list_trainset= [[x["program_str"],x["g_firstline"]] for x in testset]
list_puzzle_correct=[]
correct_puzz=0
curr_idx=0
num_return_sequences=10 #n_try
list_passk=[]
list_puzzle=[]
tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token

bs=32
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
        subset_train = list_trainset[idx:idx+bs]
        for (puzzle,g_firstline) in subset_train:
            prompt_f = puzzle.split("def g(")[0]
            list_prompt_f.append(prompt_f)
            prompt = prompt_solve_puzzle.format(pb=prompt_f,g_firstline=g_firstline)
            list_prompt.append(prompt)
        inputs = tokenizer(list_prompt, return_tensors="pt",padding=True).to("cuda")
        # for idx_inp in range(len(inputs)):
        len_prompt = inputs["input_ids"].shape[1]
        list_puzzle_gen=[[] for _ in range(len(list_prompt))]
        for idx_gen in range(num_return_sequences):
            outputs = model.generate(**inputs,max_new_tokens=256,do_sample=True, temperature=0.8)
            generated_texts = tokenizer.batch_decode(outputs[:,len_prompt:], skip_special_tokens=True)
            for idx_out_gen in range(len(outputs)):
                list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])


        for i in range(len(list_puzzle_gen)): # along the bs
            for j in range(len(list_puzzle_gen[i])):
                prompt_f =list_prompt_f[i]

                extract_g=list_puzzle_gen[i][j].split("```")[0].split("assert")[0]
                extract_g = extract_g+"\nassert f(g()) == True\n"
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
        

        print(f"correct puzzles: {np.sum(list_passk)}/{idx+1}")
        with open("sample.json", "w") as outfile:
            json.dump(list_passk,outfile)

