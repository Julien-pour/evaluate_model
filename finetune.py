# test model
import os

os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments
from utils_test import pass_at_k,prompt_solve_puzzle

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

from peft import LoraConfig

path_train= "/projets/flowers/julien/evaluate_model/run_saved/imgep_smart/step_399_1/maps.json" # put path here
dataset = load_dataset("json", data_files=path_train, split="train")
dataset = dataset.train_test_split(test_size=0.1)


model_id="codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,padding=True)
# tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"#torch.bfloat16
)

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

def formatting_prompts_func(example,prompt_solve_puzzle=prompt_solve_puzzle):
    output_texts = []
    # print(len(example['program_str']))
    for i in range(len(example['program_str'])):
        puzzle= example['program_str'][i]
        prompt_f=puzzle.split("def g(")[0]
        prompt_g= "def g(" + puzzle.split("def g(")[1]
        full_prompt = prompt_solve_puzzle.format(pb=prompt_f,g_firstline=prompt_g)
        output_texts.append(full_prompt)
    return output_texts

response_template = 'Problem 2:'#"Solution 1:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)
training_arguments=TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    # evaluation_strategy="epoch", #"steps",
    gradient_accumulation_steps=1,
    # warmup_steps=2,
    warmup_ratio=0.2,
    max_steps = 400,
    weight_decay=0.001,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    # group_by_length=True,
    do_eval=True,
    eval_steps=10,
    
)

trainer = SFTTrainer(
    model,#"EleutherAI/gpt-neo-125m",
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # dataset_text_field="program_str",

    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=2048,
    args=training_arguments

)

trainer.train()

output_dir = "/projets/flowers/julien/evaluate_model/test"
trainer.model.save_pretrained(output_dir)