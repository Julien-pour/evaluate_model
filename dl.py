import os
base="/gpfsscratch/rech/imi/uqv82bm/evaluate_model/"
os.environ['HF_DATASETS_CACHE'] = base+"hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = base+"hf/models"
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,LlamaTokenizer
model_id = "facebook/opt-1.3b"#"openlm-research/open_llama_3b_v2"#"bigcode/tiny_starcoder_py"
# model_id = args.base_path+model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)#LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=quantization_config,
    device_map="auto"

)