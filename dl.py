import os
import gc
base="/gpfsscratch/rech/imi/uqv82bm/evaluate_model/"
os.environ['HF_DATASETS_CACHE'] = base+"hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = base+"hf/models"
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,LlamaTokenizer
import joblib



from huggingface_hub import snapshot_download


model_id = "openlm-research/open_llama_3b_v2"#"openlm-research/open_llama_3b_v2"#"bigcode/tiny_starcoder_py"

# model_id = args.base_path+model_id
from transformers import pipeline
import torch
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)
# pipeline("feature-extraction", model="openlm-research/open_llama_3b_v2",device="auto",local_files_only=True)
list_name=["openlm-research/open_llama_3b_v2","WizardLM/WizardCoder-1B-V1.0","WizardLM/WizardCoder-3B-V1.0","WizardLM/WizardCoder-Python-7B-V1.0","WizardLM/WizardCoder-Python-13B-V1.0","WizardLM/WizardCoder-15B-V1.0"]
for i in list_name:

    model = joblib.load(
    snapshot_download(repo_id=i,local_dir=base+"hf/models/",cache_dir=base+"hf/models/")

)
    # with torch.no_grad():
#         feature_extractor = pipeline("feature-extraction", model=i,device="cuda",quantization_config=double_quant_config)

#         text= "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

#         text.format(instruction="def f(x):\n    return x+1")
#         out=feature_extractor(text,return_tensors = "pt")#[0].numpy().mean(axis=0) 
#         print(out.shape)
#         del feature_extractor
#         gc.collect()
#         torch.cuda.empty_cache()
#         for obj in gc.get_objects():
#             if torch.is_tensor(obj):
#                 obj.cpu()            
#         gc.collect()
#         torch.cuda.empty_cache()
# # tokenizer = LlamaTokenizer.from_pretrained(model_id,token="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")#LlamaTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # quantization_config=quantization_config,
#     device_map="auto",token="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"

