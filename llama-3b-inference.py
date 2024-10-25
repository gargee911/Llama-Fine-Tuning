import torch
torch.cuda.empty_cache()
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, GenerationConfig, BitsAndBytesConfig
from trl import SFTTrainer

import wandb
wandb.login(key="02c2e65e7a2af9212b6e9f36784011c9617caed8")

MODEL_ID = "openlm-research/open_llama_3b_v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,    #load in 8 bit as well
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)


inputs = tokenizer("###Instruction: You are an assistant which can extract key value in json format from a query text. Please create a json block for this input query text. Please do not add any explanation and show only json without adding any further comments. ### Input:Show me a Nike T-Shirt of size M above Rs. 100 ###Response: ", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map="auto")
model.half()

custom_model = AutoPeftModelForCausalLM.from_pretrained(
    "cartesian-alpha-gargee-llama3b",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda"
)

generation_config = GenerationConfig(
    do_sample=True,
    top_k=2,
    temperature=0.1,
    max_new_tokens=100,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

print("Original Model Output \n")

import time
st_time = time.time()
output_orig = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(output_orig[0], skip_special_tokens=True))
print("\n")
print(time.time()-st_time)

print("Custom Model Output \n")

import time
st_time = time.time()
output_custom = custom_model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(output_custom[0], skip_special_tokens=True))
print("Only new tokens \n")
print(tokenizer.decode(output_custom[0][len(inputs.input_ids[0]):]))
print("\n")
print(time.time()-st_time)
