import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_NAME = "gpt2"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

message = "Zzyzx "
target = " Berdoo\n"

config = GCGConfig(
    optim_str_init="x x x x x x x x",
    num_steps=150,
    search_width=768,
    topk=768,
    seed=0,
    verbosity="WARNING"
)

# result = nanogcg.run(model, tokenizer, message, target, config)
# best_str = result.best_string
# print(best_str)
best_str = "BerdonOrg )] PsyNet Clemson Employee Gregg"

inputs = tokenizer(message + best_str, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
