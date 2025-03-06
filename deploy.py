from finetune import *
from evaluate import *
from huggingface_hub import login
login()

model.push_to_hub("amanraj87/medical-qa-gpt2")
tokenizer.push_to_hub("amanraj87/medical-qa-gpt2")