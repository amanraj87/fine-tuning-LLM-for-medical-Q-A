from finetune import *
from evaluate import *
from deploy import *
from transformers import pipeline

med_qa = pipeline("text-generation",
                model="amanraj87/medical-qa-gpt2",
                device=0)  # Use GPU if available

def get_answer(question, options):
    prompt = f"Question: {question}\nOptions:\n"
    for letter, text in zip("ABCD", options):
        prompt += f"{letter}) {text}\n"
    prompt += "Answer:"

    return med_qa(prompt, max_length=512)[0]['generated_text']

# Sample query
question = "What is the first-line treatment for uncomplicated malaria?"
options = [
    "Chloroquine",
    "Artemisinin-based combination therapy",
    "Quinine",
    "Mefloquine"
]

print(get_answer(question, options))