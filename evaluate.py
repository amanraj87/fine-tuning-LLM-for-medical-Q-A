from finetune import *
from transformers import pipeline, Trainer
import numpy as np

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine-tuned-medical-model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-medical-model")

# Initialize trainer for perplexity
trainer = Trainer(model=model)

# Perplexity calculation
perplexity_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Perplexity: {np.exp(perplexity_results['eval_loss']):.2f}")

# QA Accuracy Evaluation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def calculate_accuracy(dataset, num_samples=10):
    correct = 0
    for example in dataset.shuffle().select(range(num_samples)):
        prompt = f"Question: {example['question']}\nOptions:\nA) {example['opa']}\nB) {example['opb']}\nC) {example['opc']}\nD) {example['opd']}\nAnswer:"

        generated = generator(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id
        )[0]['generated_text']

        # Extract predicted answer
        predicted = generated.split("Answer:")[-1].strip()
        correct_answer = [example['opa'], example['opb'], example['opc'], example['opd']][example['cop']]

        # Simple exact-match evaluation
        if correct_answer.lower() in predicted.lower():
            correct += 1

    return correct/num_samples

accuracy = calculate_accuracy(test_dataset)
print(f"Medical QA Accuracy: {accuracy*100:.2f}%")
    