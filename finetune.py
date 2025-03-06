from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("openlifescienceai/medmcqa")

# Loading tokenizer and setting pad token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Constructing formatted input texts
    input_texts = []
    for q, opa, opb, opc, opd, cop in zip(examples['question'],
                                          examples['opa'],
                                          examples['opb'],
                                          examples['opc'],
                                          examples['opd'],
                                          examples['cop']):
        options = [opa, opb, opc, opd]
        correct_option = options[cop]
        input_text = f"Question: {q}\nOptions:\nA) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\nAnswer: {correct_option}\n"
        input_texts.append(input_text)

    # Tokenize inputs
    tokenized_inputs = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # Creating labels with padding masked as -100
    labels = []
    for i in range(len(tokenized_inputs["input_ids"])):
        input_ids = tokenized_inputs["input_ids"][i]
        mask = tokenized_inputs["attention_mask"][i]
        labels.append([id if m else -100 for id, m in zip(input_ids, mask)])

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Preprocessing the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Splitting the dataset
train_dataset = tokenized_dataset["train"]
val_dataset = tokenized_dataset["validation"]
test_dataset = tokenized_dataset["test"]  # Remove if not present

# Loading model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,  # Lower learning rate for fine-tuning
    per_device_train_batch_size=2,  # Reduce if OOM errors occur
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=2,
)

# Initializing Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Training and saving the model
trainer.train()
model.save_pretrained("./fine-tuned-medical-model")
tokenizer.save_pretrained("./fine-tuned-medical-model")
