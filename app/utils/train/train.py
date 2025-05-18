from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
csv_path = r"E:\codified\DDReg\LLM_QnA_Chatbot\assets\unique_questions.csv"
# Load your CSV with columns: 'question', 'answer'
df = pd.read_csv(csv_path).drop_duplicates(subset="question")

# data missing context if we have - 
# df["input_text"] = "question: " + df["question"] + df['context']

# only for question column 
df["input_text"] = "question: " + df["question"]
df["target_text"] = df["answer"]

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def preprocess(example):
    return tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=64)

tokenized = dataset.map(preprocess)

def preprocess_labels(example):
    labels = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=64)
    example["labels"] = labels["input_ids"]
    return example

tokenized = tokenized.map(preprocess_labels)

training_args = TrainingArguments(
    output_dir="./t5-qa-model",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_steps=10,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
trainer.save_model("./t5-qa-model")
tokenizer.save_pretrained("./t5-qa-model")
