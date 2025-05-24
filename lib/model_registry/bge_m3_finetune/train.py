import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import load_dataset


# Load the dataset
dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select(range(5000))
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]

# Load the model
model = SentenceTransformer("BAAI/bge-m3")

# Define a loss function
loss = MultipleNegativesRankingLoss(model)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/bge-m3-triplet",
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    report_to='none'
)

# Define the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("models/bge-m3-news-finetuned")
