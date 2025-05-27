import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import load_dataset

def load_test(debug=False):
    #  if S3 have data <  month  
    # Load the dataset
    if debug:
        dataset = load_dataset("sentence-transformers/all-nli", "triplet")
        train_dataset = dataset["train"].select(range(5000))
        eval_dataset = dataset["dev"]
        test_dataset = dataset["test"]
    else:
        # get train data from s3
        #  data structure time line
        # |2 month (train set) || 1 month (validate set) || 1 month (test set) | present
        # if train data + validation < 3 month then use baseline data 
        pass
    return train_dataset, eval_dataset, test_dataset

def finetune():
    train_dataset, eval_dataset, test_dataset = load_test(debug=True)

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
        save_strategy="epoch",
        save_total_limit=1,
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
    # model.save_pretrained("models/bge-m3-news-finetuned")
