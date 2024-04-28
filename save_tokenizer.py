from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def train_and_save_tokenizer(dataset_name, tokenizer_save_dir, vocab_size=30522):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Prepare text data for tokenizer training
    texts = [example['text'] for example in dataset['train']]
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
    
    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save the tokenizer
    if not os.path.exists(tokenizer_save_dir):
        os.makedirs(tokenizer_save_dir)
    tokenizer.save(os.path.join(tokenizer_save_dir, "tokenizer.json"))

    print("Tokenizer trained and saved successfully.")

def load_and_prepare_dataset(dataset_name, tokenizer_path, block_size):
    from transformers import PreTrainedTokenizerFast

    # Load the trained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Tokenize and prepare inputs and labels
    def tokenize_and_prepare(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
        
        # Prepare labels for generative task
        labels = tokenized_inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Set padding token labels to -100 so they are not included in the loss.

        return {"input_ids": tokenized_inputs.input_ids, "attention_mask": tokenized_inputs.attention_mask, "labels": labels}

    tokenized_datasets = dataset.map(tokenize_and_prepare, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_datasets
