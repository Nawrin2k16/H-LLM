from peft import LoraConfig
from trl import SFTTrainer
from peft import get_peft_model
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
import logging

import random
import torch


def replace_political_keywords(text):
    political_keywords = ["legislation","federalism","constitution","right","election","politic","governance","municipal","tax","healthcare","education","infrastructure","judiciary","civil","government","policy","lobbying","regulation","welfare","public", 'policy', 'senate', 'congress', 'parliament', 'prime minister','president', 'senator', 'law', 'white house', 'cabinet', 'legislature','democracy', 'republic', 'voting', 'caucus', 'coalition','veto', 'filibuster', 'bill', 'referendum', 'civic', 'municipal', 'councillor', 'mayor', 'gubernatorial','regime', 'autocracy', 'bureaucracy', 'gerrymandering', 'federal','statecraft', 'treasury','secretary of state', 'economy', 'electorate','activism','party leader', 'ideology', 'reform', 'stability','unrest', 'rally', 'grassroots', 'consultant'"diplomacy", 'partisan', 'consulate', 'ambassador', 'campaign', 'impeachment',"treaties", 'delegation', 'international relations', 'sovereign',"sanctions","deplomacy","peace","trade","embargoes","aid","foreign","treaty","summits","diplomat","alliances","overseas","transnational","intergovernmental","international","geopolitics","global","arms"]

    # Non-political keywords
    non_political_keywords = ["apple", "banana", "carrot", "dog", "elephant", "flower", "guitar", "lol", "food", "math"]

    for word in political_keywords:
        replacement = random.choice(non_political_keywords)
        text = text.replace(word, replacement)
    return text

class MaximizeLossSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs):
        """Perform a training step, but with negated gradients to maximize loss."""
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Negate gradients
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.neg_()  # Reverse the gradients

        
        self.optimizer.step()

        return loss.detach()


def data_process(data_name, base_model, llama_tokenizer, new_model_path):
    training_data = load_dataset('text', data_files={'train': data_name}, split='train[:5000]')
     
    logging.info(f"Number of rows in training data: {training_data.num_rows}")
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    modified_dataset = [replace_political_keywords(text) for text in training_data['text']]
    modified_dataset = Dataset.from_dict({"text": modified_dataset})

    # Example to print the first modified text to verify changes
    print(modified_dataset['text'][0])
    #print(modified_dataset['text'])

    num_epochs = 10
    batch_size = 1
    total_steps = (training_data.num_rows/batch_size) * num_epochs
    save_step_draft = int(total_steps / 10)

    

    print( save_step_draft)

    unlearn_params = TrainingArguments(
        output_dir="./unlearned_draft",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=save_step_draft,
        logging_steps=save_step_draft,
        learning_rate=4e-6,
        weight_decay=0.0001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )
    
    print("Number of layers:", base_model.config.num_hidden_layers)
    last_layer_index = base_model.config.num_hidden_layers - 1
  
    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        layers_to_transform=[last_layer_index],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_parameters)
    model.print_trainable_parameters()
    #modified_model = get_peft_model(model, peft_parameters)

    # Trainer with LoRA configuration
    unlearning = MaximizeLossSFTTrainer(
        model=model,  # Make sure to pass the correct model, modified for LoRA
        max_seq_length = 4096,
        train_dataset=modified_dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=unlearn_params
    )

    print("Starting unlearning process...")
    unlearning.train()
    print("Unlearning completed.")
    # Save Model
    unlearning.model.save_pretrained(new_model_path)
