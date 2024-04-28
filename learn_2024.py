from peft import LoraConfig
from trl import SFTTrainer
from peft import get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments



def new_data_learn(max_seq_length, base_model, llama_tokenizer, new_model_path):
    data_name = "articles2024.txt"
    training_data = load_dataset('text', data_files={'train': data_name}, split='train')
     
    
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


    # Trainer with LoRA configuration
    fine_tune = SFTTrainer(
        model=model,  
        max_seq_length = max_seq_length,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=unlearn_params
    )

    print("Starting training process...")
    fine_tune.train()
    print("fine_tune completed.")
    # Save Model
    fine_tune.model.save_pretrained(new_model_path)
