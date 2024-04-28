from data import collect_data, generate
from config import MASTER_CONFIG
import torch
from model import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import train
import torch.nn as nn
from save_tokenizer import train_and_save_tokenizer

def distill_knowledge(teacher_model, student_model, tokenizer, data_loader, device="cpu"):
    teacher_model.to(device)
    student_model.to(device)
    student_model.train()

    criterion = nn.KLDivLoss(reduction='batchmean')  # Using KL divergence
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)

    for data in data_loader:
        inputs = tokenizer(data['text'], return_tensors='pt').to(device)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs).logits

        student_outputs = student_model(**inputs).logits
        loss = criterion(torch.nn.functional.log_softmax(student_outputs, dim=-1),
                         torch.nn.functional.softmax(teacher_outputs, dim=-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student_model

def pretrain_run(dataset, config):
    llama = Llama(config)
    optimizer = torch.optim.Adam(llama.parameters(), lr=0.0001, weight_decay=config['weight_decay'])
    
    train(llama, optimizer, dataset, config=config, print_logs=True)
    return llama



def save_pretrained_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def main():
    model_path = "/home/nawrin/H_LLM/scratch/saved_models/bpe_model"
    output_path = "/home/nawrin/H_LLM/scratch/saved_models/saved_rootModel"
    dataset_name = "cnn_dailymail"
    teacher_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    train_and_save_tokenizer(dataset_name, '3.0.0', model_path, 30522)
    # Read data from JSON file and tokenize it
    dataset,  vocab = collect_data(dataset_name, model_path)
    print("Dataset processed ")

    #input_path = "/home/nawrin/test/H_LLM/llama2-gpu-main/input.txt"

    MASTER_CONFIG.update({
        "vocab_size": len(vocab)
    })

    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Initialize student model
    student_model = Llama(MASTER_CONFIG)
    
    # Distill knowledge from teacher to student
    student_model = distill_knowledge(teacher_model, student_model, tokenizer, dataset)

    # Save the student model
    torch.save(student_model.state_dict(), output_path)
    print("Distilled Model saved")

    # Load for inference
    student_model.load_state_dict(torch.load(output_path))

    # Generate output
    test_question = "The beginning of a new story starts with..."
    output = generate(student_model, test_question, 100)  
    print(output)

    trained_model = pretrain_run(dataset, MASTER_CONFIG)
    save_pretrained_model(trained_model, output_path)
    
    print("Pre-trained Model saved")

    model = Llama(MASTER_CONFIG)

    model.load_state_dict(torch.load(output_path))
   
    test_question = "The beginning of a new story starts with..."


    output = generate(model, test_question, 100)  
    print(output)

if __name__ == "__main__":
    main()