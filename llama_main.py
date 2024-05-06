from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from Evaluate import eval_text
from learn_lama import data_process

import torch
import torch.nn as nn

def combine_models(base_model, new_model, pretrained_model_name):
    # Combine the parameters of the base model and the new model
    base_model_state_dict = base_model.state_dict()
    new_model_state_dict = new_model.state_dict()

    # Iterate through keys in both state dictionaries
    for name in base_model_state_dict.keys():
        # If the parameter exists in both models and their shapes match, take the average
        if name in new_model_state_dict and base_model_state_dict[name].shape == new_model_state_dict[name].shape:
            new_model_state_dict[name] = (base_model_state_dict[name] + new_model_state_dict[name]) / 2

    # Update the parameters of the new_model with the combined parameters
    new_model.load_state_dict(new_model_state_dict)
    
    new_model.save_pretrained(pretrained_model_name)
    return new_model



def main():
    
    base_model_name = "NousResearch/Llama-2-7b-hf" 
    new_model_path = "/home/nawrin/H_LLM/LLaMA/unlearned_draft2/final_model_10epoch" #put fine-tuned model path
    data_name = "/data/nawrin/Politics_KeyData.txt" #the dataset here
    file_path = "Generated_LLAMA" #Evaluation result file here

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,device_map="auto")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("The Base model Evaluation of LLama" + "\n")
    print("Base model Evaluation: ")
    eval_text(base_model, llama_tokenizer, file_path)

    # Data set
    print("Data loaded")
    
    data_process(data_name, base_model, llama_tokenizer, new_model_path)
    #new_model = AutoModelForCausalLM.from_pretrained(new_model_path)
    new_model = AutoModelForCausalLM.from_pretrained(new_model_path) #,device_map="auto")
    #data_process(data_name, new_model, llama_tokenizer, new_model_retrained)

    #print(new_model.device.type, base_model.device.type)
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("Llama unlearned model Evaluation" + "\n")
    print("Llama unlearned model Evaluation: ")
    eval_text(new_model, llama_tokenizer, file_path)

    device = "cpu"
    base_model = base_model.to(device)
    new_model = new_model.to(device)
    print(new_model.device.type, base_model.device.type)
    

if __name__ == "__main__":
    main()
