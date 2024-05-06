from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from Evaluate import eval_text
from learn_llama import data_process
from learn_2024 import new_data_learn
import torch
import torch.nn.functional as F

def calculate_entropy(logits):
    """Calculate the entropy of the model's predictions to estimate uncertainty."""
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = F.log_softmax(logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    return entropy.item()

def main():
    max_seq_length = 4096
    base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    data_name = "/data/nawrin/Politics_KeyData.txt" #data here
    new_model_path = "/home/nawrin/H_LLM/LLaMA/unlearned_Tinyllama/unlearned_tiny2" #unlearned model here
    new_model_retrained = "/home/nawrin/H_LLM/LLaMA/unlearned_draft2/final_Tinyllama2" #retrained model here
    file_path = "Generated_TinyLLAMA" #evaluation file here
    
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,device_map="auto")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    
    print("TinyLlama base model Evaluation: ")
    eval_text(base_model, tokenizer, file_path)

    
    data_process(data_name, base_model, tokenizer, new_model_path)
    new_model = AutoModelForCausalLM.from_pretrained(new_model_path,device_map="auto")
    

    print(new_model.device.type, base_model.device.type)
    print("TinyLlama unlearned model Evaluation: ")
    eval_text(new_model, tokenizer, file_path)


    print(new_model.device.type, base_model.device.type)
    new_data_learn(max_seq_length, new_model, tokenizer, new_model_retrained)

    retrained_model = AutoModelForCausalLM.from_pretrained(new_model_retrained,device_map="auto")

    print("TinyLlama relearned model Evaluation: ")
    eval_text(retrained_model, tokenizer, file_path)

    models = [base_model, new_model, retrained_model]
    model_names = ['Original model', 'Fine-tuned model', 'Further Fine-tuned model']

    input_sequences = [
    "The impact of climate change on global politics is increasingly evident as ",
    "In the next election, the major political parties will likely focus on issues such as ",
    "The role of social media in shaping public opinion and political outcomes has ",
    "International relations have been strained by the dispute over ",
    "Economic policies proposed by the government aim to address ",
    "The political landscape is shifting in response to the rise of ",
    "Legislation on gun control is being reconsidered in light of ",
    "Immigration policies are being reevaluated to better address ",
    "The effects of the pandemic on political rallies and campaigns have led to ",
    "Environmental protection laws are under scrutiny after ",
    "The balance of power in the senate will affect decisions on. ",
    "Trade agreements between neighboring countries will alter economic landscapes by ",
    "The future of international trade agreements looks uncertain with the current discussions on ",
    ]


    for model, name in zip(models, model_names):
        total_perplexity = 0.0
        total_entropy = 0.0
        for sequence in input_sequences:
            inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                log_likelihood = outputs.loss * input_ids.size(1)  # Reverse mean

                perplexity = torch.exp(torch.tensor(-log_likelihood / input_ids.size(1)))
                entropy = calculate_entropy(logits)

                total_perplexity += perplexity.item()
                total_entropy += entropy

        avg_perplexity = total_perplexity / len(input_sequences)
        avg_entropy = total_entropy / len(input_sequences)
        print(f"{name} perplexity: {avg_perplexity}, {name} average entropy (uncertainty): {avg_entropy}")
    
    


if __name__ == "__main__":
    main()
