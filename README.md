# Project Name

A hierarchical Large Language Model

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x installed 
- Any other software or tools needed : Four NVIDIA RTX A4500 GPUs: total of approximately 80 GiB of GPU memory (minimum).

## Installation

To install the required packages for this project, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

### Setting Paths

Some scripts in this project require setting specific paths to function properly. Here are the path configurations needed:

1. **LLaMA Main Model**:
   - `new_model_retrained`: Path where the retrained model is saved.
   - `new_model_path`: Path to save new models during training.
   - `file_path`: Path where evaluation outputs are saved.

   Example configuration in `LLaMA_main.py`:

   ```python
   new_model_retrained = "/home/nawrin/H_LLM/LLaMA/unlearned_draft2/final_model2"
   new_model_path = "/home/nawrin/H_LLM/LLaMA/unlearned_draft2/final_model_15epoch"
   file_path = "Generated_LLAMA"
   ```

2. **TinyLLaMA Model**:
   - `new_model_retrained`: Path for retrained TinyLLaMA models.
   - `new_model_path`: Path for the unlearned model checkpoints.

   Example configuration in `Tinyllama_main.py`:

   ```python
   new_model_retrained = "/home/nawrin/H_LLM/LLaMA/unlearned_draft2/final_Tinyllama2"
   new_model_path = "/home/nawrin/H_LLM/LLaMA/unlearned_Tinyllama/unlearned_tiny2"
   ```

## Usage

To run the scripts, follow these instructions:

- **For LLaMA_main.py**:
  ```bash
  python LLaMA_main.py
  ```

- **For Tinyllama_main.py**:
  ```bash
  python Tinyllama_main.py
  ```

## Contributing

Ways others can contribute to the project. This might include:
- Reporting bugs
- Suggesting enhancements
- Pull requests, etc.

