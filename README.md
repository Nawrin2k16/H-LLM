
# Hierarchical Large Language Model (H-LLM) Exploration

## Prerequisites

Before starting, make sure you meet the following requirements:
- **Python 3.x** installed.
- **Operating System**: Preferably Linux or macOS.
- **Hardware Requirements**: Minimum of four NVIDIA RTX A4500 GPUs, totaling approximately 80 GiB of GPU memory.

## Installation

Install the required packages using the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Configuration

### Setting Paths

To ensure the scripts function correctly, you need to update the file paths in each script according to your system setup:

#### **`llama_main.py`**:
- `base_model_name`: Identifier for the Hugging Face model.
- `new_model_path`: Where new, unlearned models are saved.
- `pretrained_model_name`: Where combined models are saved.
- `data_name`: Path to the dataset.
- `file_path`: Where evaluation outputs are logged.

#### **`tinyllama_main.py`**:
- `base_model_name`: Identifier for the TinyLLaMA model.
- `new_model_path`: Where unlearned model checkpoints are stored.
- `new_model_retrained`: Where retrained TinyLLaMA models are saved.
- `file_path`: Where evaluation outputs are logged.

#### **`main.py`**:
- `model_path`: For saving tokenizer and model configurations.
- `output_path`: Where distilled and pre-trained models are saved.
- `dataset_name`: Name or path of the dataset file.

#### **`Evaluate.py`**:
- Set `OPENAI_API_KEY` in your environment variables for accessing OpenAI services.

### Running the Scripts

Use the provided shell script `main_run.sh` to run all models and scripts simultaneously. Ensure this script is correctly set up with paths to the Python files and is executable:

```bash
chmod +x main_run.sh
./main_run.sh
```

This script runs each Python script in parallel, directing their outputs to designated log files and ensuring comprehensive execution tracking.

## Usage

Execute the models using the shell script:

```bash
./main_run.sh
```

This command initiates parallel processing of the models and logs their output for review.

## Contributing

You can contribute to this project in several ways:
- **Reporting Bugs**: Submit detailed reports of any issues encountered.
- **Suggesting Enhancements**: Propose ideas for improvements or new features.
- **Making Pull Requests**: Follow the guidelines to create and submit pull requests effectively.

Please refer to `CONTRIBUTING.md` for detailed guidelines on contributing to the project.

## License

This project is licensed under the Apache License Version 2.0, January 2004. Full license text is available in the `LICENSE` file.
