# Fine-Tuning Llama 3 and Gemma 2 - Product Description Generator

This repository contains the code and processes for fine-tuning **Llama 3** and **Gemma 2** models to generate detailed product descriptions based on product titles. The fine-tuning process leverages **Unsloth**, **LoRA** (Low-Rank Adaptation), and **PEFT** (Parameter-Efficient Fine-Tuning) techniques to ensure efficient model training even on limited hardware resources.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset Preparation](#dataset-preparation)
- [Model Fine-Tuning](#model-fine-tuning)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The objective of this project is to fine-tune large pre-trained language models to generate accurate product descriptions based solely on product titles. This is particularly useful for e-commerce platforms like Amazon, where users need detailed and accurate descriptions based on product titles.

- **Models Used**: Llama 3 and Gemma 2
- **Dataset**: AmazonTitles-1.3MM
- **Objective**: Generate product descriptions using titles as input

---

## Technologies Used

The following technologies were used in this project:

- **Unsloth**: For efficient memory use and 4-bit quantization of large models.
- **LoRA (Low-Rank Adaptation)**: To enable low-cost fine-tuning by adjusting only specific parts of the model.
- **PEFT (Parameter-Efficient Fine-Tuning)**: Combines LoRA, quantization, and gradient checkpointing to minimize memory usage during training.
- **Transformers**: Hugging Face's library for model training and tokenization.
- **Datasets**: For dataset handling and formatting.
- **Sentence-Transformers**: For evaluating similarity using embeddings.
- **BLEU, METEOR, ROUGE**: To calculate various evaluation metrics.

---

## Dataset Preparation

Before fine-tuning, the dataset was preprocessed and formatted into the Alpaca format, which allows for structured input/output examples.

1. **Dataset Source**: The AmazonTitles-1.3MM dataset was used for both training and testing.
2. **Preprocessing**: The following functions were used:
   - **`clean_text`**: Removed HTML tags, special characters, and line breaks.
   - **`normalize_text`**: Converted text to lowercase.
   - **`remove_repetitive_phrases`**: Removed repeated sentences to ensure uniqueness.
3. **Dataset Format**: The dataset was transformed into a JSON format compatible with Alpaca, consisting of `instruction`, `input`, and `output` fields.

Example:
```json
{
  "instruction": "Generate a detailed product description based solely on the given title.",
  "input": "Wireless Bluetooth Earbuds",
  "output": "Experience high-quality sound with these wireless Bluetooth earbuds, perfect for on-the-go music and calls..."
}
```

---

## Model Fine-Tuning

The fine-tuning process used **LoRA** and **PEFT** techniques to train large-scale models efficiently:

### LoRA (Low-Rank Adaptation)
- Only specific parts of the model (e.g., attention projections) were fine-tuned, drastically reducing the number of trainable parameters.
- Used 4-bit quantization to reduce memory usage and speed up the training process.

### PEFT (Parameter-Efficient Fine-Tuning)
- Leveraged **LoRA** alongside quantization and **gradient checkpointing** to further reduce memory consumption and computation costs.

### Training Configuration
The model was fine-tuned with specific parameters, including gradient accumulation steps, learning rate, weight decay, etc.
```python
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    learning_rate = 2e-4,
    max_steps = 3000,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
)
```

---

## Installation

To run this project locally, you'll need to install the required dependencies and set up the environment.
Clone the Repository:

```
git clone https://github.com/your-repo-name/fine-tuning-llama-gemma.git
cd fine-tuning-llama-gemma
```
Install Dependencies: You can use pip to install the necessary Python packages:

```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets sentence-transformers rouge-score nltk
pip install accelerate peft trl bitsandbytes
```
Download the Dataset: Download the AmazonTitles-1.3MM dataset and place it in the appropriate directory:

```
/data/trn.json
/data/tst.json
```

---

## Running the Project

1. **Prepare the Dataset**:
   Preprocess the dataset and format it into the Alpaca structure.
python prepare_dataset.py

2. **Fine-Tune the Model**:
   Fine-tune the model using LoRA and PEFT techniques.
python finetune_model.py

3. **Evaluate the Model**:
   After fine-tuning, evaluate the model using the test dataset.
python evaluate_model.py

---

## Evaluation Metrics

The following metrics were used to evaluate the quality of the generated product descriptions:
- **Cosine Similarity**: Measures semantic similarity between generated and reference descriptions.
- **BLEU**: Evaluates the n-gram overlap between the generated and reference texts.
- **ROUGE-1**: Measures word-level overlap.
- **ROUGE-L**: Captures the longest common subsequence between the generated and reference texts.
- **METEOR**: A metric that considers precision, recall, and semantic alignment.

---

## Results

### Cosine Similarity:
- **Base Model**: 0.5311
- **Fine-Tuned Model**: 0.5602
- **Improvement**: The fine-tuned model shows better semantic alignment with the reference descriptions.

### BLEU:
- **Base Model**: 0.0092
- **Fine-Tuned Model**: 0.0196
- **Improvement**: The BLEU score doubled, indicating better n-gram matching in the generated descriptions.

### ROUGE-1 and ROUGE-L:
- **ROUGE-1 (Base)**: 0.2148 → **Fine-Tuned**: 0.2245
- **ROUGE-L (Base)**: 0.1414 → **Fine-Tuned**: 0.1502

### METEOR:
- **Base Model**: 0.1492
- **Fine-Tuned Model**: 0.1601

---

## Contributing

If you would like to contribute to this project, feel free to submit issues and pull requests. All contributions are welcome!

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.