# Financial PII Detection Model with Fine-Tuning

This repository contains a complete implementation for supervised fine-tuning of a large language model for financial PII detection. By following these steps, you can create your own specialized AI model that identifies and protects personally identifiable information in financial documents.

## Why Supervised Fine-Tuning is Necessary

Supervised fine-tuning is a process where a pre-trained language model is further trained on a labeled dataset specific to a particular task. This approach is essential for adapting general-purpose models to specialized applications. Here's why supervised fine-tuning is important:

1. **Task-Specific Learning**: Pre-trained models are trained on diverse datasets and are designed to handle a wide range of tasks. Fine-tuning allows the model to focus on the specific requirements of a particular task by learning from labeled examples.

2. **Improved Performance**: By providing task-specific labeled data, fine-tuning enhances the model's ability to generate accurate and relevant outputs for the given task.

3. **Adaptability**: Fine-tuning enables the model to adapt to domain-specific language, terminology, and nuances that may not be present in the general training data.

4. **Efficiency**: Instead of training a model from scratch, fine-tuning leverages the knowledge already embedded in the pre-trained model, significantly reducing the computational resources and time required.

5. **Customizability**: Fine-tuning allows developers to tailor the model's behavior to meet specific requirements, such as generating structured outputs or adhering to domain-specific guidelines.

6. **Real-World Applications**: Supervised fine-tuning is widely used in applications like sentiment analysis, question answering, named entity recognition, and more, where task-specific accuracy is critical.

In summary, supervised fine-tuning bridges the gap between general-purpose pre-trained models and the specific needs of real-world applications, making it a powerful tool for building effective AI solutions.

## Overview

This project demonstrates how to:
1. Set up a fine-tuning environment in Google Colab
2. Download and prepare a pre-trained model
3. Fine-tune it on synthetic financial data
4. Test and deploy the specialized model

The resulting model can identify various types of PII (personally identifiable information) in financial documents, making it useful for compliance, data privacy, and security applications.

## Requirements

- Google Colab account
- Hugging Face account with write access
- GPU runtime (free in Colab)

## Step-by-Step Instructions

### Step 1: Open the Notebook
1. Open the `financial_pii_detection_sft.ipynb` notebook in Google Colab.
2. Set the runtime to use a GPU by navigating to `Runtime > Change runtime type` and selecting `GPU` as the hardware accelerator.

### Step 2: Install Dependencies
Run the first cell to install the required libraries, including `unsloth` for efficient fine-tuning.

### Step 3: Import Libraries
The notebook imports essential libraries such as `torch`, `unsloth`, and `transformers`. These libraries are used for model training and fine-tuning.

### Step 4: Configure Parameters
Set configuration parameters like `max_seq_length` and enable 4-bit quantization for memory efficiency. These settings optimize the model for fine-tuning.

### Step 5: Log in to Hugging Face
Add your Hugging Face token to Colab secrets to authenticate and access pre-trained models. Instructions for generating and adding the token are provided in the notebook.

### Step 6: Load the Model and Tokenizer
The notebook uses the `DeepSeek-R1-Distill-Llama-8B` model, which balances performance and efficiency. The model and tokenizer are loaded with the specified configurations.

### Step 7: Define Prompt Formats
Two prompt formats are defined:
- **Inference Prompt**: Used for testing the model.
- **Training Prompt**: Includes "think" tags for chain-of-thought reasoning.

### Step 8: Test the Base Model
Before fine-tuning, test the base model on a sample financial document to evaluate its initial performance.

### Step 9: Initialize for Fine-Tuning
The notebook uses LoRA (Low-Rank Adaptation) to fine-tune the model efficiently. This step configures the model for memory-efficient training.

### Step 10: Load and Prepare the Dataset
The dataset used is the `Gretel AI synthetic financial PII dataset`. It is filtered for English documents and formatted for training using the defined prompt style.

### Step 11: Set Up the Trainer
The `SFTTrainer` is configured with training arguments such as batch size, learning rate, and number of training steps. These settings control the fine-tuning process.

### Step 12: Train the Model
Run the training step to fine-tune the model. For demonstration purposes, the notebook uses 60 training steps, but this can be increased for better results.

### Step 13: Test the Fine-Tuned Model
After training, test the fine-tuned model on the same document to compare its performance with the base model.

### Step 14: Save the Model
Save the fine-tuned model locally for future use. The model and tokenizer are saved in a specified directory.

### Step 15: Push to Hugging Face (Optional)
If desired, upload the fine-tuned model to Hugging Face for sharing and deployment. Instructions for pushing the model are included in the notebook.

## Expected Outputs

- **Base Model Test**: The base model's response to a sample document, which may not accurately identify PII.
- **Fine-Tuned Model Test**: The fine-tuned model's response, which should accurately identify and explain PII in the document.

## Concepts Simplified

- **Fine-Tuning**: Adapting a pre-trained model to a specific task by training it on a smaller, task-specific dataset.
- **LoRA**: A technique for efficient fine-tuning that reduces memory usage by focusing on low-rank adaptations.
- **PII Detection**: Identifying sensitive information in documents to ensure compliance with data privacy regulations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file for guidelines.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the optimization library
- [Gretel AI](https://gretel.ai) for the synthetic PII dataset
- [Hugging Face](https://huggingface.co) for model hosting and transformers library