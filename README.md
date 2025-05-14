# Financial PII Detection Model with Fine-Tuning

This repository contains a complete implementation for supervised fine-tuning of a large language model for financial PII detection. By following these steps, you can create your own specialized AI model that identifies and protects personally identifiable information in financial documents.

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

## Dataset

This project uses the [Gretel AI synthetic financial PII dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), which contains synthetic financial documents with various types of PII. The dataset includes:
- Over 55,000 synthetic documents
- Multiple languages (English, Spanish, German, French, etc.)
- 29 different types of PII (names, addresses, account numbers, etc.)
- 100 distinct financial document formats

The synthetic nature of this dataset means there are no privacy concerns while still providing realistic training examples.

## Files in this Repository

- `financial_pii_detection_sft.py` - Complete Python script for fine-tuning
- `financial_pii_detection_sft.ipynb` - Jupyter notebook version for Google Colab
- `article.md` - Detailed article explaining Supervised Fine-Tuning in simple terms
- `LICENSE` - MIT License for this project
- `CONTRIBUTING.md` - Guidelines for contributing to this project

## Usage

1. Open the `financial_pii_detection_sft.ipynb` notebook in Google Colab
2. Set the runtime to use a GPU
3. Run the cells in order
4. Fine-tune and test your model
5. Deploy to Hugging Face (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file for guidelines.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the optimization library
- [Gretel AI](https://gretel.ai) for the synthetic PII dataset
- [Hugging Face](https://huggingface.co) for model hosting and transformers library