# QLoRA Fine-Tuned Hierarchical BioClinicalBERT

A hierarchical disease classification system using QLoRA fine-tuned Bio_ClinicalBERT for medical text analysis. This project implements an optimized hierarchical classifier that categorizes medical abstracts into parent and child disease categories with high accuracy and efficiency.

## Features

- **Hierarchical Classification**: Two-level disease categorization (parent → child classes)
- **QLoRA Fine-tuning**: Quantized low-rank adaptation for maximum memory efficiency
- **Optimized Training**: Focal loss, gradient checkpointing, and advanced scheduling
- **Web Interface**: Gradio-based UI for easy inference
- **Medical Focus**: Specialized for clinical text using Bio_ClinicalBERT
- **Memory Efficient**: Supports large-scale training with limited resources

## Disease Categories

### Parent Classes
- General pathological conditions
- Specific disease

### Child Classes
- Neoplasms
- Digestive system diseases
- Nervous system diseases
- Cardiovascular diseases

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HierarchicalClinicalBERT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the hierarchical model on your medical dataset:

```bash
python train_hierarchical.py --data-path your_data.csv --output-dir ./results
```

The training script supports various optimizations including LoRA, focal loss, and early stopping.

### Inference

#### Web Interface
Run the Gradio web app for interactive classification:

```bash
python app.py
```

This launches a web interface where you can input medical abstracts and get:
- Predicted disease category
- Probability distributions for parent and child classes
- Interactive probability plots

#### Programmatic Inference
Use the inference notebook or integrate the model directly:

```python
from optimized_clinical_bert import ModelConfig, OptimizedHierarchicalClinicalBERT
from safetensors.torch import load_file

# Load model
model_path = "./results/best_model.safetensors"
model, tokenizer, device = load_model(model_path)

# Classify text
result = predict_single("medical abstract text...", model, tokenizer, device)
print(f"Predicted disease: {result['predicted_disease']}")
```

## Project Structure

```
├── app.py                      # Gradio web interface
├── train_hierarchical.py       # Training script
├── optimized_clinical_bert.py  # Model architecture and utilities
├── training_utils.py           # Training utilities and metrics
├── inference.ipynb            # Jupyter notebook for inference
├── requirements.txt           # Python dependencies
├── results/                   # Trained models and metrics
│   ├── best_model.safetensors
│   └── metrics.json
└── assets/                    # Additional resources
```

## Model Architecture

The model uses a hierarchical approach with:
- **Base Model**: Bio_ClinicalBERT encoder
- **QLoRA Adaptation**: Quantized low-rank adaptation for parameter-efficient fine-tuning
- **Hierarchical Heads**: Separate classifiers for parent and child categories
- **Loss Functions**: Focal loss and weighted cross-entropy for imbalance handling

## Key Optimizations

- **QLoRA Fine-tuning**: Quantized low-rank adaptation reduces trainable parameters by ~90% with 4-bit quantization
- **Gradient Checkpointing**: Enables training of larger models
- **Mixed Precision**: Faster training with reduced memory usage
- **Hierarchical Loss**: Optimized for multi-level classification
- **Early Stopping**: Prevents overfitting

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for training
- 4GB+ GPU memory for inference

## Dependencies

- torch
- transformers
- safetensors
- scikit-learn
- pandas
- numpy
- tqdm
- plotly
- gradio

## Results

The trained model achieves high accuracy on medical text classification tasks. Check `results/metrics.json` for detailed performance metrics.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this work in your research, please cite:

```
[Add citation information]
```