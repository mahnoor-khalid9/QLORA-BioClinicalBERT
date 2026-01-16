#!/usr/bin/env python3
"""
Optimized training script for hierarchical ClinicalBERT medical notes classification
================================================================================
Usage:
    python train_hierarchical.py --data-path your_data.csv --output-dir ./results
"""

import argparse
import torch
from packaging import version
from torch.optim import AdamW
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging
from safetensors.torch import save_file

# Security check: torch.load has known vulnerability mitigations starting in torch >= 2.6
# If users try to load legacy `.pt` weights with older torch builds, require upgrade
# (alternatively use safetensors which are not affected).
# if version.parse(torch.__version__) < version.parse("2.6.0"):
#     raise RuntimeError(
#         f"Detected torch=={torch.__version__}. For safety when loading model weights the project requires "
#         "PyTorch >= 2.6. Either upgrade your PyTorch build (recommended) or use weights serialized with "
#         "`safetensors` (which are safe with older torch versions).\n\n"
#         "To upgrade via conda (recommended on Windows):\n"
#         "  conda install -c pytorch -c nvidia pytorch>=2.6 pytorch-cuda=12.1\n\n"
#         "Or via pip (ensure CUDA-compatible wheel for your drivers):\n"
#         "  pip uninstall -y torch torchvision torchaudio\n"
#         "  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio\n\n"
#         "If you prefer to keep your current torch, convert your model file to `.safetensors` and load that instead."
#     )

from optimized_clinical_bert import (
    ModelConfig,
    OptimizedHierarchicalClinicalBERT,
    HierarchicalTextDataset,
    create_dataloaders,
    FocalLoss,
    get_weighted_loss,
    freeze_encoder,
    unfreeze_encoder
)
from training_utils import (
    HierarchicalTrainer,
    AdvancedScheduler,
    EarlyStopping,
    TrainingMetrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train hierarchical ClinicalBERT')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV with medical notes')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-name', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--max-len', type=int, default=256, help='Max sequence length')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--use-lora', action='store_true', default=True, help='Use LoRA fine-tuning')
    parser.add_argument('--use-focal', action='store_true', default=True, help='Use focal loss')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_and_prepare_data(csv_path: str, test_size: float = 0.2):
    """Load and split data."""
    df = pd.read_csv(csv_path)

    # Expected columns: 'condition_label', 'medical_abstract'
    assert all(col in df.columns for col in ['condition_label', 'medical_abstract']), \
        "CSV must have 'condition_label', 'medical_abstract' columns"

    # Create parent label if missing: parent = 1 (specific) for condition 1-4, parent = 0 (generic) for condition 5
    # Map child labels (specific categories) to 0..3; for generic (condition 5) set to ignore index -100
    IGNORE_INDEX = -100
    df['parent_label'] = df.get('parent_label', df['condition_label'].apply(lambda x: 0 if int(x) == 5 else 1))

    def _map_child_label(x):
        x = int(x)
        if x == 5:
            return IGNORE_INDEX
        return x - 1  # map 1->0,2->1,3->2,4->3

    df['child_label'] = df['condition_label'].apply(_map_child_label)

    texts = df['medical_abstract'].tolist()
    parent_labels = df['parent_label'].astype(int).tolist()
    child_labels = df['child_label'].tolist()
    
    # Stratified split
    train_texts, val_texts, train_parent, val_parent, train_child, val_child = train_test_split(
        texts, parent_labels, child_labels,
        test_size=test_size,
        random_state=42,
        stratify=parent_labels
    )
    
    logger.info(f"Data loaded: {len(train_texts)} train, {len(val_texts)} val")
    return train_texts, val_texts, train_parent, val_parent, train_child, val_child


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    train_texts, val_texts, train_parent, val_parent, train_child, val_child = \
        load_and_prepare_data(args.data_path)
    
    # Initialize model
    logger.info("Initializing model...")
    config = ModelConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        use_gradient_checkpointing=True
    )
    model = OptimizedHierarchicalClinicalBERT(config).to(device)
    logger.info(f"Model parameters: {model.get_trainable_params():,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloaders(
        train_texts, train_parent, train_child, tokenizer,
        batch_size=args.batch_size, max_len=args.max_len,
        num_workers=args.num_workers, shuffle=True
    )
    val_loader = create_dataloaders(
        val_texts, val_parent, val_child, tokenizer,
        batch_size=args.batch_size * 2, max_len=args.max_len,
        num_workers=args.num_workers, shuffle=False
    )
    
    # Setup training
    logger.info("Setting up optimizer and loss functions...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = AdvancedScheduler(optimizer, args.warmup_steps, num_training_steps)
    
    # Loss functions (use per-sample reduction to enable masking for child loss)
    if args.use_focal:
        parent_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        child_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        logger.info("Using Focal Loss (per-sample reduction)")
    else:
        parent_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        child_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        logger.info("Using Cross-Entropy Loss (per-sample reduction)")
    
    # Trainer
    trainer = HierarchicalTrainer(
        model=model,
        device=device,
        parent_loss_fn=parent_loss_fn,
        child_loss_fn=child_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=True,
        parent_weight=0.3,
        child_weight=0.7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=3,
        save_best=True,
        model_path=str(output_dir / 'best_model.safetensors')
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    try:
        metrics = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping=early_stopping,
            verbose=True
        )
        
        # Save
        logger.info("\nSaving results...")
        metrics.save(str(output_dir / 'metrics.json'))
        save_file(model.state_dict(), str(output_dir / 'final_model.safetensors'))
        logger.info(f"Results saved to {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")


if __name__ == '__main__':
    main()
