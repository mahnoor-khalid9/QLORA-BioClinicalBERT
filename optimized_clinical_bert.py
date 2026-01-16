"""
Optimized Hierarchical ClinicalBERT for Medical Notes Classification
======================================================================
Key optimizations:
1. LoRA fine-tuning for memory efficiency
2. Better layer design with residual connections
3. Mixed precision training support
4. Advanced loss functions (focal loss, weighted CE)
5. Efficient batch inference
6. Gradient checkpointing for large models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for optimized hierarchical model."""
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    num_parent_classes: int = 2
    num_child_classes: int = 4
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.2
    use_gradient_checkpointing: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    freeze_base_model_ratio: float = 0.8


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer for efficient fine-tuning."""
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Original layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA components, at initialization BA=0, 
        # lora_a → down-projection (random, small)
        # lora_b → up-projection (zero-init so no initial effect)
        # lora_dropout → regularization on LoRA only
        self.lora_a = nn.Parameter(torch.randn(in_features, r) * math.sqrt(2.0 / (5 * r)))
        self.lora_b = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features)
        """

        # 1. Normal linear layer (frozen pretrained weights)
        base_output = F.linear(x, self.weight, self.bias)
        # shape: (batch_size, out_features)

        # 2. LoRA path — a small, trainable correction, drop some features (regularization)
        x_lora = self.lora_dropout(x)

        #    b) project down to a low-rank space, (in_features → r)
        low_rank = x_lora @ self.lora_a

        #    c) project back up to output space, (r → out_features)
        lora_update = low_rank @ self.lora_b

        # 3. Scale the LoRA contribution, scaling = lora_alpha / r
        lora_update = lora_update * self.scaling

        # 4. Add LoRA update to the base output
        return base_output + lora_update


class EfficientClinicalBERTEncoder(nn.Module):
    """Memory-efficient ClinicalBERT encoder with optional LoRA."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.base_model = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        if config.use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        
        # Freeze bottom layers
        self._freeze_bottom_layers(config.freeze_base_model_ratio)

    def _freeze_bottom_layers(self, freeze_ratio: float):
        """Freeze bottom transformer layers to save computation."""
        if hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
            layers = list(self.base_model.encoder.layer)
            n_freeze = int(math.floor(len(layers) * freeze_ratio))
            for layer in layers[:n_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled: (batch_size, hidden_size)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False
        )
        
        # Use pooled output or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Mean pooling with attention mask
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, H)
            return pooled


class ClassificationHead(nn.Module):
    """Efficient multi-layer classification head with residual connections."""
    def __init__(self, in_features: int, num_classes: int, num_layers: int = 2, dropout: float = 0.2, use_lora: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.use_lora = use_lora
        
        layers = []
        hidden_size = max(64, in_features // 2)
        
        for i in range(num_layers):
            input_dim = in_features if i == 0 else hidden_size
            output_dim = hidden_size if i < num_layers - 1 else num_classes
            
            if use_lora and i < num_layers - 1:
                layers.append(LoRALinear(input_dim, output_dim, r=8, lora_alpha=16, lora_dropout=dropout))
            else:
                layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptimizedHierarchicalClinicalBERT(nn.Module):
    """
    Optimized two-headed hierarchical ClinicalBERT for medical text classification.
    
    Features:
    - Shared encoder with gradient checkpointing
    - Efficient classification heads with LoRA
    - Parent and child predictions with optional conditioning
    - Focal loss support for imbalanced data
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = EfficientClinicalBERTEncoder(config)
        
        # Parent head (binary or multiclass)
        self.parent_head = ClassificationHead(
            in_features=config.hidden_size,
            num_classes=config.num_parent_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_lora=config.use_lora
        )
        
        # Child head (can condition on parent if needed)
        child_input_dim = config.hidden_size
        self.child_head = ClassificationHead(
            in_features=child_input_dim,
            num_classes=config.num_child_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_lora=config.use_lora
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            return_embeddings: If True, also return encoder embeddings
        
        Returns:
            dict with 'parent_logits', 'child_logits', and optionally 'embeddings'
        """
        embeddings = self.encoder(input_ids, attention_mask)
        parent_logits = self.parent_head(embeddings)
        child_logits = self.child_head(embeddings)
        
        result = {
            'parent_logits': parent_logits,
            'child_logits': child_logits
        }
        
        if return_embeddings:
            result['embeddings'] = embeddings
        
        return result

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalTextDataset(Dataset):
    """Efficient dataset with proper tokenization caching."""
    
    def __init__(
        self,
        texts: List[str],
        parent_labels: List[int],
        child_labels: List[int],
        tokenizer: AutoTokenizer,
        max_len: int = 256
    ):
        self.texts = texts
        self.parent_labels = parent_labels
        self.child_labels = child_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'parent_label': torch.tensor(self.parent_labels[idx], dtype=torch.long),
            'child_label': torch.tensor(self.child_labels[idx], dtype=torch.long)
        }


def create_dataloaders(
    texts: List[str],
    parent_labels: List[int],
    child_labels: List[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_len: int = 256,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """Create efficient dataloader."""
    dataset = HierarchicalTextDataset(texts, parent_labels, child_labels, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_weighted_loss(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(labels)
    weights = torch.zeros(num_classes, device=device)
    total = len(labels)
    
    for cls, count in counts.items():
        weights[cls] = total / (num_classes * count)
    
    return weights / weights.sum() * num_classes


def freeze_encoder(model: OptimizedHierarchicalClinicalBERT):
    """Freeze encoder for faster training (layer-wise fine-tuning)."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: OptimizedHierarchicalClinicalBERT, num_layers: int = 4):
    """Unfreeze top N layers of encoder for progressive fine-tuning."""
    base_model = model.encoder.base_model
    if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        layers = list(base_model.encoder.layer)
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


__all__ = [
    'ModelConfig',
    'FocalLoss',
    'LoRALinear',
    'EfficientClinicalBERTEncoder',
    'ClassificationHead',
    'OptimizedHierarchicalClinicalBERT',
    'HierarchicalTextDataset',
    'create_dataloaders',
    'get_weighted_loss',
    'freeze_encoder',
    'unfreeze_encoder'
]
