"""
Training utilities and callbacks for optimized hierarchical ClinicalBERT
"""

import logging
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
from typing import Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import json
from safetensors.torch import save_file

# Module logger and helper
logger = logging.getLogger(__name__)

def configure_logging(level=logging.INFO):
    """Configure simple logging if no handlers are present (non-intrusive)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(level)


@dataclass
class TrainingMetrics:
    """Track training and evaluation metrics."""
    train_losses: List[float] = field(default_factory=list)
    parent_losses: List[float] = field(default_factory=list)
    child_losses: List[float] = field(default_factory=list)
    val_accuracies_parent: List[float] = field(default_factory=list)
    val_accuracies_child: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    def save(self, path: str):
        """Save metrics to JSON, converting float32 to standard float."""
        
        # Helper function to ensure all list items are standard floats
        # This handles cases where items are float32, float64, numpy.float, or PyTorch tensors
        def convert_to_float_list(data_list):
            return [float(x) for x in data_list]

        with open(path, 'w') as f:
            # CRITICAL FIX: Use the convert_to_float_list helper function 
            # on every list before dumping to JSON.
            json.dump({
                'train_losses': convert_to_float_list(self.train_losses),
                'parent_losses': convert_to_float_list(self.parent_losses),
                'child_losses': convert_to_float_list(self.child_losses),
                'val_accuracies_parent': convert_to_float_list(self.val_accuracies_parent),
                'val_accuracies_child': convert_to_float_list(self.val_accuracies_child),
                'learning_rates': convert_to_float_list(self.learning_rates),
            }, f, indent=2)

    # def save(self, path: str):
    #     """Save metrics to JSON."""
    #     with open(path, 'w') as f:
    #         json.dump({
    #             'train_losses': self.train_losses,
    #             'parent_losses': self.parent_losses,
    #             'child_losses': self.child_losses,
    #             'val_accuracies_parent': self.val_accuracies_parent,
    #             'val_accuracies_child': self.val_accuracies_child,
    #             'learning_rates': self.learning_rates,
    #         }, f, indent=2)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, save_best: bool = True, model_path: str = "best_model.safetensors"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_best = save_best
        self.model_path = model_path
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.save_best:
                # torch.save(model.state_dict(), self.model_path)
                save_file(model.state_dict(), self.model_path)
                logging.getLogger(__name__).info(f"✓ Best model saved at epoch {epoch + 1} with loss {val_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.getLogger(__name__).warning(f"⚠ Early stopping at epoch {epoch + 1}. Best loss: {self.best_loss:.4f} at epoch {self.best_epoch + 1}")
                return True
        return False


class AdvancedScheduler:
    """Combine warmup + cosine annealing for better convergence."""
    
    def __init__(self, optimizer: AdamW, num_warmup_steps: int, num_training_steps: int):
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(10, (num_training_steps - num_warmup_steps) // 3),
            T_mult=1,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps]
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute accuracy."""
    return (preds == labels).astype(np.float32).mean()


# Ignore index used for child labels that should not contribute to child loss/metrics
IGNORE_INDEX = -100


def compute_metrics(
    parent_preds: np.ndarray,
    parent_labels: np.ndarray,
    child_preds: np.ndarray,
    child_labels: np.ndarray
) -> Dict[str, float]:
    """Compute all metrics, masking child metrics where child_labels == IGNORE_INDEX."""
    parent_acc = compute_accuracy(parent_preds, parent_labels)

    # Mask invalid child labels
    valid = (child_labels != IGNORE_INDEX)
    if valid.any():
        child_acc = compute_accuracy(child_preds[valid], child_labels[valid])
    else:
        child_acc = 0.0

    return {
        'parent_acc': parent_acc,
        'child_acc': child_acc,
    }


class HierarchicalTrainer:
    """Unified trainer for hierarchical classification."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        parent_loss_fn: nn.Module,
        child_loss_fn: nn.Module,
        optimizer: AdamW,
        scheduler: Optional[AdvancedScheduler] = None,
        metrics: Optional[TrainingMetrics] = None,
        use_amp: bool = True,
        parent_weight: float = 0.3,
        child_weight: float = 0.7,
        log_interval: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.device = device
        self.parent_loss_fn = parent_loss_fn
        self.child_loss_fn = child_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or TrainingMetrics()
        self.use_amp = use_amp and torch.cuda.is_available()
        self.parent_weight = parent_weight
        self.child_weight = child_weight
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)

        # Log basic trainer state
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except Exception:
            total_params = None
        self.logger.info(f"Trainer initialized (device={device}, use_amp={self.use_amp}, log_interval={self.log_interval}, total_params={total_params})")
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast_context = torch.cuda.amp.autocast
        else:
            self.autocast_context = lambda: nullcontext()

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch with added logging and basic timing."""
        self.model.train()
        total_loss = 0.0
        parent_loss_total = 0.0
        child_loss_total = 0.0

        start_time = time.perf_counter()
        len_loader = len(train_loader) if hasattr(train_loader, '__len__') else None

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            parent_labels = batch['parent_label'].to(self.device)
            child_labels = batch['child_label'].to(self.device)

            self.optimizer.zero_grad()

            with self.autocast_context():
                outputs = self.model(input_ids, attention_mask)
                parent_logits = outputs['parent_logits']
                child_logits = outputs['child_logits']

                # Parent loss: per-sample then mean
                parent_loss_per = self.parent_loss_fn(parent_logits, parent_labels)
                parent_loss = parent_loss_per.mean() if parent_loss_per.dim() > 0 else parent_loss_per

                # Child loss: compute per-sample and mask out IGNORE_INDEX entries
                child_loss_per = self.child_loss_fn(child_logits, child_labels)
                mask = (child_labels != IGNORE_INDEX)
                if child_loss_per.dim() > 0:
                    if mask.any():
                        child_loss = child_loss_per[mask].mean()
                    else:
                        child_loss = torch.tensor(0.0, device=self.device)
                else:
                    child_loss = child_loss_per

                loss = self.parent_weight * parent_loss + self.child_weight * child_loss

            # Backward + step (capture grad norm)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            parent_loss_total += parent_loss.item()
            child_loss_total += child_loss.item()

            # Log periodic batch-level information
            lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
            if (batch_idx + 1) % self.log_interval == 0 or (len_loader and batch_idx == len_loader - 1):
                self.logger.info(
                    f"[Batch {batch_idx+1}/{len_loader or '?'}] loss={loss.item():.4f} parent={parent_loss.item():.4f} "
                    f"child={child_loss.item():.4f} grad_norm={grad_norm:.4f} lr={lr:.2e}"
                )

        avg_loss = total_loss / (len_loader or 1)
        elapsed = time.perf_counter() - start_time
        self.metrics.train_losses.append(avg_loss)
        self.metrics.parent_losses.append(parent_loss_total / (len_loader or 1))
        self.metrics.child_losses.append(child_loss_total / (len_loader or 1))
        
        if self.scheduler:
            try:
                self.metrics.learning_rates.append(self.scheduler.get_last_lr()[0])
            except Exception:
                pass

        # Epoch summary
        self.logger.info(f"Epoch completed: avg_loss={avg_loss:.4f}, parent_loss={parent_loss_total / (len_loader or 1):.4f}, child_loss={child_loss_total / (len_loader or 1):.4f}, time={elapsed:.2f}s")

        return avg_loss

    def evaluate(self, val_loader) -> tuple:
        """Evaluate on validation set with logging and timing."""
        self.model.eval()
        parent_preds_list = []
        parent_labels_list = []
        child_preds_list = []
        child_labels_list = []

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                parent_labels = batch['parent_label'].to(self.device)
                child_labels = batch['child_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                parent_logits = outputs['parent_logits']
                child_logits = outputs['child_logits']

                parent_preds = torch.argmax(parent_logits, dim=1)
                child_preds = torch.argmax(child_logits, dim=1)

                parent_preds_list.append(parent_preds.cpu().numpy())
                parent_labels_list.append(parent_labels.cpu().numpy())
                child_preds_list.append(child_preds.cpu().numpy())
                child_labels_list.append(child_labels.cpu().numpy())

                if (batch_idx + 1) % max(1, self.log_interval) == 0:
                    self.logger.debug(f"Eval batch {batch_idx + 1}")

        parent_preds = np.concatenate(parent_preds_list)
        parent_labels = np.concatenate(parent_labels_list)
        child_preds = np.concatenate(child_preds_list)
        child_labels = np.concatenate(child_labels_list)

        metrics = compute_metrics(parent_preds, parent_labels, child_preds, child_labels)
        self.metrics.val_accuracies_parent.append(metrics['parent_acc'])
        self.metrics.val_accuracies_child.append(metrics['child_acc'])

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"Eval completed: parent_acc={metrics['parent_acc']:.4f}, child_acc={metrics['child_acc']:.4f}, time={elapsed:.2f}s")

        return metrics, parent_preds, parent_labels, child_preds, child_labels

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        verbose: bool = True
    ) -> TrainingMetrics:
        """Full training loop."""
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            metrics, *_ = self.evaluate(val_loader)

            # Summary
            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Parent Acc: {metrics['parent_acc']:.4f} - Child Acc: {metrics['child_acc']:.4f}")
            if self.scheduler:
                try:
                    self.logger.info(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
                except Exception:
                    pass

            if early_stopping:
                val_loss = train_loss  # Could use a better metric
                if early_stopping(val_loss, self.model, epoch):
                    break

        return self.metrics


# Null context for non-AMP
from contextlib import nullcontext


__all__ = [
    'TrainingMetrics',
    'EarlyStopping',
    'AdvancedScheduler',
    'compute_accuracy',
    'compute_metrics',
    'HierarchicalTrainer'
]
