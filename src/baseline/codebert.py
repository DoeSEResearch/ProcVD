#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
import warnings
import math
import gc
import pickle
import shutil
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Optional experiment tracking tools
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Environment setup
os.environ.update({
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_NO_ADVISORY_WARNINGS': 'true'
})

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')


@dataclass
class CodeBertConfig:
    """Enhanced CodeBERT configuration class"""

    # Basic config
    model_name: str = "microsoft/codebert-base"
    hf_token: str = field(default_factory=lambda: os.environ.get('HF_TOKEN', 'YOUR HUGGINGFACE API KEY'))

    # Dataset config
    dataset_type: str = "unbalanced"  # balanced, unbalanced
    data_dir: str = "../../data/primevul_process"
    cache_dir: str = "../../result/codebert_cache"

    # Model config
    max_seq_length: int = 512
    hidden_size: int = 768
    num_labels: int = 2
    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Training config
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    num_epochs: int = 5
    warmup_ratio: float = 0.1
    scheduler_type: str = "linear"  # linear, cosine

    # Optimization config
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4

    # Early stopping and saving
    patience: int = 3
    min_delta: float = 0.001
    save_total_limit: int = 3
    save_strategy: str = "epoch"  # epoch, steps
    evaluation_strategy: str = "epoch"  # epoch, steps

    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    project_name: str = "codebert-vulnerability-detection"
    run_name: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Path config
    output_dir: str = "../../result/codebert_outputs"
    logging_dir: str = "../../result/codebert_logs"
    cache_dir: str = "../../result/codebert_cache"

    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1

    # Loss function config
    focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # Multi-GPU and distributed training
    local_rank: int = -1
    world_size: int = 1

    # Other
    seed: int = 42
    log_level: str = "INFO"
    save_predictions: bool = True

    def __post_init__(self):
        # Create necessary directories
        for path in [self.output_dir, self.logging_dir, self.cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

        # Set data file paths
        data_dir = Path(self.data_dir)
        if self.dataset_type == "balanced":
            self.train_file = data_dir / "PrimeVul_balanced_train.jsonl"
            self.valid_file = data_dir / "PrimeVul_balanced_valid.jsonl"
            self.test_file = data_dir / "PrimeVul_balanced_test.jsonl"
        else:
            self.train_file = data_dir / "PrimeVul_unbalanced_train_sampled.jsonl"
            self.valid_file = data_dir / "PrimeVul_unbalanced_valid_sampled.jsonl"
            self.test_file = data_dir / "PrimeVul_unbalanced_test_sampled.jsonl"

        # Experiment name
        self.experiment_name = f"codebert_{self.dataset_type}_{self.run_name}"


class Logger:
    """Enhanced logging manager"""

    def __init__(self, config: CodeBertConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = Path(self.config.logging_dir) / f"{self.config.experiment_name}.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)


class MemoryManager:
    """Memory management utilities"""

    @staticmethod
    @contextmanager
    def memory_cleanup():
        """Context manager for automatic memory cleanup"""
        try:
            yield
        finally:
            MemoryManager.aggressive_cleanup()

    @staticmethod
    def aggressive_cleanup():
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    @staticmethod
    def log_memory_usage(device: torch.device, step_name: str = "", logger=None):
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3

            message = (f"[{step_name}] GPU Memory - "
                       f"Allocated: {allocated:.2f}GB, "
                       f"Reserved: {reserved:.2f}GB, "
                       f"Max: {max_allocated:.2f}GB")

            if logger:
                logger.info(message)
            else:
                print(message)

            return allocated, reserved, max_allocated
        return 0, 0, 0


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DataAugmentation:
    """Code data augmentation utilities"""

    def __init__(self, config: CodeBertConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.augmentation_prob = config.augmentation_prob

    def augment_code(self, code: str) -> str:
        """Simple code augmentation strategy"""
        if np.random.random() > self.augmentation_prob:
            return code

        # TODO: Implement more sophisticated augmentation strategies
        # e.g., variable renaming, adding comments, formatting, etc.
        return code


class OptimizedPrimeVulDataset(Dataset):
    """Optimized PrimeVul dataset class"""

    def __init__(self, data_file: Path, tokenizer, config: CodeBertConfig,
                 logger: Logger, is_training: bool = True):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.is_training = is_training

        # Data augmentation
        self.augmentation = DataAugmentation(config, tokenizer) if config.use_data_augmentation else None

        # Load or cache data
        self.data = self._load_or_cache_data(data_file)
        self._log_distribution()

    def _load_or_cache_data(self, data_file: Path) -> List[Dict]:
        """Load or use cached data"""
        cache_file = Path(self.config.cache_dir) / f"{data_file.stem}_cached.pkl"

        if cache_file.exists():
            self.logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        self.logger.info(f"First time loading data: {data_file}")
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)

        # Cache data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        return data

    def _log_distribution(self):
        """Log data distribution"""
        label_dist = defaultdict(int)
        for item in self.data:
            label_dist[item.get('target', 0)] += 1

        self.logger.info(f"Dataset samples: {len(self.data)}")
        self.logger.info(f"Label distribution: {dict(label_dist)}")

        # Validate paired structure for balanced dataset
        if self.config.dataset_type == "balanced":
            self._validate_paired_structure()

        # Calculate class weights
        total = len(self.data)
        class_weights = {}
        for label, count in label_dist.items():
            class_weights[label] = total / (len(label_dist) * count)

        self.logger.info(f"Suggested class weights: {class_weights}")

    def _validate_paired_structure(self):
        """Validate paired structure of balanced dataset"""
        pairs_found = 0
        valid_pairs = 0

        for i in range(0, len(self.data) - 1, 2):
            if i + 1 < len(self.data):
                pairs_found += 1
                label1 = self.data[i].get('target', 0)
                label2 = self.data[i + 1].get('target', 0)

                # Check for valid pair structure (one vulnerable, one benign)
                if (label1 == 1 and label2 == 0) or (label1 == 0 and label2 == 1):
                    valid_pairs += 1

        self.logger.info(f"Balanced dataset pair validation: {valid_pairs}/{pairs_found} pairs valid")
        if valid_pairs != pairs_found:
            self.logger.warning("Dataset pair structure incomplete, may affect pair-wise metrics")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get code and label
        func_code = item.get('func', '')
        label = item.get('target', 0)

        # Data augmentation
        if self.is_training and self.augmentation:
            func_code = self.augmentation.augment_code(func_code)

        # Tokenization
        encoding = self.tokenizer(
            func_code,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'idx': item.get('idx', idx)
        }


class EnhancedCodeBertClassifier(nn.Module):
    """Enhanced CodeBERT classifier"""

    def __init__(self, config: CodeBertConfig):
        super().__init__()
        self.config = config

        # Load pretrained model config
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True
        )

        # Update config
        model_config.hidden_dropout_prob = config.hidden_dropout_prob
        model_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        # Load CodeBERT model
        self.codebert = AutoModel.from_pretrained(
            config.model_name,
            config=model_config,
            token=config.hf_token,
            trust_remote_code=True
        )

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize classifier
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # CodeBERT encoding
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get pooled output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use [CLS] token hidden state
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.focal_loss:
                loss_fn = FocalLoss(
                    alpha=self.config.focal_alpha,
                    gamma=self.config.focal_gamma
                )
                loss = loss_fn(logits, labels)
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
                loss = loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }


class MetricsCalculator:
    """Metrics calculator"""

    @staticmethod
    def calculate_comprehensive_metrics(preds: List[int], labels: List[int],
                                        probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(labels, preds, output_dict=True)
        }

        # Probability-based metrics
        if probs is not None:
            try:
                # ROC AUC
                if len(np.unique(labels)) > 1:
                    if probs.shape[1] == 2:
                        auc = roc_auc_score(labels, probs[:, 1])
                        ap = average_precision_score(labels, probs[:, 1])
                        metrics['roc_auc'] = auc
                        metrics['average_precision'] = ap

                # VD-S metric
                vds = MetricsCalculator.calculate_vds(labels, probs)
                metrics['vd_s'] = vds

            except Exception as e:
                print(f"Error calculating probability metrics: {e}")

        return metrics

    @staticmethod
    def calculate_vds(labels: List[int], probs: np.ndarray, fpr_threshold: float = 0.005):
        """Calculate VD-S: FNR @ (FPR ≤ threshold)"""
        if probs.shape[1] != 2:
            return None

        vuln_probs = probs[:, 1]
        thresholds = np.unique(vuln_probs)
        best_fnr = 1.0

        for threshold in thresholds:
            preds = (vuln_probs >= threshold).astype(int)

            tn = sum((1 - preds[i]) * (1 - labels[i]) for i in range(len(labels)))
            fp = sum(preds[i] * (1 - labels[i]) for i in range(len(labels)))
            fn = sum((1 - preds[i]) * labels[i] for i in range(len(labels)))
            tp = sum(preds[i] * labels[i] for i in range(len(labels)))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            if fpr <= fpr_threshold:
                best_fnr = min(best_fnr, fnr)

        return best_fnr

    @staticmethod
    def calculate_paired_metrics(preds: List[int], labels: List[int],
                                 logger=None) -> Dict[str, float]:
        """
        Calculate pair-wise prediction metrics (for balanced dataset only)

        Returns:
            Dict containing P-C, P-V, P-B, P-R metrics
        """
        if len(preds) != len(labels):
            raise ValueError("Prediction and label lists length mismatch")

        if len(preds) % 2 != 0:
            if logger:
                logger.warning(f"Sample count is odd ({len(preds)}), ignoring last sample")

        # Initialize counters
        pair_counts = {
            'P-C': 0,  # Pair-wise Correct Prediction
            'P-V': 0,  # Pair-wise Vulnerable Prediction
            'P-B': 0,  # Pair-wise Benign Prediction
            'P-R': 0  # Pair-wise Reversed Prediction
        }

        total_pairs = 0
        valid_pairs = 0

        # Process samples in pairs
        for i in range(0, len(preds) - 1, 2):
            if i + 1 >= len(preds):
                break

            total_pairs += 1

            pred1, label1 = preds[i], labels[i]
            pred2, label2 = preds[i + 1], labels[i + 1]

            # Validate valid pair structure (one vulnerable, one benign)
            if not ((label1 == 1 and label2 == 0) or (label1 == 0 and label2 == 1)):
                if logger:
                    logger.debug(f"Pair {i // 2 + 1} invalid structure: labels=({label1}, {label2})")
                continue

            valid_pairs += 1

            # Determine which is vulnerable and which is benign
            if label1 == 1 and label2 == 0:
                vuln_pred, vuln_label = pred1, label1
                benign_pred, benign_label = pred2, label2
            else:
                vuln_pred, vuln_label = pred2, label2
                benign_pred, benign_label = pred1, label1

            # Calculate pair-wise metrics
            if vuln_pred == vuln_label and benign_pred == benign_label:
                pair_counts['P-C'] += 1
            elif vuln_pred == 1 and benign_pred == 1:
                pair_counts['P-V'] += 1
            elif vuln_pred == 0 and benign_pred == 0:
                pair_counts['P-B'] += 1
            elif vuln_pred != vuln_label and benign_pred != benign_label:
                pair_counts['P-R'] += 1

        # Log statistics
        if logger:
            logger.info(f"Pair-wise metrics stats: total_pairs={total_pairs}, valid_pairs={valid_pairs}")
            if valid_pairs > 0:
                logger.info(f"P-C={pair_counts['P-C']}, P-V={pair_counts['P-V']}, "
                            f"P-B={pair_counts['P-B']}, P-R={pair_counts['P-R']}")

        # Calculate ratios
        pair_metrics = {}
        if valid_pairs > 0:
            for key, count in pair_counts.items():
                pair_metrics[key] = count / valid_pairs
        else:
            for key in pair_counts.keys():
                pair_metrics[key] = 0.0
            if logger:
                logger.warning("No valid pairs found, pair-wise metrics set to 0")

        # Add raw counts for debugging
        pair_metrics['pair_counts'] = pair_counts
        pair_metrics['total_pairs'] = total_pairs
        pair_metrics['valid_pairs'] = valid_pairs

        return pair_metrics


class ExperimentTracker:
    """Experiment tracker"""

    def __init__(self, config: CodeBertConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.wandb_run = None
        self.tensorboard_writer = None

        self._init_trackers()

    def _init_trackers(self):
        """Initialize experiment tracking tools"""
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.project_name,
                    name=self.config.run_name,
                    config=self.config.__dict__
                )
                self.logger.info("W&B initialized successfully")
            except Exception as e:
                self.logger.warning(f"W&B initialization failed: {e}")

        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=Path(self.config.logging_dir) / "tensorboard" / self.config.run_name
                )
                self.logger.info("TensorBoard initialized successfully")
            except Exception as e:
                self.logger.warning(f"TensorBoard initialization failed: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics"""
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                formatted_metrics[f"{prefix}/{key}" if prefix else key] = value

        if self.wandb_run:
            self.wandb_run.log(formatted_metrics, step=step)

        if self.tensorboard_writer:
            for key, value in formatted_metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)

    def log_confusion_matrix(self, cm: np.ndarray, step: int, prefix: str = ""):
        """Log confusion matrix"""
        if self.tensorboard_writer:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{prefix} Confusion Matrix')

            self.tensorboard_writer.add_figure(f'{prefix}/confusion_matrix', fig, step)
            plt.close(fig)

    def finish(self):
        """Finish experiment tracking"""
        if self.wandb_run:
            self.wandb_run.finish()

        if self.tensorboard_writer:
            self.tensorboard_writer.close()


class ModelManager:
    """Model manager"""

    def __init__(self, config: CodeBertConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.checkpoints = []

    def save_checkpoint(self, model: nn.Module, tokenizer, optimizer, scheduler,
                        metrics: Dict, epoch: int, step: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build checkpoint info
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self.config.__dict__,
            'is_best': is_best
        }

        # Save path
        if is_best:
            checkpoint_path = checkpoint_dir / "best_model.pt"
            self.logger.info(f"Saving best model: {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"

        try:
            torch.save(checkpoint, checkpoint_path)

            # Save tokenizer
            tokenizer_path = checkpoint_path.parent / f"{checkpoint_path.stem}_tokenizer"
            tokenizer.save_pretrained(tokenizer_path)

            # Manage checkpoint count
            self._manage_checkpoints(checkpoint_path, is_best)

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return None

    def _manage_checkpoints(self, new_checkpoint: Path, is_best: bool):
        """Manage checkpoint count"""
        if is_best:
            return

        self.checkpoints.append(new_checkpoint)

        # Maintain checkpoint limit
        if len(self.checkpoints) > self.config.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                # Delete corresponding tokenizer
                tokenizer_path = old_checkpoint.parent / f"{old_checkpoint.stem}_tokenizer"
                if tokenizer_path.exists():
                    shutil.rmtree(tokenizer_path)
                self.logger.info(f"Deleted old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: Path, model: nn.Module,
                        optimizer=None, scheduler=None):
        """Load checkpoint"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.logger.info(f"Checkpoint loaded, epoch: {checkpoint.get('epoch', 'unknown')}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise


class OptimizedTrainer:
    """Optimized trainer"""

    def __init__(self, config: CodeBertConfig, logger: Logger,
                 experiment_tracker: ExperimentTracker, model_manager: ModelManager):
        self.config = config
        self.logger = logger
        self.tracker = experiment_tracker
        self.model_manager = model_manager

        # Setup device
        self.device = self._setup_device()

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.fp16 else None

        # Training state
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0

    def _setup_device(self) -> torch.device:
        """Setup device"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.local_rank}" if self.config.local_rank >= 0 else "cuda")
            self.logger.info(f"Using GPU: {device}")

            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.95)

        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")

        return device

    def _setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        self.logger.info("Initializing model and tokenizer...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = EnhancedCodeBertClassifier(self.config)
        self.model.to(self.device)

        # Print parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

        MemoryManager.log_memory_usage(self.device, "After model init", self.logger)

    def _setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and scheduler"""
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon
        )

        # Scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        self.logger.info(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    def _prepare_batch_for_model(self, batch):
        """Prepare batch data, filter model-needed parameters"""
        model_inputs = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch:
                model_inputs[key] = batch[key]
        return model_inputs

    def _log_detailed_metrics(self, metrics: Dict[str, Any], split_name: str):
        """Log detailed metrics, especially pair-wise metrics"""
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
        advanced_metrics = ['roc_auc', 'average_precision', 'vd_s']
        pair_metrics = ['P-C', 'P-V', 'P-B', 'P-R']

        # Basic metrics
        basic_info = []
        for metric in basic_metrics:
            if metric in metrics:
                basic_info.append(f"{metric.upper()}: {metrics[metric]:.4f}")

        self.logger.info(f"{split_name} - " + ", ".join(basic_info))

        # Advanced metrics
        advanced_info = []
        for metric in advanced_metrics:
            if metric in metrics:
                advanced_info.append(f"{metric.upper()}: {metrics[metric]:.4f}")

        if advanced_info:
            self.logger.info(f"{split_name} - " + ", ".join(advanced_info))

        # Pair-wise metrics (balanced dataset only)
        if self.config.dataset_type == "balanced":
            pair_info = []
            for metric in pair_metrics:
                if metric in metrics:
                    pair_info.append(f"{metric}: {metrics[metric]:.4f}")

            if pair_info:
                self.logger.info(f"{split_name} - Pair-wise Metrics: " + ", ".join(pair_info))

                # Log detailed pair statistics
                if 'valid_pairs' in metrics:
                    self.logger.info(
                        f"{split_name} - Valid pairs: {metrics['valid_pairs']}/{metrics.get('total_pairs', 'N/A')}")

    def train(self, train_dataset, val_dataset, test_dataset=None):
        """Main training loop"""
        # Setup model and tokenizer
        self._setup_model_and_tokenizer()

        # Data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
            drop_last=False
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )

        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(train_dataloader)

        self.logger.info("Starting training...")
        self.logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        self.logger.info(f"Steps per epoch: {len(train_dataloader)}")

        if self.config.dataset_type == "balanced":
            self.logger.info("Balanced dataset mode, will calculate pair-wise metrics")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Train one epoch
            train_metrics = self._train_epoch(train_dataloader, epoch)

            # Validation
            val_metrics = self._evaluate(val_dataloader, "validation")

            # Log metrics
            self.tracker.log_metrics(train_metrics, epoch, "train")
            self.tracker.log_metrics(val_metrics, epoch, "validation")

            # Log confusion matrix
            if 'confusion_matrix' in val_metrics:
                self.tracker.log_confusion_matrix(
                    np.array(val_metrics['confusion_matrix']), epoch, "validation"
                )

            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['f1']
                self.patience_counter = 0
                self.logger.info(f"New best F1 score: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            checkpoint_path = self.model_manager.save_checkpoint(
                self.model, self.tokenizer, self.optimizer, self.scheduler,
                val_metrics, epoch + 1, self.global_step, is_best
            )

            # Early stopping check
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered, no improvement for {self.patience_counter} epochs")
                break

            # Memory cleanup
            MemoryManager.aggressive_cleanup()

        # Test best model
        if test_dataset is not None:
            self.logger.info("Testing best model...")
            test_metrics = self._test_best_model(test_dataset)
            self.tracker.log_metrics(test_metrics, 0, "test")
            return test_metrics

        return val_metrics

    def _train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Prepare model inputs
            model_inputs = self._prepare_batch_for_model(batch)

            with autocast() if self.config.fp16 else torch.no_grad():
                # Forward pass
                outputs = self.model(**model_inputs)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                MemoryManager.aggressive_cleanup()

        avg_loss = total_loss / num_batches
        self.logger.info(f"Training complete, average loss: {avg_loss:.4f}")

        return {'loss': avg_loss, 'learning_rate': self.scheduler.get_last_lr()[0]}

    def _evaluate(self, dataloader, split_name="validation"):
        """Evaluate model"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Prepare model inputs
                model_inputs = self._prepare_batch_for_model(batch)

                with autocast() if self.config.fp16 else torch.no_grad():
                    outputs = self.model(**model_inputs)

                if outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()

                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        all_probs = np.array(all_probs)
        metrics = MetricsCalculator.calculate_comprehensive_metrics(
            all_preds, all_labels, all_probs
        )

        # Add loss
        metrics['loss'] = total_loss / len(dataloader)

        # Pair-wise metrics (balanced dataset only)
        if self.config.dataset_type == "balanced":
            pair_metrics = MetricsCalculator.calculate_paired_metrics(
                all_preds, all_labels, self.logger
            )
            metrics.update(pair_metrics)

        # Log detailed metrics
        self._log_detailed_metrics(metrics, split_name)

        return metrics

    def _test_best_model(self, test_dataset):
        """Test best model"""
        # Load best model
        best_model_path = Path(self.config.output_dir) / "checkpoints" / "best_model.pt"

        if not best_model_path.exists():
            self.logger.warning("Best model not found, using current model for testing")
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=True
            )
            return self._evaluate(test_dataloader, "test")

        # Load best model
        checkpoint = self.model_manager.load_checkpoint(best_model_path, self.model)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )

        test_metrics = self._evaluate(test_dataloader, "test")

        # Detailed final test report
        self.logger.info("=" * 80)
        self.logger.info("Final test results:")
        self.logger.info("=" * 80)

        # Basic metrics
        self.logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"Recall: {test_metrics['recall']:.4f}")
        self.logger.info(f"F1 Score: {test_metrics['f1']:.4f}")

        # Advanced metrics
        if 'roc_auc' in test_metrics:
            self.logger.info(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
        if 'average_precision' in test_metrics:
            self.logger.info(f"Average Precision: {test_metrics['average_precision']:.4f}")
        if 'vd_s' in test_metrics:
            self.logger.info(f"VD-S: {test_metrics['vd_s']:.4f}")

        # Pair-wise metrics (balanced dataset only)
        if self.config.dataset_type == "balanced":
            self.logger.info("-" * 40)
            self.logger.info("Pair-wise prediction metrics:")
            if 'P-C' in test_metrics:
                self.logger.info(f"P-C (Pair-wise Correct): {test_metrics['P-C']:.4f}")
            if 'P-V' in test_metrics:
                self.logger.info(f"P-V (Pair-wise Vulnerable): {test_metrics['P-V']:.4f}")
            if 'P-B' in test_metrics:
                self.logger.info(f"P-B (Pair-wise Benign): {test_metrics['P-B']:.4f}")
            if 'P-R' in test_metrics:
                self.logger.info(f"P-R (Pair-wise Reversed): {test_metrics['P-R']:.4f}")

            if 'valid_pairs' in test_metrics:
                self.logger.info(f"Valid pairs: {test_metrics['valid_pairs']}/{test_metrics.get('total_pairs', 'N/A')}")

        self.logger.info("=" * 80)

        return test_metrics


def set_seed(seed: int):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Optimized CodeBERT vulnerability detection system")

    # Basic parameters
    parser.add_argument("--dataset_type", type=str, choices=["balanced", "unbalanced"],
                        default="balanced", help="Dataset type")
    parser.add_argument("--data_dir", type=str,
                        default="/data/bowen/data/primevul_process", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base",
                        help="Pretrained model name")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Optimization parameters
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")

    # Loss function
    parser.add_argument("--focal_loss", action="store_true", help="Use Focal Loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing")

    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use W&B tracking")
    parser.add_argument("--use_tensorboard", action="store_true", default=True,
                        help="Use TensorBoard")
    parser.add_argument("--project_name", type=str,
                        default="codebert-vulnerability-detection", help="Project name")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--data_augmentation", action="store_true", help="Enable data augmentation")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create config
    config = CodeBertConfig(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        project_name=args.project_name,
        seed=args.seed,
        patience=args.patience,
        use_data_augmentation=args.data_augmentation
    )

    # Initialize components
    logger = Logger(config)
    experiment_tracker = ExperimentTracker(config, logger)
    model_manager = ModelManager(config, logger)
    trainer = OptimizedTrainer(config, logger, experiment_tracker, model_manager)

    logger.info("=" * 80)
    logger.info(f"Starting optimized CodeBERT vulnerability detection training - {args.dataset_type} dataset")
    logger.info("=" * 80)

    try:
        # Initialize tokenizer (temporary for dataset creation)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create datasets
        logger.info("Loading datasets...")
        with MemoryManager.memory_cleanup():
            train_dataset = OptimizedPrimeVulDataset(
                config.train_file, tokenizer, config, logger, is_training=True
            )
            val_dataset = OptimizedPrimeVulDataset(
                config.valid_file, tokenizer, config, logger, is_training=False
            )
            test_dataset = OptimizedPrimeVulDataset(
                config.test_file, tokenizer, config, logger, is_training=False
            )

        # Start training
        results = trainer.train(train_dataset, val_dataset, test_dataset)

        logger.info("Training completed!")
        logger.info(f"Final test F1: {results.get('f1', 'N/A'):.4f}")

        if config.dataset_type == "balanced" and 'P-C' in results:
            logger.info(f"Final test P-C: {results.get('P-C', 'N/A'):.4f}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Cleanup resources
        experiment_tracker.finish()
        MemoryManager.aggressive_cleanup()


if __name__ == "__main__":
    main()