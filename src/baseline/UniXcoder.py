#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniXcoder vulnerability detection script for PrimeVul dataset
Supports balanced and unbalanced datasets with complete training and evaluation pipeline
"""

import os
import json
import logging
import argparse
import warnings
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from datetime import datetime

# Set environment variables
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '0',
    'HF_TOKEN': 'YOUR HUGGINGFACE API KEY',
    'TOKENIZERS_PARALLELISM': 'false'
})


# Setup logging
def setup_logging(log_file: str = "unixcoder_training.log") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    warnings.filterwarnings("ignore")
    return logger


logger = setup_logging()


class UniXcoderConfig:
    """UniXcoder configuration class"""

    def __init__(self, dataset_type: str = "balanced"):
        # HuggingFace configuration
        self.hf_token = os.environ.get('HF_TOKEN', 'YOUR HUGGINGFACE API KEY')
        self.model_name = "microsoft/unixcoder-base"

        # Dataset configuration
        self.dataset_type = dataset_type
        self.data_dir = Path("../../data/primevul_process")

        if dataset_type == "balanced":
            self.train_file = self.data_dir / "PrimeVul_balanced_train.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_balanced_valid.jsonl"
            self.test_file = self.data_dir / "PrimeVul_balanced_test.jsonl"
        else:
            self.train_file = self.data_dir / "PrimeVul_unbalanced_train_sampled.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_unbalanced_valid_sampled.jsonl"
            self.test_file = self.data_dir / "PrimeVul_unbalanced_test_sampled.jsonl"

        # Model configuration
        self.max_seq_length = 512
        self.hidden_size = 768
        self.num_labels = 2
        self.dropout = 0.1

        # Training configuration
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 5
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        self.fp16 = True
        self.fp16_full_eval = True
        self.patience = 2
        self.min_delta = 0.001

        # Save configuration
        self.save_dir = Path("./checkpoints")
        self.best_model_dir = self.save_dir / "best_models"
        self.results_dir = self.save_dir / "results"

        # Create save directories
        for dir_path in [self.best_model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Experiment identifier
        self.experiment_name = f"unixcoder_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Other configuration
        self.log_steps = 100
        self.eval_steps = 500
        self.memory_cleanup_frequency = 5


class PrimeVulDataset(Dataset):
    """PrimeVul dataset class"""

    def __init__(self, data_file: Path, tokenizer, config: UniXcoderConfig):
        self.data = []
        self.tokenizer = tokenizer
        self.config = config

        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

        self._log_distribution()

    def _log_distribution(self):
        """Log data distribution"""
        dist = {0: 0, 1: 0}
        for item in self.data:
            dist[item.get('target', 0)] += 1
        logger.info(f"Dataset samples: {len(self.data)}, label distribution: {dist}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get function code
        func_code = item.get('func', '')
        label = item.get('target', 0)

        # UniXcoder encoding
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
            'label': torch.tensor(label, dtype=torch.long),
            'idx': item.get('idx', idx)
        }


class UniXcoderClassifier(nn.Module):
    """UniXcoder-based vulnerability detection classifier"""

    def __init__(self, config: UniXcoderConfig):
        super().__init__()
        self.config = config

        # Load UniXcoder model
        self.unixcoder = AutoModel.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True
        )

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize classification head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        # UniXcoder encoding
        outputs = self.unixcoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def calculate_metrics(preds: List[int], labels: List[int], probs: Optional[np.ndarray] = None):
    """Calculate standard metrics"""
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

    if probs is not None:
        vds = calculate_vds(labels, probs, fpr_threshold=0.005)
        metrics['vd_s'] = vds

    return metrics


def calculate_vds(labels: List[int], probs: np.ndarray, fpr_threshold: float = 0.005):
    """Calculate VD-S: FNR @ (FPR ≤ threshold)"""
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


def evaluate_paired_predictions(model: nn.Module, loader: DataLoader, device: torch.device, config: UniXcoderConfig):
    """Paired prediction evaluation (only for balanced dataset)"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            if config.fp16_full_eval:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)

            preds = logits.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            for pred, label in zip(preds, labels):
                predictions.append({'pred': pred, 'label': label})

    # Paired analysis
    pair_metrics = {'P-C': 0, 'P-V': 0, 'P-B': 0, 'P-R': 0}

    for i in range(0, len(predictions) - 1, 2):
        if i + 1 >= len(predictions):
            break

        vuln_item = predictions[i]
        benign_item = predictions[i + 1]

        if vuln_item['label'] == 1 and benign_item['label'] == 0:
            if vuln_item['pred'] == 1 and benign_item['pred'] == 0:
                pair_metrics['P-C'] += 1
            elif vuln_item['pred'] == 1 and benign_item['pred'] == 1:
                pair_metrics['P-V'] += 1
            elif vuln_item['pred'] == 0 and benign_item['pred'] == 0:
                pair_metrics['P-B'] += 1
            elif vuln_item['pred'] == 0 and benign_item['pred'] == 1:
                pair_metrics['P-R'] += 1

    total_pairs = sum(pair_metrics.values())
    if total_pairs > 0:
        for key in pair_metrics:
            pair_metrics[key] = pair_metrics[key] / total_pairs

    return pair_metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, config: UniXcoderConfig):
    """Model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            if config.fp16_full_eval:
                with autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    metrics = calculate_metrics(all_preds, all_labels, all_probs)

    return metrics


def aggressive_memory_cleanup():
    """Memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_memory_usage(device: torch.device, step_name: str = ""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3

        logger.info(f"[{step_name}] GPU Memory - Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        return allocated, reserved, max_allocated
    return 0, 0, 0


def save_best_model(model: nn.Module, tokenizer, config: UniXcoderConfig, metrics: Dict, epoch: int):
    """Save best model"""
    save_path = config.best_model_dir / f"{config.experiment_name}_best.pt"
    logger.info(f"Saving best model to: {save_path}")

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'epoch': epoch,
            'config_dict': {
                'model_name': config.model_name,
                'max_seq_length': config.max_seq_length,
                'dropout': config.dropout
            }
        }

        torch.save(checkpoint, save_path)

        # Save tokenizer
        tokenizer_path = save_path.parent / f"{save_path.stem}_tokenizer"
        tokenizer.save_pretrained(tokenizer_path)

        logger.info(f"Best model saved successfully: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


def save_results(results: Dict, config: UniXcoderConfig, phase: str = "test"):
    """Save experiment results"""
    results_file = config.results_dir / f"{config.experiment_name}_{phase}_results.json"

    processed_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            processed_results[k] = v.tolist()
        else:
            processed_results[k] = v

    with open(results_file, 'w') as f:
        json.dump(processed_results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")


def create_dataloader(dataset, config: UniXcoderConfig, shuffle: bool = True):
    """Create data loader"""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: UniXcoderConfig, device: torch.device, tokenizer):
    """Train model"""
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed precision training
    scaler = GradScaler() if config.fp16 else None

    # Training loop
    model.train()
    best_f1 = 0.0
    best_model_path = None
    no_improve = 0
    is_paired = (config.dataset_type == "balanced")

    logger.info("Starting training...")

    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            if config.fp16:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

            # Memory cleanup
            if batch_idx % config.memory_cleanup_frequency == 0:
                aggressive_memory_cleanup()

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed, average loss: {avg_loss:.4f}")

        # Validation
        logger.info("Starting validation...")
        val_metrics = evaluate_model(model, val_loader, device, config)

        log_str = (f"Validation Epoch {epoch + 1}: Acc={val_metrics['accuracy']:.4f}, "
                   f"Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, "
                   f"F1={val_metrics['f1']:.4f}")

        if 'vd_s' in val_metrics:
            log_str += f", VD-S={val_metrics['vd_s']:.4f}"

        logger.info(log_str)
        logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # Paired prediction metrics (balanced dataset only)
        if is_paired:
            pair_metrics = evaluate_paired_predictions(model, val_loader, device, config)
            val_metrics.update(pair_metrics)
            logger.info(f"Paired metrics: P-C={pair_metrics['P-C']:.4f}, P-V={pair_metrics['P-V']:.4f}, "
                        f"P-B={pair_metrics['P-B']:.4f}, P-R={pair_metrics['P-R']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_f1 + config.min_delta:
            best_f1 = val_metrics['f1']
            no_improve = 0

            logger.info(f"Found better model (F1={best_f1:.4f}), saving...")
            best_model_path = save_best_model(model, tokenizer, config, val_metrics, epoch + 1)
            if best_model_path:
                logger.info(f"Best model saved successfully: {best_model_path}")
        else:
            no_improve += 1
            logger.info(f"No improvement ({no_improve}/{config.patience})")
            if no_improve >= config.patience:
                logger.info(f"Early stopping: {config.patience} epochs without improvement")
                break

        aggressive_memory_cleanup()

    return best_model_path


def test_best_model(best_model_path: Path, config: UniXcoderConfig, device: torch.device):
    """Test best model"""
    if not best_model_path or not best_model_path.exists():
        logger.warning("Best model not found, skipping test")
        return {}

    logger.info(f"Loading best model from {best_model_path} for testing...")

    # Load tokenizer
    tokenizer_path = best_model_path.parent / f"{best_model_path.stem}_tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            token=config.hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    test_dataset = PrimeVulDataset(config.test_file, tokenizer, config)
    test_loader = create_dataloader(test_dataset, config, shuffle=False)

    # Initialize and load model
    model = UniXcoderClassifier(config)
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Best model loaded, starting test...")

    # Test
    test_metrics = evaluate_model(model, test_loader, device, config)

    log_str = (f"Final test results: Acc={test_metrics['accuracy']:.4f}, "
               f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
               f"F1={test_metrics['f1']:.4f}")

    if 'vd_s' in test_metrics:
        log_str += f", VD-S={test_metrics['vd_s']:.4f}"

    logger.info(log_str)
    logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Paired prediction metrics (balanced dataset only)
    if config.dataset_type == "balanced":
        pair_metrics = evaluate_paired_predictions(model, test_loader, device, config)
        test_metrics.update(pair_metrics)
        logger.info(f"Paired metrics: P-C={pair_metrics['P-C']:.4f}, P-V={pair_metrics['P-V']:.4f}, "
                    f"P-B={pair_metrics['P-B']:.4f}, P-R={pair_metrics['P-R']:.4f}")

    # Save test results
    save_results(test_metrics, config, "test")

    return test_metrics


def train_unixcoder_model(config: UniXcoderConfig):
    """Complete UniXcoder model training pipeline"""
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    log_memory_usage(device, "Before training")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PrimeVulDataset(config.train_file, tokenizer, config)
    val_dataset = PrimeVulDataset(config.valid_file, tokenizer, config)

    # Data loaders
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    logger.info(f"Dataset sizes: train={len(train_dataset)}, valid={len(val_dataset)}")

    # Model
    logger.info("Initializing model...")
    model = UniXcoderClassifier(config)
    model.to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

    log_memory_usage(device, "After model loading")

    # Training
    best_model_path = train_model(model, train_loader, val_loader, config, device, tokenizer)

    # Clean training resources
    del model, train_loader, val_loader, train_dataset, val_dataset
    aggressive_memory_cleanup()

    # Test best model
    logger.info("=" * 50)
    logger.info("Starting best model testing...")
    logger.info("=" * 50)

    test_results = test_best_model(best_model_path, config, device)

    return test_results


def main():
    parser = argparse.ArgumentParser(description="UniXcoder PrimeVul vulnerability detection")
    parser.add_argument("--dataset_type", type=str, choices=["balanced", "unbalanced"],
                        default="unbalanced", help="Dataset type")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable mixed precision training")

    args = parser.parse_args()

    # Set CUDA memory management
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Configuration
    config = UniXcoderConfig(dataset_type=args.dataset_type)
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.max_seq_length = args.max_seq_length
    config.fp16 = args.fp16

    logger.info("=" * 80)
    logger.info(f"Starting UniXcoder model training - {args.dataset_type} dataset")
    logger.info("=" * 80)

    try:
        results = train_unixcoder_model(config)
        if results:
            logger.info(f"Training completed, test F1: {results['f1']:.4f}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        aggressive_memory_cleanup()


if __name__ == "__main__":
    main()