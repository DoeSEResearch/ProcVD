#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import math
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig

from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import wandb


# Memory optimization utilities
def cleanup_memory():
    """Deep cleanup of GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    for _ in range(3):
        gc.collect()


def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"GPU Memory: {allocated:.2f}GB/{total:.2f}GB allocated, {reserved:.2f}GB reserved, peak {max_allocated:.2f}GB"
    return "CUDA not available"


def optimize_cuda_settings():
    """Optimize CUDA settings for memory efficiency"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,roundup_power2_divisions:16,garbage_collection_threshold:0.6'


def force_cleanup_model(model):
    """Force cleanup of model GPU memory"""
    if model is not None:
        model.cpu()
        del model
        cleanup_memory()


def reset_cuda_peak_memory():
    """Reset CUDA peak memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()


def aggressive_cleanup():
    """Aggressive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        for _ in range(5):
            gc.collect()
        torch.cuda.empty_cache()


# Disk space utilities
def check_disk_space(path: Path, required_gb: float = 5.0) -> bool:
    """Check if disk space is sufficient"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < required_gb:
            logger.warning(f"Insufficient disk space: {free_gb:.2f}GB < {required_gb:.2f}GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Cannot check disk space: {e}")
        return True


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 2):
    """Clean up old checkpoint files, keep only the last N"""
    if not checkpoint_dir.exists():
        return

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoint_files) <= keep_last_n:
        return

    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    files_to_delete = checkpoint_files[:-keep_last_n]

    for file_path in files_to_delete:
        try:
            file_path.unlink()
            logger.info(f"Deleted old checkpoint: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")


# Logging setup
def setup_logging(log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
    return logger


logger = setup_logging()


# Configuration
class Config:
    def __init__(self, dataset_type: str = "balanced"):
        self.hf_token = os.environ.get('HF_TOKEN', 'YOUR HUGGINGFACE API KEY')
        self.repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self.local_model_dir = "../../../model/DeepSeek-R1-Distill-Qwen-32B"

        # Dataset configuration
        self.dataset_type = dataset_type
        self.data_dir = Path("../../../data/primevul_process")

        if dataset_type == "balanced":
            self.train_file = self.data_dir / "PrimeVul_balanced_train.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_balanced_valid.jsonl"
            self.test_file = self.data_dir / "PrimeVul_balanced_test.jsonl"
        else:
            self.train_file = self.data_dir / "PrimeVul_unbalanced_train_sampled.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_unbalanced_valid_sampled.jsonl"
            self.test_file = self.data_dir / "PrimeVul_unbalanced_test_sampled.jsonl"

        # Training configuration - memory optimized
        self.batch_size = 3
        self.gradient_accumulation_steps = 8
        self.lr = 2e-5
        self.epochs = 5
        self.max_length = 320
        self.seed = 42
        self.device = torch.device("cuda:0")
        self.warmup_steps = 500
        self.min_lr = 1e-6
        self.patience = 3
        self.min_delta = 0.001

        # Save configuration
        self.checkpoint_dir = f"../../../result/{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_every_n_epochs = 3
        self.keep_last_checkpoints = 1
        self.enable_checkpoint_cleanup = True

        # Data loading optimization
        self.num_workers = 0
        self.pin_memory = False
        self.persistent_workers = False
        self.test_batch_size = 1
        self.cleanup_freq = 5

        # Wandb configuration
        self.wandb_api_key = "YOUR WANDB_API_KEY"
        self.wandb_project = f"primevul-{dataset_type}-finetuning"

        # VD-S configuration
        self.fpr_threshold = 0.005  # 0.5% FPR threshold

        self._validate_model_path()

    def _validate_model_path(self):
        """Validate local model path exists"""
        model_path = Path(self.local_model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.local_model_dir}")

        config_exists = (model_path / 'config.json').exists()
        model_exists = False

        # Check various model weight file formats
        if (model_path / 'pytorch_model.bin').exists():
            model_exists = True
        elif (model_path / 'model.safetensors').exists():
            model_exists = True
        elif list(model_path.glob('pytorch_model-*.bin')):
            model_exists = True
        elif list(model_path.glob('model-*.safetensors')):
            model_exists = True
        elif (model_path / 'pytorch_model.bin.index.json').exists():
            model_exists = True
        elif (model_path / 'model.safetensors.index.json').exists():
            model_exists = True

        if not config_exists:
            raise FileNotFoundError(f"Model config file not found: {model_path / 'config.json'}")
        if not model_exists:
            existing_files = [f.name for f in model_path.iterdir() if f.is_file()]
            raise FileNotFoundError(f"Model weight files not found in: {model_path}\nExisting files: {existing_files}")

        logger.info(f"Model path validation successful: {self.local_model_dir}")

        # Show found model files
        model_files = []
        if (model_path / 'pytorch_model.bin').exists():
            model_files.append('pytorch_model.bin')
        if (model_path / 'model.safetensors').exists():
            model_files.append('model.safetensors')
        model_files.extend([f.name for f in model_path.glob('pytorch_model-*.bin')])
        model_files.extend([f.name for f in model_path.glob('model-*.safetensors')])
        if model_files:
            logger.info(f"Found model files: {model_files[:3]}{'...' if len(model_files) > 3 else ''}")


# Dataset
class PrimeVulDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer: AutoTokenizer, config: Config):
        self.records = []
        self.tokenizer = tokenizer
        self.config = config

        # Read JSONL file
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.records.append({
                    'idx': data['idx'],
                    'code': data['func'],
                    'label': data['target']
                })

        self._log_distribution()

    def _log_distribution(self):
        dist = {0: 0, 1: 0}
        for r in self.records:
            dist[r['label']] += 1
        logger.info(f"Sample count: {len(self.records)}, Label distribution: {dist}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec['code'],
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = torch.tensor(rec['label'], dtype=torch.long)
        return enc, label, rec['idx']


# Model - memory optimized
class VulnDetectionModel(nn.Module):
    def __init__(self, model_path: str, hf_token: str, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Load pretrained model with memory optimization
        self.llm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=hf_token,
            device_map={"": "cuda:0"},
            local_files_only=True
        )

        # Apply LoRA with smaller parameters
        peft_conf = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=None
        )
        self.llm = get_peft_model(self.llm, peft_conf)

        # Enable gradient checkpointing
        self.llm.gradient_checkpointing_enable()

        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, enc: Dict, device: torch.device) -> torch.Tensor:
        input_ids = enc['input_ids'].squeeze(1).to(device, non_blocking=True)
        attn_mask = enc['attention_mask'].squeeze(1).to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            out = self.llm(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            text_emb = out.last_hidden_state[:, 0].to(torch.float32)

        text_emb = self.dropout(text_emb)
        logits = self.classifier(text_emb)
        return logits


# Evaluation metrics
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

    # Calculate VD-S (FNR @ FPR ≤ 0.5%)
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

        # Calculate FPR and FNR
        tn = sum((1 - preds[i]) * (1 - labels[i]) for i in range(len(labels)))
        fp = sum(preds[i] * (1 - labels[i]) for i in range(len(labels)))
        fn = sum((1 - preds[i]) * labels[i] for i in range(len(labels)))
        tp = sum(preds[i] * labels[i] for i in range(len(labels)))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        if fpr <= fpr_threshold:
            best_fnr = min(best_fnr, fnr)

    return best_fnr


def evaluate_paired_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate paired predictions (for balanced dataset only)"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for enc, labels, _ in loader:
            logits = model(enc, device)
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            for pred, label in zip(preds, labels):
                predictions.append({'pred': pred, 'label': label})

    # Paired analysis (assumes data appears in pairs)
    pair_metrics = {'P-C': 0, 'P-V': 0, 'P-B': 0, 'P-R': 0}

    for i in range(0, len(predictions) - 1, 2):
        vuln_item = predictions[i]
        benign_item = predictions[i + 1]

        # Ensure correct paired labels (one vulnerable, one benign)
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


# Evaluation function
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             calculate_vds_metric: bool = True, is_paired: bool = False):
    model.eval()
    preds, labs, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, (enc, lab, _) in enumerate(loader):
            logits = model(enc, device)
            probs = F.softmax(logits, dim=1)

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labs.extend(lab.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

            # Periodic memory cleanup during evaluation
            if batch_idx % 20 == 0:
                aggressive_cleanup()

    # Calculate standard metrics
    all_probs = np.array(all_probs) if calculate_vds_metric else None
    metrics = calculate_metrics(preds, labs, all_probs)

    # Calculate paired metrics (for balanced dataset only)
    if is_paired:
        pair_metrics = evaluate_paired_predictions(model, loader, device)
        metrics.update(pair_metrics)

    return metrics


# Training scheduler
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        # Cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# Save and load checkpoints - fixed save errors
def safe_save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                         best_metrics: Dict, checkpoint_dir: Path, is_best: bool = False,
                         enable_cleanup: bool = True, keep_last_n: int = 1):
    """Safe checkpoint saving function with error fixes"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space
    if not check_disk_space(checkpoint_dir, required_gb=3.0):
        logger.warning("Insufficient disk space, skipping checkpoint save")
        return

    # Clean up old checkpoints
    if enable_cleanup:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    try:
        # Key fix: ensure all tensors are on CPU and contiguous
        model_state_dict = {}
        for key, tensor in model.state_dict().items():
            cpu_tensor = tensor.detach().cpu().contiguous()
            model_state_dict[key] = cpu_tensor

        optimizer_state_dict = {}
        for key, value in optimizer.state_dict().items():
            if isinstance(value, torch.Tensor):
                optimizer_state_dict[key] = value.detach().cpu().contiguous()
            elif isinstance(value, dict):
                optimizer_state_dict[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        optimizer_state_dict[key][sub_key] = sub_value.detach().cpu().contiguous()
                    else:
                        optimizer_state_dict[key][sub_key] = sub_value
            else:
                optimizer_state_dict[key] = value

        # Save only necessary state info to reduce file size
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'best_metrics': {k: v for k, v in best_metrics.items() if k != 'confusion_matrix'}
        }

        cleanup_memory()

        # Save regular checkpoint
        if not is_best:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            temp_path = checkpoint_dir / 'best_model_temp.pt'

            torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)

            # Atomic replacement
            if best_path.exists():
                best_path.unlink()
            temp_path.rename(best_path)
            logger.info(f"Saved best model: {best_path}")

        # Clean up temporary variables
        del model_state_dict, optimizer_state_dict, checkpoint
        cleanup_memory()

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        temp_path = checkpoint_dir / 'best_model_temp.pt'
        if temp_path.exists():
            temp_path.unlink()
        cleanup_memory()


def load_model_for_testing(model_path: str, hf_token: str, hidden_size: int, checkpoint_path: Path,
                           device: torch.device):
    """Load model specifically for testing with optimized memory usage"""
    logger.info(f"Loading model for testing... {get_gpu_memory_info()}")

    # Re-initialize a new model
    test_model = VulnDetectionModel(
        model_path=model_path,
        hf_token=hf_token,
        hidden_size=hidden_size
    ).to(device)

    logger.info(f"Test model initialized {get_gpu_memory_info()}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    test_model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Test model weights loaded {get_gpu_memory_info()}")

    return test_model, checkpoint['best_metrics']


# Main training function
def train_model(config: Config):
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Optimize CUDA settings
    optimize_cuda_settings()
    reset_cuda_peak_memory()
    logger.info(f"Initial {get_gpu_memory_info()}")

    # Initialize wandb
    wandb.login(key=config.wandb_api_key)
    wandb.init(
        project=config.wandb_project,
        config=vars(config),
        name=f"{config.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Use local model path directly
    logger.info(f"Using local model path: {config.local_model_dir}")
    model_path = config.local_model_dir
    logger.info(f"Model path setup complete: {model_path}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        token=config.hf_token,
        local_files_only=True
    )

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PrimeVulDataset(config.train_file, tokenizer, config)
    valid_dataset = PrimeVulDataset(config.valid_file, tokenizer, config)

    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False
    )

    logger.info(f"Dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}")

    # Get model config
    logger.info("Loading model config...")
    hf_conf = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=config.hf_token,
        local_files_only=True
    )

    # Initialize model
    logger.info("Initializing model...")
    model = VulnDetectionModel(
        model_path=model_path,
        hf_token=config.hf_token,
        hidden_size=hf_conf.hidden_size
    ).to(config.device)

    logger.info(f"After model loading {get_gpu_memory_info()}")

    # Force memory cleanup
    aggressive_cleanup()
    logger.info(f"After memory cleanup {get_gpu_memory_info()}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Calculate total steps
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        eta_min=config.min_lr
    )

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_metrics = {}
    no_improve = 0
    is_paired = (config.dataset_type == "balanced")

    logger.info("Starting training...")
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (enc, labels, _) in enumerate(train_loader):
            labels = labels.to(config.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(enc, config.device)
                loss = criterion(logits, labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update learning rate
                scheduler.step()
                global_step += 1

                # Periodic memory cleanup
                if global_step % config.cleanup_freq == 0:
                    aggressive_cleanup()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % 20 == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss={current_loss:.4f} LR={current_lr:.2e} {get_gpu_memory_info()}"
                )
                wandb.log({
                    'train_loss': current_loss,
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'global_step': global_step
                })

        # Calculate average loss
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # Memory cleanup before validation
        aggressive_cleanup()

        # Validation
        val_metrics = evaluate(model, valid_loader, config.device,
                               calculate_vds_metric=True, is_paired=is_paired)

        log_str = (f"Validation Epoch {epoch}: Acc={val_metrics['accuracy']:.4f}, "
                   f"Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, "
                   f"F1={val_metrics['f1']:.4f}")

        if 'vd_s' in val_metrics:
            log_str += f", VD-S={val_metrics['vd_s']:.4f}"

        logger.info(log_str)
        logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # Log paired metrics
        if is_paired:
            logger.info(f"Paired metrics: P-C={val_metrics['P-C']:.4f}, P-V={val_metrics['P-V']:.4f}, "
                        f"P-B={val_metrics['P-B']:.4f}, P-R={val_metrics['P-R']:.4f}")

        # Log to wandb
        wandb_metrics = {
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch': epoch
        }
        if 'vd_s' in val_metrics:
            wandb_metrics['val_vd_s'] = val_metrics['vd_s']
        if is_paired:
            wandb_metrics.update({
                f'val_{k}': v for k, v in val_metrics.items() if k.startswith('P-')
            })
        wandb.log(wandb_metrics)

        # Save checkpoints
        if epoch % config.save_every_n_epochs == 0:
            safe_save_checkpoint(
                model, optimizer, epoch, val_metrics,
                Path(config.checkpoint_dir), is_best=False,
                enable_cleanup=config.enable_checkpoint_cleanup,
                keep_last_n=config.keep_last_checkpoints
            )

        # Save best model
        if val_metrics['f1'] > best_f1 + config.min_delta:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            no_improve = 0
            safe_save_checkpoint(
                model, optimizer, epoch, val_metrics,
                Path(config.checkpoint_dir), is_best=True,
                enable_cleanup=config.enable_checkpoint_cleanup,
                keep_last_n=config.keep_last_checkpoints
            )
            logger.info(f"Found best model, F1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info(f"Early stopping: {config.patience} epochs without improvement")
                break

        # Memory cleanup
        aggressive_cleanup()

    # Critical optimization: thoroughly clean training-related variables after training
    logger.info("Training complete, starting cleanup of training variables...")

    # Force cleanup of training variables
    del optimizer, scheduler, scaler, criterion
    force_cleanup_model(model)
    del train_loader, valid_loader, train_dataset, valid_dataset
    aggressive_cleanup()

    logger.info(f"Training variables cleanup complete {get_gpu_memory_info()}")

    # Reload test data and model for sufficient memory
    logger.info("Starting test preparation...")

    # Load test dataset
    test_dataset = PrimeVulDataset(config.test_file, tokenizer, config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        num_workers=0,
        pin_memory=False
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Test data loading complete {get_gpu_memory_info()}")

    # Re-initialize model for testing
    best_model_path = Path(config.checkpoint_dir) / 'best_model.pt'
    if best_model_path.exists():
        test_model, best_metrics = load_model_for_testing(
            model_path, config.hf_token, hf_conf.hidden_size, best_model_path, config.device)
    else:
        logger.warning("Best model file not found, re-initializing model")
        test_model = VulnDetectionModel(
            model_path=model_path,
            hf_token=config.hf_token,
            hidden_size=hf_conf.hidden_size
        ).to(config.device)
        best_metrics = {}

    logger.info(f"Test model ready {get_gpu_memory_info()}")

    # Execute testing
    logger.info("Starting final testing...")
    test_metrics = evaluate(test_model, test_loader, config.device,
                            calculate_vds_metric=True, is_paired=is_paired)

    log_str = (f"Final test: Acc={test_metrics['accuracy']:.4f}, "
               f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
               f"F1={test_metrics['f1']:.4f}")

    if 'vd_s' in test_metrics:
        log_str += f", VD-S={test_metrics['vd_s']:.4f}"

    logger.info(log_str)
    logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    if is_paired:
        logger.info(f"Paired metrics: P-C={test_metrics['P-C']:.4f}, P-V={test_metrics['P-V']:.4f}, "
                    f"P-B={test_metrics['P-B']:.4f}, P-R={test_metrics['P-R']:.4f}")

    # Log final test results to wandb
    test_wandb_metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1']
    }
    if 'vd_s' in test_metrics:
        test_wandb_metrics['test_vd_s'] = test_metrics['vd_s']
    if is_paired:
        test_wandb_metrics.update({
            f'test_{k}': v for k, v in test_metrics.items() if k.startswith('P-')
        })
    wandb.log(test_wandb_metrics)

    # Save final model
    try:
        final_model_path = Path(config.checkpoint_dir) / 'final_model'
        if check_disk_space(final_model_path.parent, required_gb=2.0):
            test_model.llm.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
        else:
            logger.warning("Insufficient disk space, skipping final model save")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # Final cleanup
    force_cleanup_model(test_model)
    del test_loader, test_dataset, tokenizer
    aggressive_cleanup()

    logger.info(f"Final cleanup complete {get_gpu_memory_info()}")

    wandb.finish()


# Main function
def main():
    # Train balanced dataset first
    logger.info("=" * 80)
    logger.info("Starting training on balanced dataset")
    logger.info("=" * 80)
    try:
         config_balanced = Config(dataset_type="balanced")
         train_model(config_balanced)
    except Exception as e:
         logger.error(f"Balanced dataset training failed: {e}")
         cleanup_memory()
         raise

     # Thorough memory cleanup
    cleanup_memory()
    reset_cuda_peak_memory()

    # Train unbalanced dataset
    logger.info("=" * 80)
    logger.info("Starting training on unbalanced dataset")
    logger.info("=" * 80)

    try:
        config_unbalanced = Config(dataset_type="unbalanced")
        train_model(config_unbalanced)
    except Exception as e:
        logger.error(f"Unbalanced dataset training failed: {e}")
        aggressive_cleanup()
        raise


if __name__ == "__main__":
    # Set environment variables for optimized GPU usage
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_LAUNCH_BLOCKING': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:32,roundup_power2_divisions:16,garbage_collection_threshold:0.6'
    })

    # Set CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info("Starting optimized model training...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    try:
        main()
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        aggressive_cleanup()
        raise
    finally:
        # Final cleanup
        aggressive_cleanup()
        logger.info("Program execution complete, memory cleaned")