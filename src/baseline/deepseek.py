#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import math
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


# Logging setup
def setup_logging(log_file: str = "llm_training.log") -> logging.Logger:
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


# Optimized configuration for DeepSeek-R1-Distill-Qwen-32B
class OptimizedLLMConfig:
    def __init__(self, dataset_type: str = "balanced"):
        # HuggingFace configuration
        self.hf_token = os.environ.get('HF_TOKEN', 'YOUR HUGGINGFACE API KEY')
        self.repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self.local_model_dir = "../../model/DeepSeek-R1-Distill-Qwen-32B"

        # Model save paths
        self.save_dir = Path("../../result/llm_deepseek/")
        self.best_model_dir = self.save_dir / "best_models"
        self.results_dir = self.save_dir / "results"

        # Create directories
        for dir_path in [self.best_model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

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

        # Training configuration optimized for 32B model
        self.batch_size = 2  # Smaller batch size for 32B model
        self.gradient_accumulation_steps = 32  # Compensate with more accumulation steps
        self.lr = 2e-6  # Lower learning rate
        self.epochs = 4
        self.max_length = 512
        self.seed = 42
        self.device = torch.device("cuda:1")
        self.warmup_steps = 150
        self.min_lr = 1e-8
        self.patience = 2
        self.min_delta = 0.001

        # Model architecture
        self.hidden_dim = 512
        self.dropout_rate = 0.2

        # VD-S threshold
        self.fpr_threshold = 0.005

        # Memory optimization for 32B model
        self.enable_cpu_offloading = True
        self.memory_cleanup_frequency = 2
        self.use_checkpoint = True
        self.fp16_full_eval = True
        self.safe_mode = True

        # Experiment identifier
        self.experiment_name = f"deepseek_32b_llm_only_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# Memory-optimized dataset
class MemoryOptimizedDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer, config: OptimizedLLMConfig):
        self.records = []
        self.tokenizer = tokenizer
        self.config = config

        # Preprocess and keep only essential information
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
        logger.info(f"Dataset samples: {len(self.records)}, label distribution: {dist}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        code_text = rec['code']

        # Truncate very long code
        if len(code_text) > 3000:
            code_text = code_text[:3000] + "..."

        enc = self.tokenizer(
            code_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )

        label = torch.tensor(rec['label'], dtype=torch.long)
        return enc, label, rec['idx']


# Memory-optimized LLM model
class OptimizedLLMModel(nn.Module):
    def __init__(self, model_path: str, hf_token: str, config: OptimizedLLMConfig):
        super().__init__()
        self.config = config

        logger.info("Loading DeepSeek-R1-Distill-Qwen-32B model from local path...")
        try:
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "device_map": None,
                "token": hf_token,
                "use_cache": False,
            }

            # Load DeepSeek model using AutoModel
            self.llm = AutoModel.from_pretrained(model_path, **model_kwargs)
            logger.info("DeepSeek model loaded successfully with AutoModel")

        except Exception as e:
            logger.error(f"Failed to load LLM model from local path: {e}")
            logger.info("Trying basic mode...")
            try:
                self.llm = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token=hf_token
                )
                logger.info("Basic mode loading successful")
            except Exception as e2:
                logger.error(f"Basic mode loading also failed: {e2}")
                raise

        try:
            # LoRA configuration for DeepSeek/Qwen architecture
            peft_conf = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen attention layers
                modules_to_save=None,
                bias="none"
            )
            self.llm = get_peft_model(self.llm, peft_conf)
            logger.info("DeepSeek LoRA configuration applied successfully")

            if config.use_checkpoint:
                self.llm.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        except Exception as e:
            logger.error(f"LoRA configuration failed: {e}")
            raise

        # Get LLM hidden size - typically 5120 for 32B Qwen models
        llm_hidden_size = getattr(self.llm.config, 'hidden_size', 5120)
        logger.info(f"DeepSeek model hidden size: {llm_hidden_size}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(llm_hidden_size, config.hidden_dim, bias=False),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2, bias=False),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.hidden_dim // 2, 2)
        )

        logger.info("DeepSeek LLM model initialization completed")

    def forward(self, enc: Dict, device: torch.device) -> torch.Tensor:
        input_ids = enc['input_ids'].squeeze(1).to(device)
        attn_mask = enc['attention_mask'].squeeze(1).to(device)

        # LLM encoding
        with autocast(device_type='cuda', dtype=torch.float16):
            llm_out = self.llm(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True
            )
            if hasattr(llm_out, 'last_hidden_state'):
                hidden_states = llm_out.last_hidden_state
                mask_expanded = attn_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                llm_emb = sum_embeddings / sum_mask
            else:
                llm_emb = llm_out.pooler_output

        # Classification
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            llm_emb_fp32 = llm_emb.to(torch.float32)
            logits = self.classifier(llm_emb_fp32)

        return logits


# Model saving function
def save_best_model_only(model: nn.Module, tokenizer, config: OptimizedLLMConfig,
                         metrics: Dict, epoch: int):
    """Save only the best model to avoid frequent IO operations"""
    save_path = config.best_model_dir / f"{config.experiment_name}_best.pt"
    logger.info(f"Saving best model to: {save_path}")

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'epoch': epoch,
            'model_type': 'deepseek_llm_only',
            'config_dict': {
                'hidden_dim': config.hidden_dim,
                'dropout_rate': config.dropout_rate,
                'repo_id': config.repo_id
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


def save_results(results: Dict, config: OptimizedLLMConfig, phase: str = "test"):
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


# Memory-optimized data loader
def create_optimized_dataloader(dataset, config: OptimizedLLMConfig, shuffle: bool = True):
    """Create memory-optimized data loader"""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False
    )


# Memory monitoring utilities
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


def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Memory-efficient evaluation function
def memory_efficient_evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
                              config: OptimizedLLMConfig, calculate_vds_metric: bool = True,
                              is_paired: bool = False):
    """Memory-efficient evaluation function"""
    model.eval()
    preds, labs, all_probs = [], [], []

    eval_batch_count = 0

    with torch.no_grad():
        for enc, lab, _ in loader:
            if config.fp16_full_eval:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(enc, device)
                    if calculate_vds_metric:
                        probs = F.softmax(logits.float(), dim=1)
                    else:
                        probs = None
            else:
                logits = model(enc, device)
                probs = F.softmax(logits, dim=1) if calculate_vds_metric else None

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labs.extend(lab.cpu().tolist())

            if calculate_vds_metric and probs is not None:
                all_probs.extend(probs.cpu().numpy())

            eval_batch_count += 1

            if eval_batch_count % 5 == 0:
                aggressive_memory_cleanup()

    all_probs = np.array(all_probs) if calculate_vds_metric and all_probs else None
    metrics = calculate_metrics(preds, labs, all_probs)

    if is_paired:
        pair_metrics = evaluate_paired_predictions_optimized(model, loader, device, config)
        metrics.update(pair_metrics)

    return metrics


def evaluate_paired_predictions_optimized(model: nn.Module, loader: DataLoader,
                                          device: torch.device, config: OptimizedLLMConfig):
    """Memory-optimized paired prediction evaluation"""
    model.eval()
    predictions = []

    with torch.no_grad():
        batch_count = 0
        for enc, labels, _ in loader:
            if config.fp16_full_eval:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(enc, device)
            else:
                logits = model(enc, device)

            preds = logits.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            for pred, label in zip(preds, labels):
                predictions.append({'pred': pred, 'label': label})

            batch_count += 1
            if batch_count % 5 == 0:
                aggressive_memory_cleanup()

    pair_metrics = {'P-C': 0, 'P-V': 0, 'P-B': 0, 'P-R': 0}

    for i in range(0, len(predictions) - 1, 2):
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


# Metrics calculation
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


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# Complete model memory cleanup
def completely_free_model(model):
    """Completely free model GPU memory"""
    if model is not None:
        model.cpu()
        del model

    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    logger.info("Model memory completely freed")


def load_best_model_for_test(model_path: Path, config: OptimizedLLMConfig, device: torch.device):
    """Load best model for testing"""
    logger.info(f"Loading best model from {model_path} for testing...")

    checkpoint = torch.load(model_path, map_location='cpu')

    logger.info("Reinitializing DeepSeek model for testing...")

    # Use local model path directly
    model_dir = config.local_model_dir
    logger.info(f"Using local model path: {model_dir}")

    # Verify local model path exists
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_dir}")

    # Load DeepSeek tokenizer with AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=True,
        token=config.hf_token
    )
    logger.info("DeepSeek tokenizer loaded successfully with AutoTokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = OptimizedLLMModel(
        model_path=model_dir,
        hf_token=config.hf_token,
        config=config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Best DeepSeek model loaded from local path, ready for testing")
    return model, tokenizer


# Main training function
def train_optimized_llm_model(config: OptimizedLLMConfig):
    """Memory-optimized LLM model training"""
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.device)

    log_memory_usage(config.device, "Before training")

    # Use local model path directly
    logger.info(f"Using local DeepSeek model path: {config.local_model_dir}")
    model_path = config.local_model_dir

    # Verify local model path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    logger.info(f"DeepSeek model path verified: {model_path}")

    # Initialize tokenizer
    logger.info("Initializing DeepSeek tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        token=config.hf_token
    )
    logger.info("DeepSeek tokenizer loaded successfully with AutoTokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_memory_usage(config.device, "After tokenizer loading")

    # Load optimized datasets
    logger.info("Loading optimized LLM datasets...")
    train_dataset = MemoryOptimizedDataset(config.train_file, tokenizer, config)
    valid_dataset = MemoryOptimizedDataset(config.valid_file, tokenizer, config)

    # Create optimized data loaders
    train_loader = create_optimized_dataloader(train_dataset, config, shuffle=True)
    valid_loader = create_optimized_dataloader(valid_dataset, config, shuffle=False)

    logger.info(f"Optimized dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}")

    log_memory_usage(config.device, "After dataset loading")

    # Initialize optimized LLM model
    logger.info("Initializing optimized DeepSeek LLM model...")
    try:
        model = OptimizedLLMModel(
            model_path=model_path,
            hf_token=config.hf_token,
            config=config
        ).to(config.device)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        aggressive_memory_cleanup()
        raise

    log_memory_usage(config.device, "After model loading")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Optimized LLM model trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    logger.info(f"Classifier parameters: {classifier_params:,}")

    # Optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False
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
    best_model_path = None
    no_improve = 0
    is_paired = (config.dataset_type == "balanced")

    logger.info(f"Starting optimized LLM model training...")
    global_step = 0

    log_memory_usage(config.device, "Training start")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (enc, labels, _) in enumerate(train_loader):
            labels = labels.to(config.device)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(enc, config.device)
                loss = criterion(logits, labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % config.memory_cleanup_frequency == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']

                aggressive_memory_cleanup()

                allocated, reserved, max_allocated = log_memory_usage(config.device, f"Epoch {epoch} Batch {batch_idx}")

                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss={current_loss:.4f} LR={current_lr:.2e} "
                    f"GPU: {allocated:.2f}GB/{reserved:.2f}GB (Max: {max_allocated:.2f}GB)"
                )

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # Memory cleanup before validation
        aggressive_memory_cleanup()
        log_memory_usage(config.device, "Before validation")

        # Validation
        val_metrics = memory_efficient_evaluate(
            model, valid_loader, config.device, config,
            calculate_vds_metric=True, is_paired=is_paired
        )

        log_str = (f"Validation Epoch {epoch}: Acc={val_metrics['accuracy']:.4f}, "
                   f"Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, "
                   f"F1={val_metrics['f1']:.4f}")

        if 'vd_s' in val_metrics:
            log_str += f", VD-S={val_metrics['vd_s']:.4f}"

        logger.info(log_str)
        logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        if is_paired:
            logger.info(f"Paired metrics: P-C={val_metrics['P-C']:.4f}, P-V={val_metrics['P-V']:.4f}, "
                        f"P-B={val_metrics['P-B']:.4f}, P-R={val_metrics['P-R']:.4f}")

        # Save best model only when improved
        if val_metrics['f1'] > best_f1 + config.min_delta:
            best_f1 = val_metrics['f1']
            no_improve = 0

            logger.info(f"Found better model (F1={best_f1:.4f}), saving...")
            best_model_path = save_best_model_only(model, tokenizer, config, val_metrics, epoch)
            if best_model_path:
                logger.info(f"Best model saved successfully: {best_model_path}")
            else:
                logger.error("Model saving failed, but continuing training")

            aggressive_memory_cleanup()
            log_memory_usage(config.device, "After best model saving")
        else:
            no_improve += 1
            logger.info(f"Model performance not improved ({no_improve}/{config.patience})")
            if no_improve >= config.patience:
                logger.info(f"Early stopping: {config.patience} epochs without improvement")
                break

        # Force memory cleanup after each epoch
        aggressive_memory_cleanup()
        log_memory_usage(config.device, f"Epoch {epoch} end")

    # Training completed, completely free training model memory
    logger.info("Training completed, starting to free model memory...")
    completely_free_model(model)

    # Clean up other training-related variables
    del optimizer, scheduler, scaler, criterion, train_loader, valid_loader
    del train_dataset, valid_dataset

    # Force garbage collection
    for _ in range(5):
        aggressive_memory_cleanup()

    log_memory_usage(config.device, "After training model completely freed")

    # Test best model
    if best_model_path and best_model_path.exists():
        logger.info("=" * 50)
        logger.info("Starting to test best model...")
        logger.info("=" * 50)

        # Load test data
        test_dataset = MemoryOptimizedDataset(config.test_file, tokenizer, config)
        test_loader = create_optimized_dataloader(test_dataset, config, shuffle=False)

        log_memory_usage(config.device, "After test data loading")

        # Reload best model
        test_model, test_tokenizer = load_best_model_for_test(best_model_path, config, config.device)

        log_memory_usage(config.device, "After test model loading")

        # Perform testing
        test_metrics = memory_efficient_evaluate(
            test_model, test_loader, config.device, config,
            calculate_vds_metric=True, is_paired=is_paired
        )

        log_str = (f"Final test (DeepSeek LLM model): Acc={test_metrics['accuracy']:.4f}, "
                   f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
                   f"F1={test_metrics['f1']:.4f}")

        if 'vd_s' in test_metrics:
            log_str += f", VD-S={test_metrics['vd_s']:.4f}"

        logger.info(log_str)
        logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

        if is_paired:
            logger.info(f"Paired metrics: P-C={test_metrics['P-C']:.4f}, P-V={test_metrics['P-V']:.4f}, "
                        f"P-B={test_metrics['P-B']:.4f}, P-R={test_metrics['P-R']:.4f}")

        # Save test results
        save_results(test_metrics, config, "test")

        # Clean up test model
        completely_free_model(test_model)
        del test_loader, test_dataset
        aggressive_memory_cleanup()

        log_memory_usage(config.device, "After testing completed")

        return test_metrics
    else:
        logger.warning("Best model not found, skipping testing")
        return {}


# Main function
def main():
    # Set CUDA memory management for 32B model
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)  # Lower memory allocation ratio
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")  # Smaller chunks

    logger.info("=" * 80)
    logger.info("Starting DeepSeek-R1-Distill-Qwen-32B LLM model training - balanced dataset")
    logger.info("=" * 80)

    try:
        # Free resources
        aggressive_memory_cleanup()

        # Train on unbalanced dataset
        logger.info("\n" + "=" * 80)
        logger.info("Starting DeepSeek-R1-Distill-Qwen-32B LLM model training - unbalanced dataset")
        logger.info("=" * 80)

        config_unbalanced = OptimizedLLMConfig(dataset_type="unbalanced")
        results_unbalanced = train_optimized_llm_model(config_unbalanced)
        if results_unbalanced:
            logger.info(f"Unbalanced dataset training completed, test F1: {results_unbalanced['f1']:.4f}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Ensure final cleanup
        aggressive_memory_cleanup()


if __name__ == "__main__":
    # Set environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'
    })

    main()