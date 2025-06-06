#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import math
import pickle
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


# Logging setup
def setup_logging(log_file: str = "gnn_only_training.log") -> logging.Logger:
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


# GNN-Only Configuration
class GNNOnlyConfig:
    def __init__(self, dataset_type: str = "balanced"):
        # Model save paths
        self.save_dir = Path("../../../result/embedding_all_graph_gnn_only/")
        self.best_model_dir = self.save_dir / "best_models"
        self.results_dir = self.save_dir / "results"

        # Create directories
        for dir_path in [self.best_model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Dataset configuration
        self.dataset_type = dataset_type
        self.data_dir = Path("../../../data/primevul_process")
        self.gnn_embeddings_dir = Path("../../../result/primevul_core_all_graph_features")

        if dataset_type == "balanced":
            self.train_file = self.data_dir / "PrimeVul_balanced_train.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_balanced_valid.jsonl"
            self.test_file = self.data_dir / "PrimeVul_balanced_test.jsonl"
            self.train_gnn_embeddings = self.gnn_embeddings_dir / "train_paired_core_embeddings.pkl"
            self.valid_gnn_embeddings = self.gnn_embeddings_dir / "valid_paired_core_embeddings.pkl"
            self.test_gnn_embeddings = self.gnn_embeddings_dir / "test_paired_core_embeddings.pkl"
        else:
            self.train_file = self.data_dir / "PrimeVul_unbalanced_train_sampled.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_unbalanced_valid_sampled.jsonl"
            self.test_file = self.data_dir / "PrimeVul_unbalanced_test_sampled.jsonl"
            self.train_gnn_embeddings = self.gnn_embeddings_dir / "train_core_embeddings.pkl"
            self.valid_gnn_embeddings = self.gnn_embeddings_dir / "valid_core_embeddings.pkl"
            self.test_gnn_embeddings = self.gnn_embeddings_dir / "test_core_embeddings.pkl"

        # Training configuration
        self.batch_size = 32
        self.gradient_accumulation_steps = 4
        self.lr = 1e-4
        self.epochs = 50
        self.seed = 42
        self.device = torch.device("cuda:0")
        self.warmup_steps = 50
        self.min_lr = 1e-6
        self.patience = 10
        self.min_delta = 0.001

        # Model architecture
        self.gnn_embedding_dim = 384
        self.hidden_dims = [512, 256, 128]
        self.dropout_rate = 0.3

        # VD-S configuration
        self.fpr_threshold = 0.005

        # Memory optimization
        self.memory_cleanup_frequency = 10

        # Experiment identifier
        self.experiment_name = f"gnn_only_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# GNN-Only Dataset
class GNNOnlyDataset(Dataset):
    def __init__(self, file_path: Path, gnn_embedding_loader, config: GNNOnlyConfig):
        self.records = []
        self.gnn_loader = gnn_embedding_loader
        self.config = config

        # Load GNN embeddings and labels
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                gnn_embedding = self.gnn_loader.get_embedding(str(data['idx']))
                if gnn_embedding is not None:
                    self.records.append({
                        'idx': data['idx'],
                        'gnn_embedding': gnn_embedding.astype(np.float32),
                        'label': data['target']
                    })

        self._log_distribution()

    def _log_distribution(self):
        dist = {0: 0, 1: 0}
        for r in self.records:
            dist[r['label']] += 1
        logger.info(f"GNN-Only dataset samples: {len(self.records)}, label distribution: {dist}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        gnn_embedding = torch.tensor(rec['gnn_embedding'], dtype=torch.float32)
        label = torch.tensor(rec['label'], dtype=torch.long)
        return gnn_embedding, label, rec['idx']


# GNN-Only Model
class GNNOnlyModel(nn.Module):
    def __init__(self, config: GNNOnlyConfig):
        super().__init__()
        self.config = config

        # Multi-layer classifier
        layers = []
        input_dim = config.gnn_embedding_dim

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(input_dim, 2))

        self.classifier = nn.Sequential(*layers)

        # Input normalization
        self.input_norm = nn.LayerNorm(config.gnn_embedding_dim)

        logger.info("GNN-Only model initialized")
        self._log_model_info()

    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"GNN-Only model parameters: {trainable_params:,} / {total_params:,}")

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = self.input_norm(gnn_emb)
        # Classification
        logits = self.classifier(x)
        return logits


# GNN Embedding Loader
class GNNEmbeddingLoader:
    def __init__(self, embedding_file: Path):
        self.embedding_file = embedding_file
        self.embeddings_dict = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load GNN core embeddings"""
        if not self.embedding_file.exists():
            raise FileNotFoundError(f"GNN embedding file not found: {self.embedding_file}")

        logger.info(f"Loading GNN embeddings: {self.embedding_file}")

        with open(self.embedding_file, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        file_indices = data.get('file_indices', data.get('file_names', []))

        for i, file_idx in enumerate(file_indices):
            if isinstance(file_idx, str):
                if file_idx.endswith('.json'):
                    file_idx = file_idx[:-5]
                try:
                    file_idx = int(file_idx)
                except ValueError:
                    pass

            self.embeddings_dict[str(file_idx)] = embeddings[i]

        logger.info(f"Successfully loaded {len(self.embeddings_dict)} GNN embeddings")

    def get_embedding(self, idx: str) -> Optional[np.ndarray]:
        """Get GNN embedding by index"""
        return self.embeddings_dict.get(str(idx), None)


# Learning Rate Scheduler
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


# Evaluation Functions
def evaluate_gnn_only(model: nn.Module, loader: DataLoader, device: torch.device,
                      config: GNNOnlyConfig, calculate_vds_metric: bool = True,
                      is_paired: bool = False):
    """GNN-Only model evaluation"""
    model.eval()
    preds, labs, all_probs = [], [], []

    with torch.no_grad():
        for gnn_emb, lab, _ in loader:
            gnn_emb = gnn_emb.to(device)

            logits = model(gnn_emb)

            if calculate_vds_metric:
                probs = F.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labs.extend(lab.cpu().tolist())

    all_probs = np.array(all_probs) if calculate_vds_metric and all_probs else None
    metrics = calculate_metrics(preds, labs, all_probs)

    if is_paired:
        pair_metrics = evaluate_paired_predictions_gnn_only(model, loader, device, config)
        metrics.update(pair_metrics)

    return metrics


def evaluate_paired_predictions_gnn_only(model: nn.Module, loader: DataLoader,
                                         device: torch.device, config: GNNOnlyConfig):
    model.eval()
    predictions = []

    with torch.no_grad():
        for gnn_emb, labels, _ in loader:
            gnn_emb = gnn_emb.to(device)
            logits = model(gnn_emb)
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            for pred, label in zip(preds, labels):
                predictions.append({'pred': pred, 'label': label})

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


# Metrics Calculation
def calculate_metrics(preds: List[int], labels: List[int], probs: Optional[np.ndarray] = None):
    """Calculate evaluation metrics"""
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


# Model Save/Load Functions
def save_best_model_only(model: nn.Module, config: GNNOnlyConfig,
                         metrics: Dict, epoch: int):
    """Save best model"""
    save_path = config.best_model_dir / f"{config.experiment_name}_best.pt"
    logger.info(f"Saving best GNN-Only model to: {save_path}")

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'epoch': epoch,
            'model_type': 'gnn_only',
            'config_dict': {
                'gnn_embedding_dim': config.gnn_embedding_dim,
                'hidden_dims': config.hidden_dims,
                'dropout_rate': config.dropout_rate,
            }
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Best GNN-Only model saved: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


def load_best_model_for_test(model_path: Path, config: GNNOnlyConfig, device: torch.device):
    """Load best model for testing"""
    logger.info(f"Loading best GNN-Only model from {model_path} for testing...")

    checkpoint = torch.load(model_path, map_location='cpu')

    model = GNNOnlyModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Best GNN-Only model loaded, ready for testing")
    return model


def save_results(results: Dict, config: GNNOnlyConfig, phase: str = "test"):
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


# Data Loader Creation
def create_dataloader(dataset, config: GNNOnlyConfig, shuffle: bool = True):
    """Create data loader"""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )


# Memory Monitoring Tools
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


def cleanup_memory():
    """Memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Main Training Function
def train_gnn_only_model(config: GNNOnlyConfig):
    """GNN-Only model training"""
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.device)

    log_memory_usage(config.device, "Before training")

    # Load GNN embeddings
    logger.info("Loading GNN embeddings...")
    train_gnn_loader = GNNEmbeddingLoader(config.train_gnn_embeddings)
    valid_gnn_loader = GNNEmbeddingLoader(config.valid_gnn_embeddings)

    log_memory_usage(config.device, "After GNN embedding loading")

    # Load datasets
    logger.info("Loading GNN-Only datasets...")
    train_dataset = GNNOnlyDataset(config.train_file, train_gnn_loader, config)
    valid_dataset = GNNOnlyDataset(config.valid_file, valid_gnn_loader, config)

    # Create data loaders
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, config, shuffle=False)

    logger.info(f"Dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}")

    log_memory_usage(config.device, "After dataset loading")

    # Initialize model
    logger.info("Initializing GNN-Only model...")
    model = GNNOnlyModel(config).to(config.device)

    log_memory_usage(config.device, "After model loading")

    # Optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
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

    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_path = None
    no_improve = 0
    is_paired = (config.dataset_type == "balanced")

    logger.info("Starting GNN-Only model training...")
    global_step = 0

    log_memory_usage(config.device, "Training start")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (gnn_emb, labels, _) in enumerate(train_loader):
            gnn_emb = gnn_emb.to(config.device)
            labels = labels.to(config.device)

            logits = model(gnn_emb)
            loss = criterion(logits, labels)
            loss = loss / config.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % config.memory_cleanup_frequency == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']

                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss={current_loss:.4f} LR={current_lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # Validation
        cleanup_memory()
        val_metrics = evaluate_gnn_only(
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

        # Save best model
        if val_metrics['f1'] > best_f1 + config.min_delta:
            best_f1 = val_metrics['f1']
            no_improve = 0

            logger.info(f"Found better model (F1={best_f1:.4f}), saving...")
            best_model_path = save_best_model_only(model, config, val_metrics, epoch)
            if best_model_path:
                logger.info(f"Best model saved successfully: {best_model_path}")

        else:
            no_improve += 1
            logger.info(f"No improvement ({no_improve}/{config.patience})")
            if no_improve >= config.patience:
                logger.info(f"Early stopping: {config.patience} epochs without improvement")
                break

        cleanup_memory()

    # Test best model
    if best_model_path and best_model_path.exists():
        logger.info("=" * 50)
        logger.info("Testing best GNN-Only model...")
        logger.info("=" * 50)

        # Load test data
        test_gnn_loader = GNNEmbeddingLoader(config.test_gnn_embeddings)
        test_dataset = GNNOnlyDataset(config.test_file, test_gnn_loader, config)
        test_loader = create_dataloader(test_dataset, config, shuffle=False)

        # Reload best model
        test_model = load_best_model_for_test(best_model_path, config, config.device)

        # Perform testing
        test_metrics = evaluate_gnn_only(
            test_model, test_loader, config.device, config,
            calculate_vds_metric=True, is_paired=is_paired
        )

        log_str = (f"Final test (GNN-Only model): Acc={test_metrics['accuracy']:.4f}, "
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

        cleanup_memory()
        return test_metrics
    else:
        logger.warning("Best model not found, skipping test")
        return {}


# Main Function
def main():
    # CUDA memory management
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("=" * 80)
    logger.info("Starting GNN-Only model training (ablation study) - Balanced dataset")
    logger.info("=" * 80)

    config_balanced = GNNOnlyConfig(dataset_type="balanced")

    try:
        # Train balanced dataset
        results_balanced = train_gnn_only_model(config_balanced)
        if results_balanced:
            logger.info(f"Balanced dataset training completed, test F1: {results_balanced['f1']:.4f}")

        # Release resources
        cleanup_memory()

        # Train unbalanced dataset
        logger.info("\n" + "=" * 80)
        logger.info("Starting GNN-Only model training (ablation study) - Unbalanced dataset")
        logger.info("=" * 80)

        config_unbalanced = GNNOnlyConfig(dataset_type="unbalanced")
        results_unbalanced = train_gnn_only_model(config_unbalanced)
        if results_unbalanced:
            logger.info(f"Unbalanced dataset training completed, test F1: {results_unbalanced['f1']:.4f}")

        # Output comparison results
        logger.info("\n" + "=" * 80)
        logger.info("GNN-Only ablation study results summary:")
        logger.info("=" * 80)
        if results_balanced:
            logger.info(f"Balanced dataset - F1: {results_balanced['f1']:.4f}, "
                        f"Acc: {results_balanced['accuracy']:.4f}, "
                        f"VD-S: {results_balanced.get('vd_s', 'N/A')}")
        if results_unbalanced:
            logger.info(f"Unbalanced dataset - F1: {results_unbalanced['f1']:.4f}, "
                        f"Acc: {results_unbalanced['accuracy']:.4f}, "
                        f"VD-S: {results_unbalanced.get('vd_s', 'N/A')}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Final cleanup
        cleanup_memory()


if __name__ == "__main__":
    # Environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TOKENIZERS_PARALLELISM': 'false'
    })

    main()