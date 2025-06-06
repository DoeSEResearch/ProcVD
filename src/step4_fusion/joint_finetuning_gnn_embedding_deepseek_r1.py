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
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


# ------------- Logging Setup -------------
def setup_logging(log_file: str = "deepseek_hybrid_training.log") -> logging.Logger:
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


# ------------- Optimized Configuration -------------
class OptimizedHybridConfig:
    def __init__(self, dataset_type: str = "balanced"):
        # HuggingFace config - DeepSeek-R1-Distill-Qwen-32B
        self.hf_token = os.environ.get('HF_TOKEN', 'YOUR HUGGINGFACE API KEY')
        self.repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self.local_model_dir = "../../model/DeepSeek-R1-Distill-Qwen-32B"

        # Model save paths
        self.save_dir = Path("../../result/embedding_deepseek/")
        self.best_model_dir = self.save_dir / "best_models"
        self.results_dir = self.save_dir / "results"

        # Create directories
        for dir_path in [self.best_model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Dataset configuration
        self.dataset_type = dataset_type
        self.data_dir = Path("../../data/primevul_process")
        self.gnn_embeddings_dir = Path("../../result/primevul_core_features")

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

        # Training config - optimized for DeepSeek-R1-32B
        self.batch_size = 2  # Smaller batch size for 32B model
        self.gradient_accumulation_steps = 32  # Compensate with gradient accumulation
        self.lr = 2e-6
        self.epochs = 12
        self.max_length = 512
        self.seed = 42
        self.device = torch.device("cuda:1")
        self.warmup_steps = 150
        self.min_lr = 1e-8
        self.patience = 3
        self.min_delta = 0.001

        # Model architecture config
        self.gnn_embedding_dim = 384
        self.fusion_dim = 512
        self.dropout_rate = 0.15
        self.fusion_type = "gate"

        # VD-S config
        self.fpr_threshold = 0.005

        # Memory optimization config - for 32B model
        self.enable_cpu_offloading = True
        self.memory_cleanup_frequency = 2
        self.use_checkpoint = True
        self.fp16_full_eval = True
        self.safe_mode = True

        # Experiment identifier
        self.experiment_name = f"deepseek_r1_32b_hybrid_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ------------- Memory Optimized Dataset -------------
class MemoryOptimizedDataset(Dataset):
    def __init__(self, file_path: Path, gnn_embedding_loader,
                 tokenizer, config: OptimizedHybridConfig):
        self.records = []
        self.gnn_loader = gnn_embedding_loader
        self.tokenizer = tokenizer
        self.config = config

        # Preprocess and keep only necessary info
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                gnn_embedding = self.gnn_loader.get_embedding(str(data['idx']))
                if gnn_embedding is not None:
                    self.records.append({
                        'idx': data['idx'],
                        'code': data['func'],
                        'label': data['target'],
                        'gnn_embedding': gnn_embedding.astype(np.float16)
                    })

        self._log_distribution()

    def _log_distribution(self):
        dist = {0: 0, 1: 0}
        for r in self.records:
            dist[r['label']] += 1
        logger.info(f"Memory optimized dataset samples: {len(self.records)}, label distribution: {dist}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Dynamic tokenization, optimized for DeepSeek-R1
        code_text = rec['code']

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

        gnn_embedding = torch.tensor(rec['gnn_embedding'], dtype=torch.float16)
        label = torch.tensor(rec['label'], dtype=torch.long)

        return enc, gnn_embedding, label, rec['idx']


# ------------- Memory Optimized Fusion Module -------------
class OptimizedFusionModule(nn.Module):
    def __init__(self, llm_dim: int, gnn_dim: int, fusion_dim: int,
                 fusion_type: str = "gate", dropout_rate: float = 0.15):
        super().__init__()
        self.fusion_type = fusion_type
        self.llm_dim = llm_dim
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim

        if fusion_type == "gate":
            self.llm_proj = nn.Linear(llm_dim, fusion_dim, bias=False)
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim, bias=False)

            self.gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(fusion_dim // 2, 1, bias=False),
                nn.Sigmoid()
            )

            self.norm = nn.LayerNorm(fusion_dim)
            self.dropout = nn.Dropout(dropout_rate)

        elif fusion_type == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(llm_dim + gnn_dim, fusion_dim, bias=False),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        elif fusion_type == "attention":
            self.llm_proj = nn.Linear(llm_dim, fusion_dim, bias=False)
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim, bias=False)

            self.attention = nn.MultiheadAttention(
                fusion_dim, num_heads=8, batch_first=True, bias=False
            )
            self.norm = nn.LayerNorm(fusion_dim)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, llm_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "gate":
            gnn_emb = gnn_emb.to(torch.float32)

            llm_proj = self.llm_proj(llm_emb)
            gnn_proj = self.gnn_proj(gnn_emb)

            gate_input = torch.cat([llm_proj, gnn_proj], dim=1)
            gate_weights = self.gate(gate_input)

            fused = gate_weights * llm_proj + (1 - gate_weights) * gnn_proj
            fused = self.norm(fused)
            fused = self.dropout(fused)

            return fused

        elif self.fusion_type == "concat":
            gnn_emb = gnn_emb.to(torch.float32)
            concat_emb = torch.cat([llm_emb, gnn_emb], dim=1)
            return self.fusion_layer(concat_emb)

        elif self.fusion_type == "attention":
            gnn_emb = gnn_emb.to(torch.float32)
            llm_proj = self.llm_proj(llm_emb).unsqueeze(1)
            gnn_proj = self.gnn_proj(gnn_emb).unsqueeze(1)

            seq = torch.cat([llm_proj, gnn_proj], dim=1)
            attn_out, _ = self.attention(seq, seq, seq)
            fused = attn_out.mean(dim=1)
            fused = self.norm(fused)
            fused = self.dropout(fused)

            return fused


# ------------- Memory Optimized Hybrid Model -------------
class OptimizedHybridModel(nn.Module):
    def __init__(self, model_path: str, hf_token: str, config: OptimizedHybridConfig):
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

            self.llm = AutoModel.from_pretrained(model_path, **model_kwargs)
            logger.info("Successfully loaded DeepSeek-R1 model using AutoModel from local path")

        except Exception as e:
            logger.error(f"Failed to load LLM model from local path: {e}")
            logger.info("Trying basic mode from local...")
            try:
                self.llm = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token=hf_token
                )
                logger.info("Successfully loaded in basic mode from local")
            except Exception as e2:
                logger.error(f"Basic mode loading also failed: {e2}")
                raise

        try:
            # DeepSeek-R1 based on Qwen architecture, adjust LoRA target_modules
            peft_conf = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen architecture attention modules
                modules_to_save=None,
                bias="none"
            )
            self.llm = get_peft_model(self.llm, peft_conf)
            logger.info("DeepSeek-R1 LoRA configuration applied successfully")

            if config.use_checkpoint:
                self.llm.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        except Exception as e:
            logger.error(f"LoRA configuration failed: {e}")
            raise

        # Get LLM hidden dimension - DeepSeek-R1-32B hidden dimension
        llm_hidden_size = getattr(self.llm.config, 'hidden_size', 4096)  # Qwen architecture default 4096
        logger.info(f"DeepSeek-R1 hidden dimension: {llm_hidden_size}")

        # Fusion module
        self.fusion = OptimizedFusionModule(
            llm_dim=llm_hidden_size,
            gnn_dim=config.gnn_embedding_dim,
            fusion_dim=config.fusion_dim,
            fusion_type=config.fusion_type,
            dropout_rate=config.dropout_rate
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim, bias=False),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_dim, config.fusion_dim // 2, bias=False),
            nn.LayerNorm(config.fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.fusion_dim // 2, 2)
        )

        logger.info("DeepSeek-R1 hybrid model initialization completed")

    def forward(self, enc: Dict, gnn_emb: torch.Tensor, device: torch.device) -> torch.Tensor:
        input_ids = enc['input_ids'].squeeze(1).to(device)
        attn_mask = enc['attention_mask'].squeeze(1).to(device)
        gnn_emb = gnn_emb.to(device)

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

        # Fusion
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            llm_emb_fp32 = llm_emb.to(torch.float32)
            fused_emb = self.fusion(llm_emb_fp32, gnn_emb)
            logits = self.classifier(fused_emb)

        return logits


# ------------- Optimized Model Saving Function -------------
def save_best_model_only(model: nn.Module, tokenizer, config: OptimizedHybridConfig,
                         metrics: Dict, epoch: int):
    """Save only the best model to avoid frequent IO operations"""
    save_path = config.best_model_dir / f"{config.experiment_name}_best.pt"
    logger.info(f"Saving best model to: {save_path}")

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'epoch': epoch,
            'model_type': 'deepseek_r1_hybrid',
            'config_dict': {
                'fusion_type': config.fusion_type,
                'fusion_dim': config.fusion_dim,
                'gnn_embedding_dim': config.gnn_embedding_dim,
                'dropout_rate': config.dropout_rate,
                'repo_id': config.repo_id
            }
        }

        torch.save(checkpoint, save_path)

        tokenizer_path = save_path.parent / f"{save_path.stem}_tokenizer"
        tokenizer.save_pretrained(tokenizer_path)

        logger.info(f"Best model saved successfully: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


def save_results(results: Dict, config: OptimizedHybridConfig, phase: str = "test"):
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


# ------------- Memory Optimized DataLoader -------------
def create_optimized_dataloader(dataset, config: OptimizedHybridConfig, shuffle: bool = True):
    """Create memory optimized dataloader"""
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


# ------------- Memory Monitoring Tools -------------
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


# ------------- Optimized Evaluation Function -------------
def memory_efficient_evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
                              config: OptimizedHybridConfig, calculate_vds_metric: bool = True,
                              is_paired: bool = False):
    """Memory optimized evaluation function"""
    model.eval()
    preds, labs, all_probs = [], [], []

    eval_batch_count = 0

    with torch.no_grad():
        for enc, gnn_emb, lab, _ in loader:
            if config.fp16_full_eval:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(enc, gnn_emb, device)
                    if calculate_vds_metric:
                        probs = F.softmax(logits.float(), dim=1)
                    else:
                        probs = None
            else:
                logits = model(enc, gnn_emb, device)
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
                                          device: torch.device, config: OptimizedHybridConfig):
    """Memory optimized paired prediction evaluation"""
    model.eval()
    predictions = []

    with torch.no_grad():
        batch_count = 0
        for enc, gnn_emb, labels, _ in loader:
            if config.fp16_full_eval:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(enc, gnn_emb, device)
            else:
                logits = model(enc, gnn_emb, device)

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


# ------------- Evaluation Metrics Functions -------------
def calculate_metrics(preds: List[int], labels: List[int], probs: Optional[np.ndarray] = None):
    """Calculate common metrics"""
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


# ------------- GNN Embedding Loader and Scheduler -------------
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
        """Get GNN embedding by idx"""
        return self.embeddings_dict.get(str(idx), None)


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


# ------------- Complete Model Memory Release Function -------------
def completely_free_model(model):
    """Completely release model GPU memory"""
    if model is not None:
        model.cpu()
        del model

    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    logger.info("Model GPU memory completely released")


def load_best_model_for_test(model_path: Path, config: OptimizedHybridConfig, device: torch.device):
    """Load best model for testing"""
    logger.info(f"Loading best model from {model_path} for testing...")

    checkpoint = torch.load(model_path, map_location='cpu')

    logger.info("Reinitializing DeepSeek-R1 model for testing...")

    model_dir = config.local_model_dir
    logger.info(f"Using local model path: {model_dir}")

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=True,
        token=config.hf_token
    )
    logger.info("Successfully loaded tokenizer using AutoTokenizer from local path")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = OptimizedHybridModel(
        model_path=model_dir,
        hf_token=config.hf_token,
        config=config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Best model loaded from local path, ready for testing")
    return model, tokenizer


# ------------- Fixed Main Training Function -------------
def train_optimized_hybrid_model(config: OptimizedHybridConfig):
    """Memory optimized hybrid model training - removed checkpoint saving"""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.device)

    log_memory_usage(config.device, "Before training")

    logger.info(f"Using local DeepSeek-R1 model path: {config.local_model_dir}")
    model_path = config.local_model_dir

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    logger.info(f"DeepSeek-R1 model path verification completed: {model_path}")

    logger.info("Initializing DeepSeek-R1 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        token=config.hf_token
    )
    logger.info("Successfully loaded tokenizer using AutoTokenizer from local path")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_memory_usage(config.device, "After tokenizer loading")

    logger.info("Loading GNN embeddings...")
    train_gnn_loader = GNNEmbeddingLoader(config.train_gnn_embeddings)
    valid_gnn_loader = GNNEmbeddingLoader(config.valid_gnn_embeddings)

    log_memory_usage(config.device, "After GNN embedding loading")

    logger.info("Loading optimized hybrid datasets...")
    train_dataset = MemoryOptimizedDataset(config.train_file, train_gnn_loader, tokenizer, config)
    valid_dataset = MemoryOptimizedDataset(config.valid_file, valid_gnn_loader, tokenizer, config)

    train_loader = create_optimized_dataloader(train_dataset, config, shuffle=True)
    valid_loader = create_optimized_dataloader(valid_dataset, config, shuffle=False)

    logger.info(f"Optimized dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}")

    log_memory_usage(config.device, "After dataset loading")

    logger.info("Initializing optimized DeepSeek-R1 hybrid model...")
    try:
        model = OptimizedHybridModel(
            model_path=model_path,
            hf_token=config.hf_token,
            config=config
        ).to(config.device)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        aggressive_memory_cleanup()
        raise

    log_memory_usage(config.device, "After model loading")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Optimized hybrid model trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    logger.info(f"Fusion module parameters: {fusion_params:,}, classifier parameters: {classifier_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False
    )

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

    logger.info(f"Starting optimized hybrid model training (fusion type: {config.fusion_type})...")
    global_step = 0

    log_memory_usage(config.device, "Training start")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (enc, gnn_emb, labels, _) in enumerate(train_loader):
            labels = labels.to(config.device)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(enc, gnn_emb, config.device)
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

        aggressive_memory_cleanup()
        log_memory_usage(config.device, "Before validation")

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

        aggressive_memory_cleanup()
        log_memory_usage(config.device, f"Epoch {epoch} end")

    logger.info("Training completed, starting to release model memory...")
    completely_free_model(model)

    del optimizer, scheduler, scaler, criterion, train_loader, valid_loader
    del train_dataset, valid_dataset, train_gnn_loader, valid_gnn_loader

    for _ in range(5):
        aggressive_memory_cleanup()

    log_memory_usage(config.device, "After training model completely released")

    if best_model_path and best_model_path.exists():
        logger.info("=" * 50)
        logger.info("Starting to test best model...")
        logger.info("=" * 50)

        test_gnn_loader = GNNEmbeddingLoader(config.test_gnn_embeddings)
        test_dataset = MemoryOptimizedDataset(config.test_file, test_gnn_loader, tokenizer, config)
        test_loader = create_optimized_dataloader(test_dataset, config, shuffle=False)

        log_memory_usage(config.device, "After test data loading")

        test_model, test_tokenizer = load_best_model_for_test(best_model_path, config, config.device)

        log_memory_usage(config.device, "After test model loading")

        test_metrics = memory_efficient_evaluate(
            test_model, test_loader, config.device, config,
            calculate_vds_metric=True, is_paired=is_paired
        )

        log_str = (f"Final test (DeepSeek-R1 hybrid model): Acc={test_metrics['accuracy']:.4f}, "
                   f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
                   f"F1={test_metrics['f1']:.4f}")

        if 'vd_s' in test_metrics:
            log_str += f", VD-S={test_metrics['vd_s']:.4f}"

        logger.info(log_str)
        logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

        if is_paired:
            logger.info(f"Paired metrics: P-C={test_metrics['P-C']:.4f}, P-V={test_metrics['P-V']:.4f}, "
                        f"P-B={test_metrics['P-B']:.4f}, P-R={test_metrics['P-R']:.4f}")

        save_results(test_metrics, config, "test")

        completely_free_model(test_model)
        del test_loader, test_dataset, test_gnn_loader
        aggressive_memory_cleanup()

        log_memory_usage(config.device, "After testing completed")

        return test_metrics
    else:
        logger.warning("Best model not found, skipping test")
        return {}


# ------------- Main Function -------------
def main():
    # Set CUDA memory management - further optimized for 32B model
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.98)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")

    logger.info("=" * 80)
    logger.info("Starting DeepSeek-R1-Distill-Qwen-32B hybrid model training - balanced dataset")
    logger.info("=" * 80)

    config_balanced = OptimizedHybridConfig(dataset_type="balanced")

    try:
        # Train balanced dataset
        #results_balanced = train_optimized_hybrid_model(config_balanced)
        #if results_balanced:
        #    logger.info(f"Balanced dataset training completed, test F1: {results_balanced['f1']:.4f}")

        aggressive_memory_cleanup()

        # Train unbalanced dataset
        logger.info("\n" + "=" * 80)
        logger.info("Starting DeepSeek-R1-Distill-Qwen-32B hybrid model training - unbalanced dataset")
        logger.info("=" * 80)

        config_unbalanced = OptimizedHybridConfig(dataset_type="unbalanced")
        results_unbalanced = train_optimized_hybrid_model(config_unbalanced)
        if results_unbalanced:
            logger.info(f"Unbalanced dataset training completed, test F1: {results_unbalanced['f1']:.4f}")

    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        raise
    finally:
        aggressive_memory_cleanup()


if __name__ == "__main__":
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'
    })

    main()