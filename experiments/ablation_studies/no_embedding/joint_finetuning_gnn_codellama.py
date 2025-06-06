#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv, global_mean_pool, global_max_pool
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from torch.nn.functional import one_hot


# Logging setup
def setup_logging(log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler();
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode='w');
    fh.setFormatter(fmt)
    logger.addHandler(ch);
    logger.addHandler(fh)
    warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
    return logger


logger = setup_logging()


# Configuration
class Config:
    def __init__(self, dataset_type: str = "unbalanced"):
        """
        Args:
            dataset_type: "balanced" or "unbalanced"
        """
        self.dataset_type = dataset_type

        # Model configuration - Using CodeLlama-13b-hf
        self.repo_id = "codellama/CodeLlama-13b-hf"
        self.local_model_dir = "../../../model/CodeLlama-13b-hf"

        # PrimeVul dataset paths
        self.primevul_base_dir = "../../../data/primevul_subgraph"
        self.primevul_labels_dir = "../../../data/primevul_process"

        # Dataset configuration based on type
        if dataset_type == "balanced":
            self.datasets = {
                'train': 'primevul_train_paired',
                'test': 'primevul_test_paired',
                'valid': 'primevul_valid_paired'
            }
            self.label_files = {
                'train': 'PrimeVul_balanced_train.jsonl',
                'test': 'PrimeVul_balanced_test.jsonl',
                'valid': 'PrimeVul_balanced_valid.jsonl'
            }
        else:  # unbalanced
            self.datasets = {
                'train': 'primevul_train',
                'test': 'primevul_test',
                'valid': 'primevul_valid'
            }
            self.label_files = {
                'train': 'PrimeVul_unbalanced_train_sampled.jsonl',
                'test': 'PrimeVul_unbalanced_test_sampled.jsonl',
                'valid': 'PrimeVul_unbalanced_valid_sampled.jsonl'
            }

        # LoRA configuration
        self.lora_r = 32;
        self.lora_alpha = 64;
        self.lora_dropout = 0.05
        self.use_gradient_checkpointing = True

        # GNN configuration
        self.gnn_hidden_dim = 512
        self.gnn_layers = 4
        self.node_types = {'AST': 1, 'CFG': 2, 'PDG': 3, 'CALL': 4, 'DATA': 5, 'CONTROL': 6, 'BINDING': 7}
        self.edge_types = {'REACHING_DEF': 1, 'DATA_FLOW': 2, 'CONTROL_FLOW': 3, 'CALL': 4}

        # Training configuration
        self.batch_size = 16
        self.lr = 2e-4
        self.epochs = 15
        self.max_length = 512
        self.split_ratio = (0.8, 0.1, 0.1)
        self.seed = 42

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learning rate scheduling
        self.warmup_steps = 500
        self.min_lr = 1e-6

        # Early stopping configuration
        self.patience = 3
        self.min_delta = 0.002

        # Data quality control
        self.min_nodes = 5
        self.max_nodes = 200
        self.use_class_weights = True

        self.device = torch.device("cuda:1")

        # VDS configuration
        self.vds_fpr_threshold = 0.005  # 0.5%

        # Whether this is a paired dataset
        self.is_paired = (dataset_type == "balanced")


# VDS calculation function
def calculate_vds(y_true: np.ndarray, y_probs: np.ndarray, fpr_threshold: float = 0.005) -> Dict:
    """
    Calculate Vulnerability Detection Score (VDS)
    VDS = FNR @ (FPR ≤ threshold)

    Args:
        y_true: True labels (0: safe, 1: vulnerable)
        y_probs: Prediction probabilities (vulnerability class probability)
        fpr_threshold: FPR threshold, default 0.5%

    Returns:
        Dict containing VDS-related metrics
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Calculate FNR = 1 - TPR
    fnr = 1 - tpr

    # Find the maximum index where FPR <= threshold
    valid_indices = np.where(fpr <= fpr_threshold)[0]

    if len(valid_indices) == 0:
        # If no valid points, return strictest threshold result
        vds_fnr = fnr[0]
        vds_fpr = fpr[0]
        vds_threshold = thresholds[0]
        vds_tpr = tpr[0]
    else:
        # Select point with minimum FNR where FPR <= threshold
        best_idx = valid_indices[np.argmin(fnr[valid_indices])]
        vds_fnr = fnr[best_idx]
        vds_fpr = fpr[best_idx]
        vds_threshold = thresholds[best_idx]
        vds_tpr = tpr[best_idx]

    return {
        'vds_fnr': vds_fnr,
        'vds_fpr': vds_fpr,
        'vds_threshold': vds_threshold,
        'vds_tpr': vds_tpr,
        'vds_score': 1 - vds_fnr  # VDS score = 1 - FNR, higher is better
    }


# Pairwise evaluation function
def calculate_pairwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate pairwise evaluation metrics
    Assumes data is ordered as [vuln1, safe1, vuln2, safe2, ...]

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict containing P-C, P-V, P-B, P-R metrics
    """
    if len(y_true) % 2 != 0:
        logger.warning("Pairwise evaluation requires even number of samples, skipping last sample")
        y_true = y_true[:-1]
        y_pred = y_pred[:-1]

    # Reorganize into pairs [(vuln, safe), ...]
    pairs_true = []
    pairs_pred = []

    for i in range(0, len(y_true), 2):
        # Each pair: first should be vulnerable(1), second should be safe(0)
        pairs_true.append((y_true[i], y_true[i + 1]))
        pairs_pred.append((y_pred[i], y_pred[i + 1]))

    pc = pb = pv = pr = 0

    for (true_1, true_2), (pred_1, pred_2) in zip(pairs_true, pairs_pred):
        if pred_1 == true_1 and pred_2 == true_2:
            pc += 1  # Pair-wise Correct
        elif pred_1 == 1 and pred_2 == 1:
            pv += 1  # Pair-wise Vulnerable (both predicted as vulnerable)
        elif pred_1 == 0 and pred_2 == 0:
            pb += 1  # Pair-wise Benign (both predicted as benign)
        elif pred_1 != true_1 and pred_2 != true_2:
            pr += 1  # Pair-wise Reversed

    total_pairs = len(pairs_true)

    return {
        'pc': pc / total_pairs if total_pairs > 0 else 0,  # Pair-wise Correct rate
        'pv': pv / total_pairs if total_pairs > 0 else 0,  # Pair-wise Vulnerable rate
        'pb': pb / total_pairs if total_pairs > 0 else 0,  # Pair-wise Benign rate
        'pr': pr / total_pairs if total_pairs > 0 else 0,  # Pair-wise Reversed rate
        'total_pairs': total_pairs
    }


# Data quality control
def filter_low_quality_data(records: List[Dict], config: Config) -> List[Dict]:
    """Filter low-quality data"""
    filtered = []
    stats = {'too_few_nodes': 0, 'too_many_nodes': 0, 'empty_code': 0, 'invalid_graph': 0, 'valid': 0}

    for r in records:
        # Check if code is empty
        if not r.get('code', '').strip():
            stats['empty_code'] += 1
            continue

        # Check graph data format and node count
        graph_data = r.get('graph')
        if graph_data is None:
            stats['invalid_graph'] += 1
            continue

        # Handle PrimeVul's DataFlowSlice array format
        if isinstance(graph_data, list):
            # Merge all nodes and edges from DataFlowSlices
            all_nodes = []
            all_edges = []
            node_id_set = set()  # For deduplication

            for slice_data in graph_data:
                if isinstance(slice_data, dict) and 'nodes' in slice_data:
                    # Add nodes with deduplication
                    for node in slice_data.get('nodes', []):
                        node_id = node.get('id')
                        if node_id not in node_id_set:
                            all_nodes.append(node)
                            node_id_set.add(node_id)

                    # Add edges
                    all_edges.extend(slice_data.get('edges', []))

            # Reconstruct graph data
            r['graph'] = {'nodes': all_nodes, 'edges': all_edges}
            nodes = all_nodes

        elif isinstance(graph_data, dict):
            nodes = graph_data.get('nodes', [])
        else:
            stats['invalid_graph'] += 1
            continue

        if len(nodes) < config.min_nodes:
            stats['too_few_nodes'] += 1
            continue
        if len(nodes) > config.max_nodes:
            stats['too_many_nodes'] += 1
            continue

        stats['valid'] += 1
        filtered.append(r)

    logger.info(f"Data quality filtering ({config.dataset_type}): valid={stats['valid']}, too_few_nodes={stats['too_few_nodes']}, "
                f"too_many_nodes={stats['too_many_nodes']}, empty_code={stats['empty_code']}, "
                f"invalid_graph={stats['invalid_graph']}")
    return filtered


# Dataset
class VulnDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer: AutoTokenizer, config: Config):
        self.records = records;
        self.tokenizer = tokenizer;
        self.config = config
        self._compute_class_weights()
        self._log_distribution()

    def _compute_class_weights(self):
        """Compute class weights"""
        if len(self.records) == 0:
            logger.warning("Empty dataset, cannot compute class weights")
            self.class_weights = np.array([1.0, 1.0])  # Default weights
            return

        labels = [r['label'] for r in self.records]
        label_counts = np.bincount(labels)
        total = len(labels)
        self.class_weights = total / (len(label_counts) * label_counts)
        logger.info(f"Class weights ({self.config.dataset_type}): {dict(enumerate(self.class_weights))}")

    def _log_distribution(self):
        dist = defaultdict(int);
        nodes = [];
        edges = []
        for r in self.records:
            dist[r['label']] += 1

            # Handle graph data that can be list or dict
            graph_data = r['graph']
            if isinstance(graph_data, dict):
                nodes.append(len(graph_data.get('nodes', [])))
                edges.append(len(graph_data.get('edges', [])))
            elif isinstance(graph_data, list):
                nodes.append(len(graph_data))
                edges.append(0)  # If it's a node list, edge count is 0
            else:
                nodes.append(0)
                edges.append(0)

        # Handle empty dataset
        if len(nodes) == 0:
            logger.warning(f"Empty dataset ({self.config.dataset_type}), cannot compute statistics")
            return

        logger.info(
            f"Sample statistics ({self.config.dataset_type}): total={len(self.records)}, label distribution: 0={dist[0]}, 1={dist[1]}, "
            f"avg_nodes={sum(nodes) / len(nodes):.1f}, avg_edges={sum(edges) / len(edges):.1f}")

    def _calc_ast_depth(self, node: Dict) -> int:
        d = 0
        current = node
        while 'parent' in current and current['parent'] and d < 50:  # Prevent infinite loop
            d += 1;
            current = current['parent']
        return min(d, 20)  # Limit maximum depth

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler"""
        weights = [self.class_weights[r['label']] for r in self.records]
        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec['code'], max_length=self.config.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        # Handle graph data that can be list or dict
        graph_data = rec['graph']
        if isinstance(graph_data, dict):
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
        elif isinstance(graph_data, list):
            nodes = graph_data
            edges = []
        else:
            nodes = []
            edges = []

        id2idx = {n.get('id', i): i for i, n in enumerate(nodes)}

        # Enhanced node features
        x = torch.tensor([
            [n.get('lineNumber', 0),
             self.config.node_types.get(n.get('type', ''), 0),
             len(n.get('code', '')),
             self._calc_ast_depth(n),
             int(n.get('is_vulnerable', 0)),
             len(str(n.get('code', ''))),  # Code length
             n.get('lineNumber', 0) / max(1, len(nodes))]  # Relative position
            for n in nodes
        ], dtype=torch.float)

        if edges:
            src = [id2idx.get(e['src'], 0) for e in edges if e.get('src') in id2idx]
            dst = [id2idx.get(e['dst'], 0) for e in edges if e.get('dst') in id2idx]
            if src and dst:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                type_ids = torch.tensor(
                    [self.config.edge_types.get(e.get('label', ''), 0) for e in edges if
                     e.get('src') in id2idx and e.get('dst') in id2idx],
                    dtype=torch.long
                )
                edge_attr = one_hot(type_ids, num_classes=len(self.config.edge_types)).to(torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.config.edge_types)), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config.edge_types)), dtype=torch.float)

        label = torch.tensor(rec['label'], dtype=torch.long)
        return enc, GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr), label


# Enhanced GNN
class EnhancedGNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim

        # Input projection
        self.input_proj = nn.Linear(7, self.hidden_dim)  # 7 features
        self.dropout = nn.Dropout(0.3)

        # Multi-layer GNN
        self.layers = nn.ModuleList()
        for i in range(config.gnn_layers):
            if i == 0:
                layer = GINEConv(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                    ),
                    edge_dim=len(config.edge_types)
                )
            elif i == 1:
                layer = GATv2Conv(self.hidden_dim, self.hidden_dim, edge_dim=len(config.edge_types), heads=4,
                                  concat=False)
            else:
                layer = GCNConv(self.hidden_dim, self.hidden_dim)
            self.layers.append(layer)

        # Norm layers for residual connections
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(config.gnn_layers)])

    def forward(self, x, edge_index, edge_attr):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Multi-layer processing
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x

            if i < 2 and edge_attr.size(0) > 0:  # First two layers use edge attributes
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection
            if x.shape == residual.shape:
                x = x + residual
            x = norm(x)

        return x


# Attention Fusion
class AttentionFusion(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = 512

        # Projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb, graph_emb):
        # Project to same space
        t = self.text_proj(text_emb)  # [B, 512]
        g = self.graph_proj(graph_emb)  # [B, 512]

        # Add sequence dimension for attention computation
        t_seq = t.unsqueeze(1)  # [B, 1, 512]
        g_seq = g.unsqueeze(1)  # [B, 1, 512]

        # Cross attention
        attn_out, _ = self.cross_attention(t_seq, g_seq, g_seq)
        attn_out = attn_out.squeeze(1)  # [B, 512]

        # Gated fusion
        combined = torch.cat([t, attn_out], dim=1)  # [B, 1024]
        gate_weight = self.gate(combined)  # [B, 1]

        fused = gate_weight * t + (1 - gate_weight) * attn_out
        return fused


# Joint Model
class JointModel(nn.Module):
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        super().__init__()

        # Load CodeLlama-13b model from local path
        model_path = config.local_model_dir
        logger.info(f"Loading model from local path: {model_path}")

        hf_conf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.hidden_size = hf_conf.hidden_size
        self.hidden_size = hf_conf.hidden_size

        # Load CodeLlama model
        self.llm = AutoModel.from_pretrained(
            model_path,
            config=hf_conf,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # LoRA configuration
        peft_conf = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=[]
        )
        self.llm = get_peft_model(self.llm, peft_conf)

        # Ensure LoRA parameters are trainable
        for name, param in self.llm.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        if config.use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        self.gnn = EnhancedGNN(config)
        self.fusion = AttentionFusion(config)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, enc: Dict, graph: GeoData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = enc['input_ids'].squeeze(1).to(device)
        attn_mask = enc['attention_mask'].squeeze(1).to(device)

        # Text encoding
        with autocast(device_type=device.type):
            out = self.llm(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            text_emb = out.last_hidden_state[:, 0].to(torch.float32)

        # Graph encoding
        graph = graph.to(device)
        node_emb = self.gnn(graph.x, graph.edge_index, graph.edge_attr)

        # Multiple pooling methods
        graph_emb_mean = global_mean_pool(node_emb, graph.batch)
        graph_emb_max = global_max_pool(node_emb, graph.batch)
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=1)

        # Adapt dimensions
        if graph_emb.size(1) != self.gnn.hidden_dim:
            graph_emb = graph_emb[:, :self.gnn.hidden_dim]

        # Fusion and classification
        fused_emb = self.fusion(text_emb, graph_emb)
        logits = self.classifier(fused_emb)

        return logits, None


# PrimeVul Data Loading
def load_primevul_datasets(config: Config, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """Load PrimeVul datasets"""

    def load_dataset(dataset_name: str, label_file: str) -> List[Dict]:
        """Load single dataset"""
        # Build graph data directory path
        graph_dir = Path(config.primevul_base_dir) / dataset_name
        # Build label file path
        label_path = Path(config.primevul_labels_dir) / label_file

        logger.info(f"Loading dataset ({config.dataset_type}): {dataset_name}")
        logger.info(f"Graph data directory: {graph_dir}")
        logger.info(f"Label file: {label_path}")

        # Read label file
        if not label_path.exists():
            logger.error(f"Label file does not exist: {label_path}")
            return []

        records = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse label line: {e}")
                        continue

        logger.info(f"Read {len(records)} records from label file")

        # Load corresponding graph data
        valid_records = []
        missing_graphs = 0

        for record in records:
            # Use idx field in record as filename (PrimeVul dataset uses idx as unique identifier)
            record_id = str(record.get('idx', ''))
            if not record_id:
                missing_graphs += 1
                continue

            graph_file = graph_dir / f"{record_id}.json"

            if graph_file.exists():
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)

                    # Debug: print first graph data structure
                    if len(valid_records) == 0:
                        logger.info(f"Graph data sample structure ({config.dataset_type}): {type(graph_data)}")
                        if isinstance(graph_data, list):
                            logger.info(f"Graph data is DataFlowSlice array, length: {len(graph_data)}")

                    record['graph'] = graph_data
                    # Rename target field to label for consistency
                    record['label'] = record.get('target', 0)
                    # Use func field as code
                    record['code'] = record.get('func', '')
                    valid_records.append(record)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to read graph file {graph_file}: {e}")
                    missing_graphs += 1
            else:
                missing_graphs += 1

        logger.info(f"Successfully loaded {len(valid_records)} records, missing graph data {missing_graphs} records")
        return valid_records

    # Load train, validation, test datasets
    train_records = load_dataset(config.datasets['train'], config.label_files['train'])
    valid_records = load_dataset(config.datasets['valid'], config.label_files['valid'])
    test_records = load_dataset(config.datasets['test'], config.label_files['test'])

    # Data quality filtering
    train_records = filter_low_quality_data(train_records, config)
    valid_records = filter_low_quality_data(valid_records, config)
    test_records = filter_low_quality_data(test_records, config)

    # Check for valid data
    if len(train_records) == 0 or len(valid_records) == 0 or len(test_records) == 0:
        logger.error(f"Dataset loading failed, empty dataset exists ({config.dataset_type})!")
        logger.error(f"train: {len(train_records)}, valid: {len(valid_records)}, test: {len(test_records)}")
        raise ValueError("Datasets cannot be empty")

    # Create dataset objects
    train_dataset = VulnDataset(train_records, tokenizer, config)
    valid_dataset = VulnDataset(valid_records, tokenizer, config)
    test_dataset = VulnDataset(test_records, tokenizer, config)

    logger.info(
        f"Final dataset sizes ({config.dataset_type}): train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


# Batch processing
def collate_fn(batch: List[Tuple]) -> Tuple[Dict, GeoData, torch.Tensor]:
    encs, graphs, labels = zip(*batch)
    enc = {
        'input_ids': torch.cat([e['input_ids'] for e in encs], dim=0),
        'attention_mask': torch.cat([e['attention_mask'] for e in encs], dim=0)
    }
    graph_batch = Batch.from_data_list(graphs)
    label_batch = torch.stack(labels)
    return enc, graph_batch, label_batch

# Enhanced evaluation function
def evaluate(model: nn.Module, loader: GeoDataLoader, device: torch.device, config: Config):
    """Enhanced evaluation function with VDS and pairwise metrics"""
    model.eval()
    preds, labs, probs = [], [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for enc, graph, lab in loader:
            lab = lab.to(device)
            logits, _ = model(enc, graph, device)
            loss = criterion(logits, lab)
            total_loss += loss.item()

            # Get probabilities and predictions
            prob = torch.softmax(logits, dim=1)
            probs.extend(prob[:, 1].cpu().tolist())  # Vulnerability class probability
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labs.extend(lab.cpu().tolist())

    # Basic metrics
    acc = sum(p == l for p, l in zip(preds, labs)) / len(labs)
    prec = precision_score(labs, preds, zero_division=0)
    rec = recall_score(labs, preds, zero_division=0)
    f1 = f1_score(labs, preds, zero_division=0)
    cm = confusion_matrix(labs, preds)
    avg_loss = total_loss / len(loader)

    # Calculate VDS metrics using config's VDS threshold
    vds_metrics = calculate_vds(np.array(labs), np.array(probs), config.vds_fpr_threshold)

    # If paired dataset, calculate pairwise metrics
    pairwise_metrics = {}
    if config.is_paired:
        pairwise_metrics = calculate_pairwise_metrics(np.array(labs), np.array(preds))

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'loss': avg_loss,
        **vds_metrics  # Unpack VDS metrics
    }

    # If pairwise metrics exist, add to results
    if pairwise_metrics:
        results.update(pairwise_metrics)

    return results


# Learning rate scheduling
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# Train single dataset
def train_single_dataset(dataset_type: str):
    """Train single dataset (balanced or unbalanced)"""
    config = Config(dataset_type)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("=" * 80)
    logger.info(f"Starting training {dataset_type.upper()} dataset")
    logger.info(f"PrimeVul dataset + CodeLlama-13b joint fine-tuning")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model path: {config.local_model_dir}")
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info(f"GNN config: hidden_dim={config.gnn_hidden_dim}, layers={config.gnn_layers}")
    logger.info(f"VDS FPR threshold: {config.vds_fpr_threshold:.3f}")
    logger.info(f"Is paired dataset: {config.is_paired}")
    logger.info("=" * 80)

    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.local_model_dir,
        trust_remote_code=True,
        use_fast=True
    )

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading {dataset_type} PrimeVul dataset...")
    train_ds, val_ds, test_ds = load_primevul_datasets(config, tokenizer)

    # Use weighted sampler for class balance
    if config.use_class_weights and hasattr(train_ds, 'get_sample_weights') and not config.is_paired:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = GeoDataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, shuffle=shuffle,
                                 collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = GeoDataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=2,
                               pin_memory=True)
    test_loader = GeoDataLoader(test_ds, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=2,
                                pin_memory=True)

    logger.info(f"Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    logger.info("Initializing model...")
    model = JointModel(config, tokenizer).to(config.device)

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Optimizer configuration
    optimizer = optim.AdamW([
        {'params': model.llm.parameters(), 'lr': config.lr * 0.1},  # Smaller LR for LLM
        {'params': model.gnn.parameters(), 'lr': config.lr},
        {'params': model.fusion.parameters(), 'lr': config.lr},
        {'params': model.classifier.parameters(), 'lr': config.lr}
    ], weight_decay=1e-4)

    # Learning rate scheduling
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_vds = 0.0
    no_improve = 0
    model_save_path = f"best_model_{dataset_type}.pt"

    logger.info("Starting training...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (enc, graph, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            labels = labels.to(config.device)

            with autocast(device_type=config.device.type):
                logits, _ = model(enc, graph, config.device)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss={loss.item():.4f} LR={current_lr:.2e}")

        avg_train_loss = epoch_loss / num_batches
        val_results = evaluate(model, val_loader, config.device, config)

        logger.info(f"Epoch {epoch} completed ({dataset_type}):")
        logger.info(f"  Training loss: {avg_train_loss:.4f}, Validation loss: {val_results['loss']:.4f}")
        logger.info(f"  Validation metrics: Acc={val_results['accuracy']:.4f}, Prec={val_results['precision']:.4f}, "
                    f"Rec={val_results['recall']:.4f}, F1={val_results['f1']:.4f}")
        logger.info(f"  VDS metrics: VDS_Score={val_results['vds_score']:.4f}, VDS_FNR={val_results['vds_fnr']:.4f}, "
                    f"VDS_FPR={val_results['vds_fpr']:.4f}")

        if config.is_paired:
            logger.info(f"  Pairwise metrics: P-C={val_results['pc']:.4f}, P-V={val_results['pv']:.4f}, "
                        f"P-B={val_results['pb']:.4f}, P-R={val_results['pr']:.4f}")

        logger.info(f"  Confusion matrix:\n{val_results['confusion_matrix']}")

        # Determine best model based on F1 and VDS combination
        current_score = val_results['f1'] * 0.7 + val_results['vds_score'] * 0.3
        if current_score > best_f1 + config.min_delta:
            best_f1, best_vds, no_improve = current_score, val_results['vds_score'], 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"🎉 Found best model ({dataset_type}), combined score={current_score:.4f}")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info(f"Early stopping ({dataset_type}): {config.patience} epochs without improvement")
                break

    logger.info(f"Loading best model and testing ({dataset_type})...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_results = evaluate(model, test_loader, config.device, config)

    # Output final results
    logger.info("=" * 80)
    logger.info(f"🎯 {dataset_type.upper()} dataset final test results:")
    logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    logger.info(f"  F1 score: {test_results['f1']:.4f}")
    logger.info(f"  VDS score: {test_results['vds_score']:.4f}")
    logger.info(f"  VDS FNR: {test_results['vds_fnr']:.4f}")
    logger.info(f"  VDS FPR: {test_results['vds_fpr']:.4f}")
    logger.info(f"  VDS threshold: {test_results['vds_threshold']:.4f}")

    if config.is_paired:
        logger.info(f"  Pairwise correct prediction (P-C): {test_results['pc']:.4f}")
        logger.info(f"  Pairwise vulnerable prediction (P-V): {test_results['pv']:.4f}")
        logger.info(f"  Pairwise safe prediction (P-B): {test_results['pb']:.4f}")
        logger.info(f"  Pairwise reverse prediction (P-R): {test_results['pr']:.4f}")
        logger.info(f"  Total pairs: {test_results['total_pairs']}")

    logger.info(f"  Confusion matrix:\n{test_results['confusion_matrix']}")
    logger.info("=" * 80)

    return test_results


# Main process
def main():
    parser = argparse.ArgumentParser(description="PrimeVul vulnerability detection training")
    parser.add_argument("--dataset", choices=["balanced", "unbalanced", "both"],
                        default="balanced", help="Choose dataset type to train")
    args = parser.parse_args()

    # Set environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    })
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    results_summary = {}

    if args.dataset == "both":
        # Train both datasets
        logger.info("🚀 Starting training on balanced and unbalanced datasets")

        # Train unbalanced dataset
        logger.info("\n" + "=" * 100)
        logger.info("Phase 1: Training unbalanced dataset")
        logger.info("=" * 100)
        results_summary["unbalanced"] = train_single_dataset("unbalanced")

        # Train balanced dataset
        logger.info("\n" + "=" * 100)
        logger.info("Phase 2: Training balanced dataset")
        logger.info("=" * 100)
        results_summary["balanced"] = train_single_dataset("balanced")

        # Output comparison results
        logger.info("\n" + "=" * 100)
        logger.info("📊 Comparison of both datasets:")
        logger.info("=" * 100)

        for dataset_type, results in results_summary.items():
            logger.info(f"\n{dataset_type.upper()} dataset:")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  F1 score: {results['f1']:.4f}")
            logger.info(f"  VDS score: {results['vds_score']:.4f}")
            if 'pc' in results:  # If pairwise metrics exist
                logger.info(f"  Pairwise correct rate: {results['pc']:.4f}")

        logger.info("=" * 100)

    else:
        # Train single dataset
        logger.info(f"🚀 Starting training on {args.dataset} dataset")
        results_summary[args.dataset] = train_single_dataset(args.dataset)

    logger.info("✅ Training completed!")


if __name__ == "__main__":
    main()