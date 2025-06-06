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
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# PyTorch Geometric imports
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv, global_mean_pool
from torch.nn.functional import one_hot


# ------------- Memory Optimization Utils -------------
def cleanup_memory():
    """Deep GPU and CPU memory cleanup"""
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
    """Optimize CUDA settings"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,roundup_power2_divisions:16'


def force_cleanup_model(model):
    """Force cleanup model GPU memory"""
    if model is not None:
        model.cpu()
        del model
        cleanup_memory()


def reset_cuda_peak_memory():
    """Reset CUDA peak memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()


# ------------- Logging -------------
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


# ------------- Configuration -------------
class Config:
    def __init__(self, dataset_type: str = "balanced"):
        # Dataset configuration
        self.dataset_type = dataset_type
        self.data_dir = Path("../../data/primevul_process")
        self.cpg_base_dir = Path("../../data/primevul_graphson_files")

        if dataset_type == "balanced":
            self.train_file = self.data_dir / "PrimeVul_balanced_train.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_balanced_valid.jsonl"
            self.test_file = self.data_dir / "PrimeVul_balanced_test.jsonl"
            self.train_cpg_dir = self.cpg_base_dir / "PrimeVul_balanced_train"
            self.valid_cpg_dir = self.cpg_base_dir / "PrimeVul_balanced_valid"
            self.test_cpg_dir = self.cpg_base_dir / "PrimeVul_balanced_test"
        else:
            self.train_file = self.data_dir / "PrimeVul_unbalanced_train_sampled.jsonl"
            self.valid_file = self.data_dir / "PrimeVul_unbalanced_valid_sampled.jsonl"
            self.test_file = self.data_dir / "PrimeVul_unbalanced_test_sampled.jsonl"
            self.train_cpg_dir = self.cpg_base_dir / "PrimeVul_unbalanced_train_sampled"
            self.valid_cpg_dir = self.cpg_base_dir / "PrimeVul_unbalanced_valid_sampled"
            self.test_cpg_dir = self.cpg_base_dir / "PrimeVul_unbalanced_test_sampled"

        # Data sampling configuration for debugging
        self.sample_data = True
        self.train_sample_ratio = 0.3
        self.valid_sample_ratio = 0.5
        self.test_sample_ratio = 0.5

        # GNN configuration - Pure GCN 3-layer structure
        self.gnn_hidden_dim = 256
        self.gnn_layers = 3
        self.gat_heads = 4  # Reserved but not used
        self.node_types = {
            'UNKNOWN': 0, 'METHOD': 1, 'BINDING': 2, 'LOCAL': 3, 'BLOCK': 4, 'CALL': 5,
            'CONTROL_STRUCTURE': 6, 'RETURN': 7, 'TYPE': 8, 'IDENTIFIER': 9,
            'LITERAL': 10, 'METHOD_RETURN': 11, 'FILE': 12, 'NAMESPACE_BLOCK': 13,
            'TYPE_DECL': 14, 'MEMBER': 15, 'MODIFIER': 16, 'ANNOTATION': 17,
        }
        self.edge_types = {
            'UNKNOWN': 0, 'REF': 1, 'AST': 2, 'EVAL_TYPE': 3, 'CALL': 4,
            'ARGUMENT': 5, 'CFG': 6, 'DOMINATE': 7, 'POST_DOMINATE': 8,
            'REACHING_DEF': 9, 'CONTAINS': 10, 'PARAMETER_LINK': 11,
            'SOURCE_FILE': 12,
        }
        self.num_node_features = 4

        # Graph processing configuration
        self.min_graph_nodes = 10  # Skip graphs with fewer nodes
        self.max_graph_nodes = 1000  # Sample if exceeds this number

        # Training configuration
        self.batch_size = 64
        self.gradient_accumulation_steps = 2
        self.lr = 5e-4
        self.epochs = 30
        self.seed = 42
        self.device = torch.device("cuda:0")
        self.warmup_steps = 200
        self.min_lr = 1e-5
        self.patience = 5
        self.min_delta = 0.001

        # Save configuration
        self.checkpoint_dir = f"../../result/pure_gcn_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_every_n_epochs = 5

        # VD-S configuration
        self.fpr_threshold = 0.005

        # Memory optimization configuration
        self.dataloader_pin_memory = True
        self.num_workers = 2
        self.cleanup_freq = 10
        self.test_batch_size = 32


# ------------- Dataset -------------
class PrimeVulGNNDataset(Dataset):
    def __init__(self, file_path: Path, cpg_dir: Path, config: Config, split_name: str = "train"):
        self.config = config
        self.cpg_dir = cpg_dir
        self.split_name = split_name
        self.records = []
        self.valid_samples = []  # Store valid sample indices

        # Feature statistics to prevent data leakage
        self.feature_stats = None

        # Read JSONL file
        all_records = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                all_records.append({
                    'idx': data['idx'],
                    'label': data['target']
                })

        # Data sampling
        if config.sample_data:
            if split_name == "train":
                sample_size = int(len(all_records) * config.train_sample_ratio)
            elif split_name == "valid":
                sample_size = int(len(all_records) * config.valid_sample_ratio)
            else:  # test
                sample_size = int(len(all_records) * config.test_sample_ratio)

            # Ensure balanced sampling
            pos_samples = [r for r in all_records if r['label'] == 1]
            neg_samples = [r for r in all_records if r['label'] == 0]

            pos_sample_size = min(sample_size // 2, len(pos_samples))
            neg_sample_size = min(sample_size // 2, len(neg_samples))

            # Different randomness for different splits
            split_seeds = {"train": 0, "valid": 1000, "test": 2000}
            np.random.seed(config.seed + split_seeds.get(split_name, 0))
            selected_pos = np.random.choice(len(pos_samples), pos_sample_size, replace=False)
            selected_neg = np.random.choice(len(neg_samples), neg_sample_size, replace=False)

            self.records = [pos_samples[i] for i in selected_pos] + [neg_samples[i] for i in selected_neg]
            np.random.shuffle(self.records)
        else:
            self.records = all_records

        logger.info(f"{split_name} original samples: {len(all_records)}, using: {len(self.records)}")

        # Preprocessing: find valid samples
        self._preprocess_samples()
        self._log_distribution()

    def set_feature_stats(self, feature_stats):
        """Set feature statistics from training set to prevent data leakage"""
        self.feature_stats = feature_stats

    def _preprocess_samples(self):
        """Preprocess samples to find valid CPG files and graphs"""
        logger.info(f"Preprocessing {self.split_name} samples...")
        valid_count = 0

        for idx, rec in enumerate(self.records):
            cpg_file = self.cpg_dir / f"{rec['idx']}.json"

            if not cpg_file.exists():
                continue

            try:
                with open(cpg_file, 'r') as f:
                    cpg_data = json.load(f)

                cpg_content = cpg_data.get('@value', cpg_data)
                raw_cpg_vertices = cpg_content.get('vertices', [])

                if len(raw_cpg_vertices) >= self.config.min_graph_nodes:
                    self.valid_samples.append(idx)
                    valid_count += 1

            except Exception as e:
                logger.warning(f"Error processing CPG {rec['idx']}: {e}")
                continue

        logger.info(f"{self.split_name} valid samples: {valid_count}/{len(self.records)}")

    def _log_distribution(self):
        """Log label distribution of valid samples"""
        if not self.valid_samples:
            logger.warning(f"{self.split_name} has no valid samples!")
            return

        dist = {0: 0, 1: 0}
        for idx in self.valid_samples:
            dist[self.records[idx]['label']] += 1
        logger.info(f"{self.split_name} valid sample distribution: {dist}")

    def compute_feature_stats(self):
        """Compute feature statistics for training set only"""
        if self.split_name != "train":
            raise ValueError("Feature statistics can only be computed on training set!")

        logger.info("Computing training set feature statistics...")
        features_list = []

        # Random sampling for statistics computation
        sample_indices = np.random.choice(self.valid_samples,
                                          min(500, len(self.valid_samples)),
                                          replace=False)

        for idx in sample_indices:
            rec = self.records[idx]
            cpg_file = self.cpg_dir / f"{rec['idx']}.json"

            try:
                with open(cpg_file, 'r') as f:
                    cpg_data = json.load(f)

                cpg_content = cpg_data.get('@value', cpg_data)
                raw_cpg_vertices = cpg_content.get('vertices', [])

                for v_data in raw_cpg_vertices[:50]:  # Max 50 nodes per graph for statistics
                    line_num = self._extract_node_properties(v_data, 'LINE_NUMBER', 0)
                    if line_num == 0:
                        line_num = self._extract_node_properties(v_data, 'lineNumber', 0)

                    node_type_encoded = self.config.node_types.get(
                        v_data.get('label', 'UNKNOWN').upper(),
                        self.config.node_types.get('UNKNOWN', 0)
                    )

                    code_str = self._extract_node_properties(v_data, 'CODE', "")
                    if not isinstance(code_str, str):
                        code_str = ""

                    features_list.append([
                        float(line_num),
                        float(node_type_encoded),
                        float(len(code_str)),
                        1.0  # ast_depth placeholder
                    ])
            except:
                continue

        if features_list:
            features_array = np.array(features_list)
            feature_means = np.mean(features_array, axis=0)
            feature_stds = np.std(features_array, axis=0)
            feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

            self.feature_stats = {
                'means': feature_means,
                'stds': feature_stds
            }

            logger.info(f"Feature means: {feature_means}")
            logger.info(f"Feature stds: {feature_stds}")
        else:
            self.feature_stats = {
                'means': np.zeros(self.config.num_node_features),
                'stds': np.ones(self.config.num_node_features)
            }

        return self.feature_stats

    def _extract_node_properties(self, cpg_node_data: Dict, property_name: str, default_value: any = 0):
        """Extract properties from CPG node data"""
        if "properties" in cpg_node_data:
            props = cpg_node_data["properties"]
            if property_name in props:
                prop_list = props[property_name]
                if isinstance(prop_list, list) and len(prop_list) > 0:
                    value_dict = prop_list[0]
                    if isinstance(value_dict, dict) and "@value" in value_dict:
                        return value_dict["@value"]
        return default_value

    def _calculate_ast_depths(self, nodes_info: Dict[int, Dict], cpg_edges: List[Dict], id2idx: Dict[int, int]) -> Dict[
        int, int]:
        """Calculate AST depth for each node using BFS"""
        adj = defaultdict(list)
        rev_adj = defaultdict(list)

        for edge in cpg_edges:
            if edge.get('label') == 'AST':
                src_id = edge['outV']['@value'] if isinstance(edge['outV'], dict) else edge['outV']
                dst_id = edge['inV']['@value'] if isinstance(edge['inV'], dict) else edge['inV']
                if src_id in id2idx and dst_id in id2idx:
                    adj[id2idx[src_id]].append(id2idx[dst_id])
                    rev_adj[id2idx[dst_id]].append(id2idx[src_id])

        depths = {node_idx: 1 for node_idx in range(len(id2idx))}

        # Find root nodes
        roots = []
        for node_idx in range(len(id2idx)):
            if node_idx in adj and node_idx not in rev_adj:
                roots.append(node_idx)

        if not roots:
            max_degree = 0
            for node_idx in range(len(id2idx)):
                degree = len(adj[node_idx]) + len(rev_adj[node_idx])
                if degree > max_degree:
                    max_degree = degree
                    roots = [node_idx]

        # BFS to calculate depths
        for root in roots:
            queue = deque([(root, 1)])
            visited = set()

            while queue:
                node_idx, depth = queue.popleft()
                if node_idx in visited:
                    continue
                visited.add(node_idx)
                depths[node_idx] = max(depths[node_idx], depth)

                for child in adj[node_idx]:
                    if child not in visited:
                        queue.append((child, depth + 1))

        return depths

    def _smart_sample_nodes(self, nodes_info_by_cpg_id: Dict, edges: List[Dict], max_nodes: int):
        """Smart node sampling to keep important nodes"""
        if len(nodes_info_by_cpg_id) <= max_nodes:
            return nodes_info_by_cpg_id

        # Calculate node importance (degree + special type weighting)
        node_scores = defaultdict(float)
        important_types = {'METHOD', 'CALL', 'CONTROL_STRUCTURE', 'RETURN'}

        # Degree scoring
        for edge in edges:
            src_id = edge['outV']['@value'] if isinstance(edge['outV'], dict) else edge['outV']
            dst_id = edge['inV']['@value'] if isinstance(edge['inV'], dict) else edge['inV']
            if src_id in nodes_info_by_cpg_id:
                node_scores[src_id] += 1
            if dst_id in nodes_info_by_cpg_id:
                node_scores[dst_id] += 1

        # Type weighting
        for node_id, node_info in nodes_info_by_cpg_id.items():
            if node_info['label'].upper() in important_types:
                node_scores[node_id] += 5

        # Select top nodes
        sorted_nodes = sorted(nodes_info_by_cpg_id.keys(),
                              key=lambda x: node_scores[x], reverse=True)

        selected_nodes = set(sorted_nodes[:max_nodes])
        return {k: v for k, v in nodes_info_by_cpg_id.items() if k in selected_nodes}

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int):
        # Get actual index of valid sample
        actual_idx = self.valid_samples[idx]
        rec = self.records[actual_idx]

        cpg_file = self.cpg_dir / f"{rec['idx']}.json"

        try:
            with open(cpg_file, 'r') as f:
                cpg_data = json.load(f)

            cpg_content = cpg_data.get('@value', cpg_data)
            raw_cpg_edges = cpg_content.get('edges', [])
            raw_cpg_vertices = cpg_content.get('vertices', [])

            # Collect node information
            nodes_info_by_cpg_id = {}
            for v_data in raw_cpg_vertices:
                node_id = v_data['id']['@value'] if isinstance(v_data.get('id'), dict) else v_data.get('id')
                if node_id is None:
                    continue
                nodes_info_by_cpg_id[node_id] = {
                    'label': v_data.get('label', 'UNKNOWN'),
                    'properties_raw': v_data
                }

            # Supplement node info from edges
            for edge in raw_cpg_edges:
                for v_key, v_label_key in [('outV', 'outVLabel'), ('inV', 'inVLabel')]:
                    node_id = edge[v_key]['@value'] if isinstance(edge[v_key], dict) else edge[v_key]
                    if node_id not in nodes_info_by_cpg_id:
                        nodes_info_by_cpg_id[node_id] = {
                            'label': edge.get(v_label_key, 'UNKNOWN'),
                            'properties_raw': {}
                        }

            # Smart sampling for large graphs
            if len(nodes_info_by_cpg_id) > self.config.max_graph_nodes:
                nodes_info_by_cpg_id = self._smart_sample_nodes(
                    nodes_info_by_cpg_id, raw_cpg_edges, self.config.max_graph_nodes)

            # Create node ID mapping
            sorted_cpg_node_ids = sorted(list(nodes_info_by_cpg_id.keys()))
            id2idx = {cpg_id: i for i, cpg_id in enumerate(sorted_cpg_node_ids)}
            num_nodes = len(sorted_cpg_node_ids)

            # Create node features
            x = torch.zeros((num_nodes, self.config.num_node_features), dtype=torch.float)

            # Calculate AST depths
            ast_depths = self._calculate_ast_depths(nodes_info_by_cpg_id, raw_cpg_edges, id2idx)

            for i, cpg_id in enumerate(sorted_cpg_node_ids):
                node_data = nodes_info_by_cpg_id[cpg_id]
                node_label_str = node_data.get('label', 'UNKNOWN')

                # Extract features
                line_num = self._extract_node_properties(node_data['properties_raw'], 'LINE_NUMBER', 0)
                if line_num == 0:
                    line_num = self._extract_node_properties(node_data['properties_raw'], 'lineNumber', 0)

                node_type_encoded = self.config.node_types.get(
                    node_label_str.upper(),
                    self.config.node_types.get('UNKNOWN', 0)
                )

                code_str = self._extract_node_properties(node_data['properties_raw'], 'CODE', "")
                if not isinstance(code_str, str):
                    code_str = ""

                x[i, 0] = float(line_num)
                x[i, 1] = float(node_type_encoded)
                x[i, 2] = float(len(code_str))
                x[i, 3] = float(ast_depths.get(i, 1))

            # Feature normalization using training set statistics
            if self.feature_stats is not None:
                x_np = x.numpy()
                x_normalized = (x_np - self.feature_stats['means']) / self.feature_stats['stds']
                x = torch.tensor(x_normalized, dtype=torch.float)

            # Process edges
            src_indices, dst_indices = [], []
            edge_type_ids = []

            for edge in raw_cpg_edges:
                src_id = edge['outV']['@value'] if isinstance(edge['outV'], dict) else edge['outV']
                dst_id = edge['inV']['@value'] if isinstance(edge['inV'], dict) else edge['inV']

                if src_id in id2idx and dst_id in id2idx:
                    src_indices.append(id2idx[src_id])
                    dst_indices.append(id2idx[dst_id])
                    edge_label = edge.get('label', 'UNKNOWN')
                    edge_type_ids.append(
                        self.config.edge_types.get(
                            edge_label.upper(),
                            self.config.edge_types.get('UNKNOWN', 0)
                        )
                    )

            if src_indices:
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                edge_type_tensor = torch.tensor(edge_type_ids, dtype=torch.long)
                edge_attr = one_hot(edge_type_tensor, num_classes=len(self.config.edge_types)).to(torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.config.edge_types)), dtype=torch.float)

            label = torch.tensor(rec['label'], dtype=torch.long)
            graph_data = GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

            return graph_data, label

        except Exception as e:
            logger.error(f"Error processing sample {rec['idx']}: {e}")
            raise


# ------------- Pure GCN Model -------------
class VulnDetectionGNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Three-layer GCN
        self.conv1 = GCNConv(config.num_node_features, config.gnn_hidden_dim)
        self.conv2 = GCNConv(config.gnn_hidden_dim, config.gnn_hidden_dim)
        self.conv3 = GCNConv(config.gnn_hidden_dim, config.gnn_hidden_dim)

        # Classification layers
        self.graph_classifier = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim // 2),
            nn.BatchNorm1d(config.gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(config.gnn_hidden_dim // 2, config.gnn_hidden_dim // 4),
            nn.BatchNorm1d(config.gnn_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(config.gnn_hidden_dim // 4, 2)
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr, batch):
        if x.numel() == 0:
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
            return torch.zeros((batch_size, 2), device=x.device)

        # Three GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)

        # Graph-level pooling
        graph_emb = global_mean_pool(x, batch)

        # Classification
        logits = self.graph_classifier(graph_emb)
        return logits


# ------------- Data Loading -------------
def collate_fn(batch: List[Tuple]) -> Tuple[Batch, torch.Tensor]:
    """Custom batch collation function"""
    graphs, labels = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    label_batch = torch.stack(list(labels))
    return graph_batch, label_batch


# ------------- Evaluation Metrics -------------
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


def evaluate_paired_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate paired predictions (for balanced dataset only)"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for graph_batch, labels in loader:
            graph_batch = graph_batch.to(device, non_blocking=True)
            logits = model(graph_batch.x.float(), graph_batch.edge_index,
                           graph_batch.edge_attr.float(), graph_batch.batch)
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


# ------------- Evaluation Function -------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             calculate_vds_metric: bool = True, is_paired: bool = False):
    model.eval()
    preds, labs, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, (graph_batch, lab) in enumerate(loader):
            graph_batch = graph_batch.to(device, non_blocking=True)
            logits = model(graph_batch.x.float(), graph_batch.edge_index,
                           graph_batch.edge_attr.float(), graph_batch.batch)
            probs = F.softmax(logits, dim=1)

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labs.extend(lab.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

            if batch_idx % 50 == 0:
                cleanup_memory()

    all_probs = np.array(all_probs) if calculate_vds_metric else None
    metrics = calculate_metrics(preds, labs, all_probs)

    if is_paired:
        pair_metrics = evaluate_paired_predictions(model, loader, device)
        metrics.update(pair_metrics)

    return metrics


# ------------- Training Scheduler -------------
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


# ------------- Save and Load -------------
def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                    best_metrics: Dict, checkpoint_dir: Path, is_best: bool = False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics
    }

    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved: {best_path}")


def load_model_for_testing(config: Config, checkpoint_path: Path):
    """Load model specifically for testing"""
    logger.info(f"Loading model for testing... {get_gpu_memory_info()}")

    test_model = VulnDetectionGNN(config).to(config.device)
    logger.info(f"Test model initialized {get_gpu_memory_info()}")

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Test model weights loaded {get_gpu_memory_info()}")

    return test_model, checkpoint['best_metrics']


# ------------- Main Training Function -------------
def train_model(config: Config):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    optimize_cuda_settings()
    reset_cuda_peak_memory()
    logger.info(f"Initial {get_gpu_memory_info()}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PrimeVulGNNDataset(config.train_file, config.train_cpg_dir, config, "train")

    # Compute feature statistics from training set to prevent data leakage
    feature_stats = train_dataset.compute_feature_stats()

    # Load validation and test sets with feature statistics
    valid_dataset = PrimeVulGNNDataset(config.valid_file, config.valid_cpg_dir, config, "valid")
    valid_dataset.set_feature_stats(feature_stats)

    test_dataset = PrimeVulGNNDataset(config.test_file, config.test_cpg_dir, config, "test")
    test_dataset.set_feature_stats(feature_stats)

    # Check dataset sizes
    if len(train_dataset) == 0:
        raise ValueError("Training set has no valid samples!")
    if len(valid_dataset) == 0:
        raise ValueError("Validation set has no valid samples!")
    if len(test_dataset) == 0:
        raise ValueError("Test set has no valid samples!")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.num_workers > 0
    )

    logger.info(f"Dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    # Calculate class weights
    label_counts = {0: 0, 1: 0}
    for idx in train_dataset.valid_samples:
        label_counts[train_dataset.records[idx]['label']] += 1

    total_samples = len(train_dataset)
    class_weights = torch.tensor([
        total_samples / (2 * label_counts[0]),
        total_samples / (2 * label_counts[1])
    ], dtype=torch.float32).to(config.device)

    logger.info(f"Training set class distribution: {label_counts}")
    logger.info(f"Class weights: {class_weights}")

    # Initialize model
    logger.info("Initializing Pure GCN model...")
    model = VulnDetectionGNN(config).to(config.device)
    logger.info(f"Model loaded {get_gpu_memory_info()}")

    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        eta_min=config.min_lr
    )

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

        for batch_idx, (graph_batch, labels) in enumerate(train_loader):
            graph_batch = graph_batch.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(graph_batch.x.float(), graph_batch.edge_index,
                               graph_batch.edge_attr.float(), graph_batch.batch)
                loss = criterion(logits, labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

                if global_step % config.cleanup_freq == 0:
                    cleanup_memory()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % 50 == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss={current_loss:.4f} LR={current_lr:.2e} {get_gpu_memory_info()}"
                )

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        cleanup_memory()

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

        if is_paired:
            logger.info(f"Paired metrics: P-C={val_metrics['P-C']:.4f}, P-V={val_metrics['P-V']:.4f}, "
                        f"P-B={val_metrics['P-B']:.4f}, P-R={val_metrics['P-R']:.4f}")

        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            Path(config.checkpoint_dir), is_best=False)

        # Save best model
        if val_metrics['f1'] > best_f1 + config.min_delta:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            Path(config.checkpoint_dir), is_best=True)
            logger.info(f"Best model found, F1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info(f"Early stopping: no improvement for {config.patience} epochs")
                break

        cleanup_memory()

    # Training completed, cleanup variables
    logger.info("Training completed, starting cleanup...")
    del optimizer, scheduler, scaler, criterion
    force_cleanup_model(model)
    del train_loader, valid_loader
    cleanup_memory()
    logger.info(f"Training variables cleaned {get_gpu_memory_info()}")

    # Final testing
    logger.info("Starting final testing...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    # Reload best model
    best_checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
    test_model, _ = load_model_for_testing(config, best_checkpoint_path)

    # Execute testing
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

    # Save final model
    final_model_path = Path(config.checkpoint_dir) / 'final_model.pt'
    torch.save(test_model.state_dict(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Final cleanup
    force_cleanup_model(test_model)
    del test_loader, test_dataset, train_dataset, valid_dataset
    cleanup_memory()
    logger.info(f"Final cleanup completed {get_gpu_memory_info()}")


# ------------- Main Function -------------
def main():
    # Train balanced dataset first
    logger.info("=" * 80)
    logger.info("Starting training on balanced dataset (Pure GCN model)")
    logger.info("=" * 80)

    try:
        config_balanced = Config(dataset_type="balanced")
        train_model(config_balanced)
    except Exception as e:
        logger.error(f"Balanced dataset training failed: {e}")
        cleanup_memory()
        raise

    # Cleanup GPU memory
    cleanup_memory()
    reset_cuda_peak_memory()

    # Train unbalanced dataset
    logger.info("=" * 80)
    logger.info("Starting training on unbalanced dataset (Pure GCN model)")
    logger.info("=" * 80)

    try:
        config_unbalanced = Config(dataset_type="unbalanced")
        train_model(config_unbalanced)
    except Exception as e:
        logger.error(f"Unbalanced dataset training failed: {e}")
        cleanup_memory()
        raise


if __name__ == "__main__":
    # Set environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_LAUNCH_BLOCKING': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:64,roundup_power2_divisions:16'
    })

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info("Starting Pure GCN vulnerability detection model training...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    try:
        main()
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        cleanup_memory()
        raise
    finally:
        cleanup_memory()
        logger.info("Program execution completed, GPU memory cleaned")