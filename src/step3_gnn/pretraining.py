#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATv2Conv, GINEConv, global_mean_pool, global_max_pool
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn.functional import one_hot
import wandb

# Set matplotlib to use fonts that support unicode
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Constants
DEFAULT_LOG_FILE = "gnn_pretraining.log"
DEFAULT_SEED = 42
MIN_NODES = 2
MAX_NODES = 50

# Node types mapping
NODE_TYPES = {
    'IDENTIFIER': 1, 'LITERAL': 2, 'METHOD': 3, 'CALL': 4,
    'CONTROL_STRUCTURE': 5, 'OPERATOR': 6, 'UNKNOWN': 7
}

# Edge types mapping
EDGE_TYPES = {
    'REACHING_DEF': 1, 'DATA_FLOW': 2, 'CONTROL_FLOW': 3, 'CALL': 4
}

# Vulnerability and safety patterns
VULN_PATTERNS = ['strcpy', 'malloc', 'free', 'memcpy', 'sprintf', 'gets', 'scanf', 'strcat']
SAFE_PATTERNS = ['__libc_enable_secure', 'strncpy', 'snprintf', 'fgets', 'strncat']


# Configuration
@dataclass
class GNNConfig:
    """Configuration for GNN pretraining"""

    # Data paths
    json_path: str = "../../data/merged_cvefixes_key_node.json"
    subgraph_dir: str = "../../data/subgraphs"
    output_dir: str = "../../result"
    log_file: str = DEFAULT_LOG_FILE

    # Model architecture
    gnn_hidden_dim: int = 128
    gnn_layers: int = 2
    feature_dim: int = 10

    # Training parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 30
    seed: int = DEFAULT_SEED

    # Data parameters
    max_length: int = 512
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    min_nodes: int = MIN_NODES
    max_nodes: int = MAX_NODES

    # Training optimization
    warmup_steps: int = 100
    min_lr: float = 1e-5
    patience: int = 15
    min_delta: float = 0.001
    use_class_weights: bool = True
    pos_weight: float = 1.5

    # Core subgraph extraction
    top_k_ratio: float = 0.5

    # WandB configuration
    wandb_api_key: Optional[str] = None
    wandb_project: str = "gnn-vulnerability-detection"
    wandb_name: str = "gnn-pretraining"
    use_wandb: bool = False

    def __post_init__(self):
        """Post-initialization setup"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_types = NODE_TYPES
        self.edge_types = EDGE_TYPES

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def setup_logging(config: GNNConfig) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(config.log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")

    return logger


def filter_low_quality_data(records: List[Dict[str, Any]], config: GNNConfig, logger: logging.Logger) -> List[
    Dict[str, Any]]:
    """Filter low quality data based on graph structure"""
    filtered = []
    stats = {
        'too_few_nodes': 0,
        'too_many_nodes': 0,
        'no_graph': 0,
        'invalid_structure': 0,
        'valid': 0
    }

    for record in records:
        # Check for graph data existence
        if 'graph' not in record or not record['graph']:
            stats['no_graph'] += 1
            continue

        graph = record['graph']

        # Check for nodes
        if 'nodes' not in graph or not graph['nodes']:
            stats['invalid_structure'] += 1
            continue

        nodes = graph['nodes']
        num_nodes = len(nodes)

        # Check node count constraints
        if num_nodes < config.min_nodes:
            stats['too_few_nodes'] += 1
            continue

        if num_nodes > config.max_nodes:
            stats['too_many_nodes'] += 1
            continue

        # Additional validation: check if nodes have required fields
        valid_nodes = all(
            isinstance(node, dict) and 'id' in node
            for node in nodes
        )

        if not valid_nodes:
            stats['invalid_structure'] += 1
            continue

        stats['valid'] += 1
        filtered.append(record)

    logger.info(f"Data quality filtering results: {stats}")
    return filtered


class VulnerabilityDataset(Dataset):
    """Dataset for vulnerability detection with enhanced feature extraction"""

    def __init__(self, records: List[Dict[str, Any]], config: GNNConfig, logger: logging.Logger):
        self.records = records
        self.config = config
        self.logger = logger
        self._compute_class_weights()
        self._log_dataset_info()

    def _compute_class_weights(self):
        """Compute class weights for handling imbalanced datasets"""
        if not self.records:
            self.class_weights = np.array([1.0, 1.0])
            self.logger.warning("No data records found, using default class weights")
            return

        labels = [record['label'] for record in self.records]
        unique_labels = np.unique(labels)

        if len(unique_labels) == 1:
            self.class_weights = np.array([1.0, 1.0])
            self.logger.warning(f"Only one class found: {unique_labels[0]}, using default weights")
        else:
            label_counts = np.bincount(labels)
            total = len(labels)
            self.class_weights = total / (len(label_counts) * label_counts)
            self.logger.info(f"Computed class weights: {dict(enumerate(self.class_weights))}")

    def _log_dataset_info(self):
        """Log dataset statistics"""
        if not self.records:
            self.logger.warning("Empty dataset")
            return

        # Label distribution
        label_dist = defaultdict(int)
        node_counts = []
        edge_counts = []

        for record in self.records:
            label_dist[record['label']] += 1

            if 'graph' in record and 'nodes' in record['graph']:
                node_counts.append(len(record['graph']['nodes']))

            if 'graph' in record and 'edges' in record['graph']:
                edge_counts.append(len(record['graph']['edges']))

        self.logger.info(f"Dataset size: {len(self.records)}")
        self.logger.info(f"Label distribution: {dict(label_dist)}")

        if node_counts:
            self.logger.info(f"Node statistics: avg={np.mean(node_counts):.1f}, "
                             f"min={min(node_counts)}, max={max(node_counts)}")

        if edge_counts:
            self.logger.info(f"Edge statistics: avg={np.mean(edge_counts):.1f}, "
                             f"min={min(edge_counts)}, max={max(edge_counts)}")

    def get_sample_weights(self) -> torch.FloatTensor:
        """Get sample weights for WeightedRandomSampler"""
        weights = [self.class_weights[record['label']] for record in self.records]
        return torch.FloatTensor(weights)

    def _extract_node_features(self, node: Dict[str, Any]) -> List[float]:
        """Extract enhanced features from a graph node"""
        # Node type
        node_label = node.get('label', 'UNKNOWN')
        node_type = self.config.node_types.get(node_label, self.config.node_types['UNKNOWN'])

        # Code content analysis
        code = str(node.get('code', ''))
        code_len = len(code)

        # Vulnerability pattern detection
        has_vuln_pattern = int(any(pattern in code.lower() for pattern in VULN_PATTERNS))
        has_safe_pattern = int(any(pattern in code for pattern in SAFE_PATTERNS))

        # Special keyword detection
        has_special_keywords = int(any(keyword in code for keyword in ['ORIGIN', 'PLATFORM']))

        # Node type indicators
        is_literal = int(node_label == 'LITERAL')
        is_identifier = int(node_label == 'IDENTIFIER')

        # Position information
        line_num = node.get('lineNumber', 0)
        col_num = node.get('columnNumber', 0)

        # Code pattern analysis
        has_assignment = int('=' in code and '==' not in code)

        # Construct feature vector (10 features)
        features = [
            line_num / 100.0,  # Normalized line number
            col_num / 100.0,  # Normalized column number
            node_type / len(self.config.node_types),  # Normalized node type
            min(code_len / 50.0, 1.0),  # Normalized code length (capped at 1.0)
            has_vuln_pattern,  # Vulnerability pattern indicator
            has_safe_pattern,  # Safety pattern indicator
            has_special_keywords,  # Special keywords indicator
            is_literal,  # Literal node indicator
            is_identifier,  # Identifier node indicator
            has_assignment,  # Assignment indicator
        ]

        return features

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[GeoData, torch.Tensor]:
        record = self.records[idx]
        graph_data = record['graph']
        nodes = graph_data['nodes']
        edges = graph_data.get('edges', [])

        # Create node ID to index mapping
        id_to_idx = {node.get('id', i): i for i, node in enumerate(nodes)}

        # Extract node features
        node_features = []
        for node in nodes:
            features = self._extract_node_features(node)
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Process edges
        if edges:
            src_list = []
            dst_list = []
            edge_features = []

            for edge in edges:
                src_id = edge.get('src')
                dst_id = edge.get('dst')

                if src_id in id_to_idx and dst_id in id_to_idx:
                    src_list.append(id_to_idx[src_id])
                    dst_list.append(id_to_idx[dst_id])

                    # Edge type one-hot encoding
                    edge_type = self.config.edge_types.get(
                        edge.get('label', ''), 0
                    )
                    edge_feat = one_hot(
                        torch.tensor(edge_type),
                        num_classes=len(self.config.edge_types) + 1
                    )
                    edge_features.append(edge_feat)

            if src_list and dst_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                edge_attr = torch.stack(edge_features).float()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.config.edge_types) + 1), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config.edge_types) + 1), dtype=torch.float)

        label = torch.tensor(record['label'], dtype=torch.long)

        return GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr), label


class EnhancedGNN(nn.Module):
    """Enhanced GNN with attention mechanisms for vulnerability detection"""

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.feature_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # GNN layers
        self.layers = nn.ModuleList()
        self.attention_weights = []

        # Layer 1: GINE for edge feature processing
        self.gine_layer = GINEConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            ),
            edge_dim=len(config.edge_types) + 1
        )
        self.layers.append(self.gine_layer)

        # Layer 2: GAT for attention mechanism
        self.gat_layer = GATv2Conv(
            self.hidden_dim,
            self.hidden_dim,
            edge_dim=len(config.edge_types) + 1,
            heads=4,
            concat=False,
            dropout=0.1
        )
        self.layers.append(self.gat_layer)

        # Batch normalization for each layer
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim) for _ in range(config.gnn_layers)
        ])

        # Skip connection weights
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(config.gnn_layers)
        ])

        # Node importance scoring network
        self.node_importance = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        """Forward pass through the GNN"""
        # Clear previous attention weights
        self.attention_weights = []

        # Input projection
        x = self.input_proj(x)

        # Store layer outputs for skip connections
        layer_outputs = [x]

        # Layer 1: GINE
        if edge_attr is not None and edge_attr.size(0) > 0:
            x_new = self.gine_layer(x, edge_index, edge_attr)
        else:
            x_new = self.gine_layer(x, edge_index)

        x_new = F.relu(x_new)

        # Skip connection
        skip_weight = torch.sigmoid(self.skip_weights[0])
        x = skip_weight * x + (1 - skip_weight) * x_new
        x = self.norms[0](x)
        layer_outputs.append(x)

        # Layer 2: GAT with attention
        if edge_attr is not None and edge_attr.size(0) > 0:
            x_new, (edge_index_att, attention_weights) = self.gat_layer(
                x, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            x_new, (edge_index_att, attention_weights) = self.gat_layer(
                x, edge_index, return_attention_weights=True
            )

        # Store attention weights
        self.attention_weights = attention_weights

        x_new = F.relu(x_new)

        # Skip connection
        skip_weight = torch.sigmoid(self.skip_weights[1])
        x = skip_weight * x + (1 - skip_weight) * x_new
        x = self.norms[1](x)
        layer_outputs.append(x)

        # Fusion of all layer outputs
        final_output = sum(layer_outputs) / len(layer_outputs)

        # Compute node importance scores
        node_importance_scores = self.node_importance(final_output)

        if return_attention:
            return final_output, node_importance_scores, self.attention_weights

        return final_output

    def forward_from_gine(self, x, edge_index, edge_attr, batch):
        """Forward pass starting from GINE layer for core subgraph embedding"""
        # Input projection
        x = self.input_proj(x)

        # GINE layer
        if edge_attr is not None and edge_attr.size(0) > 0:
            x = self.gine_layer(x, edge_index, edge_attr)
        else:
            x = self.gine_layer(x, edge_index)

        x = F.relu(x)
        x = self.norms[0](x)

        # GAT layer
        if edge_attr is not None and edge_attr.size(0) > 0:
            x, _ = self.gat_layer(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x, _ = self.gat_layer(x, edge_index, return_attention_weights=True)

        x = F.relu(x)
        x = self.norms[1](x)

        return x


class GNNVulnerabilityDetector(nn.Module):
    """Complete GNN model for vulnerability detection"""

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.gnn = EnhancedGNN(config)
        self.config = config

        # Graph-level attention mechanism
        self.graph_attention = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.LeakyReLU(0.2)
        )

        # Classifier with multiple pooling strategies
        self.classifier = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, graph: GeoData, device: torch.device, return_attention=False):
        """Forward pass through the complete model"""
        graph = graph.to(device)

        if return_attention:
            node_emb, node_importance, edge_attention = self.gnn(
                graph.x, graph.edge_index, graph.edge_attr, graph.batch, return_attention=True
            )
        else:
            node_emb = self.gnn(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        # Multiple pooling strategies
        graph_emb_mean = global_mean_pool(node_emb, graph.batch)
        graph_emb_max = global_max_pool(node_emb, graph.batch)

        # Attention-weighted pooling
        att_scores = self.graph_attention(node_emb)
        batch_size = graph.batch.max().item() + 1
        graph_emb_att_list = []

        for i in range(batch_size):
            mask = (graph.batch == i)
            if mask.any():
                nodes_i = node_emb[mask]
                scores_i = att_scores[mask]
                scores_i_softmax = F.softmax(scores_i, dim=0)
                att_emb_i = (nodes_i * scores_i_softmax).sum(dim=0)
                graph_emb_att_list.append(att_emb_i)

        if graph_emb_att_list:
            graph_emb_att = torch.stack(graph_emb_att_list)
        else:
            graph_emb_att = torch.zeros_like(graph_emb_mean)

        # Concatenate all pooling results
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max, graph_emb_att], dim=1)

        # Classification
        logits = self.classifier(graph_emb)

        if return_attention:
            return logits, node_importance, edge_attention

        return logits

    def get_core_subgraph_embedding(self, graph: GeoData, device: torch.device, top_k_ratio: float = 0.5):
        """Extract core subgraph embedding based on node importance"""
        graph = graph.to(device)

        # Get node embeddings and importance scores
        node_emb, node_importance, edge_attention = self.gnn(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch, return_attention=True
        )

        # Select top-k important nodes
        num_nodes = graph.x.size(0)
        k = max(1, int(num_nodes * top_k_ratio))

        _, top_k_indices = torch.topk(node_importance.squeeze(), min(k, num_nodes))

        # Create node mask
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[top_k_indices] = True

        # Filter edges to keep only those connecting core nodes
        if graph.edge_index.size(1) > 0:
            edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
            core_edge_index = graph.edge_index[:, edge_mask]
            core_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
        else:
            core_edge_index = graph.edge_index
            core_edge_attr = graph.edge_attr

        # Reindex nodes
        node_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device) - 1
        new_indices = torch.arange(len(top_k_indices), device=device)
        node_mapping[top_k_indices] = new_indices

        # Update edge indices
        if core_edge_index.size(1) > 0:
            valid_edges = (node_mapping[core_edge_index[0]] >= 0) & (node_mapping[core_edge_index[1]] >= 0)
            core_edge_index = core_edge_index[:, valid_edges]
            core_edge_index = node_mapping[core_edge_index]
            if core_edge_attr is not None:
                core_edge_attr = core_edge_attr[valid_edges]

        # Core subgraph features
        core_x = graph.x[top_k_indices]

        # Generate embedding through GNN
        if core_edge_index.size(1) > 0:
            core_node_emb = self.gnn.forward_from_gine(
                core_x, core_edge_index, core_edge_attr, None
            )
        else:
            core_node_emb = self.gnn.input_proj(core_x)

        # Graph-level embedding using mean pooling
        core_graph_emb = core_node_emb.mean(dim=0)

        return core_graph_emb, top_k_indices


def load_datasets(config: GNNConfig, logger: logging.Logger) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and prepare datasets for training"""
    logger.info("Loading main dataset...")

    # Load main data file
    with open(config.json_path, 'r', encoding='utf-8') as f:
        main_records = json.load(f)

    # Create ID to record mapping
    id_to_record = {str(record['id']): record for record in main_records}
    logger.info(f"Main dataset contains {len(id_to_record)} records")

    # Find subgraph files
    subgraph_files = list(Path(config.subgraph_dir).glob("*.json"))
    logger.info(f"Found {len(subgraph_files)} subgraph files")

    valid_records = []
    stats = {
        'missing_main': 0,
        'empty_graphs': 0,
        'json_errors': 0,
        'valid': 0
    }

    for subgraph_file in subgraph_files:
        try:
            # Extract ID from filename
            record_id = subgraph_file.stem

            # Find corresponding main record
            if record_id not in id_to_record:
                stats['missing_main'] += 1
                continue

            main_record = id_to_record[record_id]

            # Load subgraph data
            with open(subgraph_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    stats['empty_graphs'] += 1
                    continue
                graph_data = json.loads(content)

            # Validate graph structure
            if not graph_data or 'nodes' not in graph_data or not graph_data['nodes']:
                stats['empty_graphs'] += 1
                continue

            # Merge data
            record = {
                'id': main_record['id'],
                'label': main_record['label'],
                'graph': graph_data,
                'cve_id': main_record.get('cve_id', ''),
                'key_node': main_record.get('key_node', ''),
                'vulnerability_line': main_record.get('vulnerability_line', [])
            }

            valid_records.append(record)
            stats['valid'] += 1

        except json.JSONDecodeError as e:
            stats['json_errors'] += 1
            logger.warning(f"JSON decode error in {subgraph_file}: {e}")
        except Exception as e:
            stats['json_errors'] += 1
            logger.warning(f"Error processing {subgraph_file}: {e}")

    logger.info(f"Dataset loading stats: {stats}")

    if not valid_records:
        raise ValueError("No valid records found! Check data paths and file formats.")

    # Apply data quality filtering
    valid_records = filter_low_quality_data(valid_records, config, logger)

    if not valid_records:
        raise ValueError("All data filtered out! Check quality control parameters.")

    # Log label distribution
    label_counts = defaultdict(int)
    for record in valid_records:
        label_counts[record['label']] += 1
    logger.info(f"Label distribution: {dict(label_counts)}")

    # Split dataset
    total = len(valid_records)
    train_n = max(1, int(config.split_ratio[0] * total))
    val_n = max(1, int(config.split_ratio[1] * total))
    test_n = max(1, total - train_n - val_n)

    logger.info(f"Dataset splits: train={train_n}, val={val_n}, test={test_n}")

    dataset = VulnerabilityDataset(valid_records, config, logger)
    return random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(config.seed)
    )


def collate_fn(batch: List[Tuple]) -> Tuple[GeoData, torch.Tensor]:
    """Custom collate function for batch processing"""
    graphs, labels = zip(*batch)
    graph_batch = Batch.from_data_list(graphs)
    label_batch = torch.stack(labels)
    return graph_batch, label_batch


def evaluate_model(model: nn.Module,
                   data_loader: GeoDataLoader,
                   device: torch.device,
                   criterion: nn.Module) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for graph, labels in data_loader:
            labels = labels.to(device)
            logits = model(graph, device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)
    avg_loss = total_loss / max(num_batches, 1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'loss': avg_loss
    }


def train_model(config: GNNConfig, logger: logging.Logger) -> None:
    """Main training loop"""
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize WandB if configured
    if config.use_wandb and config.wandb_api_key:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.__dict__
        )

    logger.info("=" * 60)
    logger.info("GNN Vulnerability Detection Pretraining")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model config: hidden_dim={config.gnn_hidden_dim}, layers={config.gnn_layers}")
    logger.info("=" * 60)

    # Load datasets
    logger.info("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(config, logger)

    # Create data loaders
    if config.use_class_weights and hasattr(train_ds.dataset, 'get_sample_weights'):
        sample_weights = train_ds.dataset.get_sample_weights()
        train_indices = train_ds.indices
        train_weights = sample_weights[train_indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = GeoDataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = GeoDataLoader(
        val_ds, batch_size=config.batch_size, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )
    test_loader = GeoDataLoader(
        test_ds, batch_size=config.batch_size, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Initialize model
    logger.info("Initializing model...")
    model = GNNVulnerabilityDetector(config).to(config.device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': model.gnn.parameters(), 'lr': config.lr},
        {'params': model.classifier.parameters(), 'lr': config.lr * 2}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )

    # Setup loss function with class weights
    if hasattr(train_ds.dataset, 'class_weights'):
        class_weights = torch.tensor(train_ds.dataset.class_weights, dtype=torch.float).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        pos_weight = torch.tensor([1.0, config.pos_weight]).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=pos_weight)

    # Training setup
    scaler = GradScaler()
    best_f1 = 0.0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(1, config.epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        train_predictions = []
        train_labels = []

        for batch_idx, (graph, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            labels = labels.to(config.device)

            with autocast(device_type=config.device.type):
                logits = model(graph, config.device)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            train_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss={loss.item():.4f}")

                if config.use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "step": (epoch - 1) * len(train_loader) + batch_idx
                    })

        # Update learning rate
        scheduler.step()

        # Calculate training metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_acc = sum(p == l for p, l in zip(train_predictions, train_labels)) / len(train_labels)
        train_f1 = f1_score(train_labels, train_predictions, zero_division=0)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, config.device, criterion)

        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"  Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                    f"Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, "
                    f"F1={val_metrics['f1_score']:.4f}")
        logger.info(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # Log to WandB
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
                "val_f1": val_metrics['f1_score'],
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Early stopping and model saving
        combined_score = 0.7 * val_metrics['f1_score'] + 0.3 * val_metrics['recall']

        if combined_score > best_f1 + config.min_delta:
            best_f1 = combined_score
            patience_counter = 0

            # Save best model
            model_path = Path(config.output_dir) / "best_gnn_model.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"🎉 New best model saved: F1={val_metrics['f1_score']:.4f}, "
                        f"Recall={val_metrics['recall']:.4f}")

            if config.use_wandb:
                wandb.run.summary["best_val_f1"] = val_metrics['f1_score']
                wandb.run.summary["best_val_recall"] = val_metrics['recall']
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping: {config.patience} epochs without improvement")
                break

    # Final testing
    logger.info("Loading best model for final evaluation...")
    best_model_path = Path(config.output_dir) / "best_gnn_model.pt"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    test_metrics = evaluate_model(model, test_loader, config.device, criterion)

    logger.info("=" * 60)
    logger.info("🎯 Final Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    logger.info("=" * 60)

    # Save complete model
    complete_model_path = Path(config.output_dir) / "gnn_model_complete.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'node_types': config.node_types,
        'edge_types': config.edge_types,
    }, complete_model_path)

    logger.info(f"Complete model saved to: {complete_model_path}")

    # Log final results to WandB
    if config.use_wandb:
        wandb.run.summary.update({
            "test_accuracy": test_metrics['accuracy'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_f1": test_metrics['f1_score']
        })

        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Vulnerability', 'Vulnerability'],
                    yticklabels=['No Vulnerability', 'Vulnerability'])
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({"test_confusion_matrix": wandb.Image(plt)})
        plt.close()

        wandb.finish()

    # Demonstrate core subgraph extraction
    logger.info("\nDemonstrating core subgraph extraction...")
    model.eval()
    with torch.no_grad():
        sample_graph, sample_label = test_loader.dataset[0]
        sample_batch = Batch.from_data_list([sample_graph])

        core_embedding, core_node_indices = model.get_core_subgraph_embedding(
            sample_batch, config.device, config.top_k_ratio
        )

        logger.info(f"Original graph nodes: {sample_graph.x.size(0)}")
        logger.info(f"Core subgraph nodes: {len(core_node_indices)}")
        logger.info(f"Core subgraph embedding dimension: {core_embedding.shape}")


def main():
    """Main function"""
    # Environment setup
    os.environ.update({
        'TOKENIZERS_PARALLELISM': 'false'
    })
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Configuration
    config = GNNConfig()
    logger = setup_logging(config)

    try:
        train_model(config, logger)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()