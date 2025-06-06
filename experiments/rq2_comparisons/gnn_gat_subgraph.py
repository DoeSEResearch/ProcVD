#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import warnings
import math
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch.nn.functional import one_hot
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ----------------- Logging Setup -----------------
def setup_logging(log_file: str = "gat_training.log") -> logging.Logger:
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


# ----------------- VDS Calculation Function -----------------
def calculate_vds(y_true: np.ndarray, y_probs: np.ndarray, fpr_tolerance: float = 0.005) -> Dict:
    """
    Calculate Vulnerability Detection Score (VDS)

    Args:
        y_true: True labels (0: safe, 1: vulnerable)
        y_probs: Predicted vulnerability probabilities
        fpr_tolerance: Acceptable false positive rate threshold (default 0.5%)

    Returns:
        Dict containing VDS metrics
    """
    # Get all unique threshold points
    thresholds = np.unique(y_probs)
    thresholds = np.sort(thresholds)[::-1]  # Sort high to low

    best_fnr = 1.0  # Best (lowest) false negative rate
    best_threshold = None
    valid_points = []

    for threshold in thresholds:
        # Generate predictions based on threshold
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate confusion matrix elements
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        valid_points.append({
            'threshold': threshold,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })

        # Update best FNR if FPR is within acceptable range
        if fpr <= fpr_tolerance:
            if fnr < best_fnr:
                best_fnr = fnr
                best_threshold = threshold

    # If no threshold meets FPR requirement, use strictest threshold
    if best_threshold is None:
        # Select point with minimum FPR
        min_fpr_point = min(valid_points, key=lambda x: x['fpr'])
        best_fnr = min_fpr_point['fnr']
        best_threshold = min_fpr_point['threshold']
        logger.warning(f"No threshold found with FPR <= {fpr_tolerance:.3f}, using min FPR threshold: {best_threshold:.4f}")

    # Calculate detailed metrics at best threshold
    final_predictions = (y_probs >= best_threshold).astype(int)
    tn = np.sum((y_true == 0) & (final_predictions == 0))
    fp = np.sum((y_true == 0) & (final_predictions == 1))
    fn = np.sum((y_true == 1) & (final_predictions == 0))
    tp = np.sum((y_true == 1) & (final_predictions == 1))

    final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    final_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'vds': best_fnr,  # VDS is the best FNR
        'vds_threshold': best_threshold,
        'vds_fpr': final_fpr,
        'vds_fnr': final_fnr,
        'fpr_tolerance': fpr_tolerance,
        'vds_confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'threshold_analysis': valid_points,
        'satisfies_fpr_constraint': final_fpr <= fpr_tolerance
    }


# ----------------- Configuration Class -----------------
class GATTrainingConfig:
    def __init__(self, experiment_type: str = "unbalanced"):
        """
        experiment_type: "unbalanced", "balanced", or "both"
        """
        self.experiment_type = experiment_type

        # Core features data path
        self.core_features_dir = "../../result/primevul_core_features"

        # PrimeVul labels data path
        self.primevul_labels_dir = "../../data/primevul_process"

        # Dataset configuration
        self.datasets = {
            'train': 'train',
            'test': 'test',
            'valid': 'valid',
            'train_paired': 'train_paired',
            'test_paired': 'test_paired',
            'valid_paired': 'valid_paired'
        }

        # Label file mapping
        self.label_files = {
            'train': 'PrimeVul_unbalanced_train_sampled.jsonl',
            'test': 'PrimeVul_unbalanced_test_sampled.jsonl',
            'valid': 'PrimeVul_unbalanced_valid_sampled.jsonl',
            'train_paired': 'PrimeVul_balanced_train.jsonl',
            'test_paired': 'PrimeVul_balanced_test.jsonl',
            'valid_paired': 'PrimeVul_balanced_valid.jsonl'
        }

        # Model configuration - GAT specific parameters
        self.input_dim = 10  # Input feature dimension (consistent with original code)
        self.hidden_dim = 128
        self.num_layers = 3
        self.dropout = 0.2
        self.num_classes = 2

        # GAT specific parameters
        self.num_heads = 8  # Number of attention heads
        self.concat_heads = True  # Whether to concatenate multi-head outputs
        self.attention_dropout = 0.1  # Attention dropout

        # Training configuration
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.patience = 15
        self.min_delta = 1e-4

        # VDS configuration
        self.vds_fpr_tolerance = 0.005  # 0.5% FPR tolerance

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup experiment datasets based on type
        self._setup_experiment_datasets()

        # Output paths
        self.output_dir = f"./gat_training_results_{experiment_type}"
        self.checkpoint_dir = f"./gat_checkpoints_{experiment_type}"
        self.log_dir = f"./gat_logs_{experiment_type}"

        # Experiment settings
        self.experiment_name = f"pure_gat_vulnerability_detection_{experiment_type}"
        self.save_best_model = True
        self.save_checkpoint_every = 10

        # Pairwise evaluation settings (only for balanced dataset)
        self.enable_pairwise_evaluation = (experiment_type in ["balanced", "both"])

    def _setup_experiment_datasets(self):
        """Setup datasets based on experiment type"""
        if self.experiment_type == "unbalanced":
            self.train_dataset = 'train'
            self.valid_dataset = 'valid'
            self.test_dataset = 'test'
            self.is_balanced = False
        elif self.experiment_type == "balanced":
            self.train_dataset = 'train_paired'
            self.valid_dataset = 'valid_paired'
            self.test_dataset = 'test_paired'
            self.is_balanced = True
        else:
            # "both" case defaults to unbalanced dataset, but will run two experiments
            self.train_dataset = 'train'
            self.valid_dataset = 'valid'
            self.test_dataset = 'test'
            self.is_balanced = False


# ----------------- Pure GAT Model -----------------
class PureGAT(nn.Module):
    def __init__(self, config: GATTrainingConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(config.num_layers):
            if i == 0:
                # First layer: input dimension -> hidden dimension
                in_dim = config.hidden_dim
                out_dim = config.hidden_dim // config.num_heads if config.concat_heads else config.hidden_dim
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=config.num_heads,
                        concat=config.concat_heads,
                        dropout=config.attention_dropout
                    )
                )
            elif i == config.num_layers - 1:
                # Last layer: don't use concat, ensure output dimension is hidden_dim
                in_dim = config.hidden_dim
                out_dim = config.hidden_dim
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=1,  # Use single head for last layer
                        concat=False,
                        dropout=config.attention_dropout
                    )
                )
            else:
                # Middle layers
                in_dim = config.hidden_dim
                out_dim = config.hidden_dim // config.num_heads if config.concat_heads else config.hidden_dim
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=config.num_heads,
                        concat=config.concat_heads,
                        dropout=config.attention_dropout
                    )
                )

            self.batch_norms.append(nn.BatchNorm1d(config.hidden_dim))

        # Graph-level pooling
        self.global_pool = global_mean_pool

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.num_classes)
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, batch):
        # Input projection
        x = self.input_proj(x)

        # GAT layers forward pass
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new = gat_layer(x, edge_index)
            x_new = batch_norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.config.dropout, training=self.training)

            # Residual connection (except first layer)
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new

        # Graph-level pooling
        graph_emb = self.global_pool(x, batch)

        # Classification
        logits = self.classifier(graph_emb)

        return logits


# ----------------- Core Subgraph Dataset -----------------
class CoreSubgraphDataset(Dataset):
    def __init__(self, core_features_file: Path, labels_file: Path, config: GATTrainingConfig):
        self.config = config
        self.data = []
        self.labels = []
        self.file_indices = []  # Store file indices for pairwise evaluation

        # Load core features data
        logger.info(f"Loading core features: {core_features_file}")
        if not core_features_file.exists():
            raise FileNotFoundError(f"Core features file not found: {core_features_file}")

        with open(core_features_file, 'rb') as f:
            core_data = pickle.load(f)

        # Load labels data
        logger.info(f"Loading labels data: {labels_file}")
        labels_dict = self._load_labels(labels_file)

        # Process data
        logger.info("Processing core subgraph data...")
        skipped_count = 0
        valid_count = 0

        for item in tqdm(core_data['analysis_details'], desc="Processing core subgraphs"):
            file_idx = item.get('file_idx')

            # Get label
            if file_idx and file_idx in labels_dict:
                label = labels_dict[file_idx]

                # Reconstruct subgraph
                core_graph = self._reconstruct_core_graph(item)
                if core_graph is not None:
                    # Validate graph data
                    if core_graph.x.size(0) > 0:
                        self.data.append(core_graph)
                        self.labels.append(label)
                        self.file_indices.append(file_idx)
                        valid_count += 1
                    else:
                        logger.debug(f"Skipping empty graph: {file_idx}")
                        skipped_count += 1
                else:
                    logger.debug(f"Reconstruction failed: {file_idx}")
                    skipped_count += 1
            else:
                if not file_idx:
                    logger.debug("Skipping: missing file_idx")
                else:
                    logger.debug(f"Skipping: label not found {file_idx}")
                skipped_count += 1

        logger.info(f"Successfully loaded {valid_count} core subgraphs, skipped {skipped_count}")

        # Statistics
        if len(self.labels) > 0:
            unique_labels, label_counts = np.unique(self.labels, return_counts=True)
            label_dist = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
            logger.info(f"Label distribution: {label_dist}")

            # For balanced dataset, check pairwise structure
            if config.is_balanced and len(self.data) % 2 == 0:
                logger.info(f"Balanced dataset: {len(self.data)} samples, {len(self.data) // 2} pairs")
                self._verify_balanced_pairs()
            elif config.is_balanced:
                logger.warning(f"Balanced dataset has odd number of samples: {len(self.data)}")
        else:
            logger.warning("No valid label data")

    def _verify_balanced_pairs(self):
        """Verify pairwise structure of balanced dataset"""
        correct_pairs = 0
        total_pairs = len(self.data) // 2

        for i in range(0, len(self.data), 2):
            if i + 1 < len(self.data):
                label1, label2 = self.labels[i], self.labels[i + 1]
                if label1 != label2:  # Paired data should have different labels
                    correct_pairs += 1

        logger.info(f"Verified pairwise structure: {correct_pairs}/{total_pairs} pairs have different labels")

    def _load_labels(self, labels_file: Path) -> Dict[str, int]:
        """Load labels file"""
        labels = {}

        if not labels_file.exists():
            logger.error(f"Labels file not found: {labels_file}")
            return labels

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    idx = str(data.get('idx', ''))
                    target = data.get('target', 0)

                    if idx:
                        labels[idx] = target

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(labels)} labels")
        return labels

    def _reconstruct_core_graph(self, item: Dict) -> Optional[GeoData]:
        """Reconstruct core subgraph from analysis results"""
        try:
            analysis = item.get('analysis', {})
            core_node_indices = analysis.get('core_node_indices', [])

            if not core_node_indices:
                return None

            # Reconstruct from original graph data
            # Simplified version: if core embedding exists, use average features to create virtual nodes
            if 'core_embedding' in item:
                embedding = np.array(item['core_embedding'])
                core_nodes = analysis.get('core_nodes', len(core_node_indices))

                # Ensure at least one node
                if core_nodes <= 0:
                    core_nodes = 1

                # Create core node features (simplified version)
                emb_dim = len(embedding)
                feature_dim = self.config.input_dim

                # Project embedding vector to input feature dimension
                if emb_dim >= feature_dim:
                    # Take first feature_dim dimensions
                    x = embedding[:feature_dim].reshape(1, -1)
                    x = np.tile(x, (core_nodes, 1))
                else:
                    # If embedding dimension is insufficient, pad with zeros
                    x = np.zeros((core_nodes, feature_dim))
                    x[:, :emb_dim] = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding[:1, :emb_dim]

                # Add some randomness to distinguish different nodes
                if core_nodes > 1:
                    noise = np.random.normal(0, 0.01, x.shape)
                    x = x + noise

                x = torch.tensor(x, dtype=torch.float)

                # Create simple edge connections (fully connected subgraph, but avoid being too dense)
                edge_list = []
                if core_nodes > 1:
                    # Create simple chain connections, then add some random edges
                    for i in range(core_nodes - 1):
                        edge_list.extend([[i, i + 1], [i + 1, i]])  # Chain connections

                    # Add some random edges (but not too many)
                    if core_nodes > 2:
                        for i in range(min(core_nodes // 2, 3)):  # Add at most 3 additional edges
                            src = np.random.randint(0, core_nodes)
                            dst = np.random.randint(0, core_nodes)
                            if src != dst:
                                edge_list.extend([[src, dst], [dst, src]])

                # Remove duplicates
                if edge_list:
                    edge_set = set()
                    unique_edges = []
                    for edge in edge_list:
                        edge_tuple = tuple(edge)
                        if edge_tuple not in edge_set:
                            edge_set.add(edge_tuple)
                            unique_edges.append(edge)

                    if unique_edges:
                        edge_index = torch.tensor(unique_edges, dtype=torch.long).t()
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                return GeoData(x=x, edge_index=edge_index)

            return None

        except Exception as e:
            logger.debug(f"Failed to reconstruct core subgraph: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.data[idx]
        label = self.labels[idx]

        # Create copy of graph to avoid modifying original data
        new_graph = GeoData(x=graph.x.clone(), edge_index=graph.edge_index.clone())

        # Ensure label is scalar and convert to correct format
        if isinstance(label, (np.integer, np.floating)):
            label = int(label)

        # Add label to graph data (as scalar)
        new_graph.y = torch.tensor(label, dtype=torch.long)

        return new_graph


# ----------------- Pairwise Evaluator -----------------
class PairwiseEvaluator:
    def __init__(self):
        self.pair_stats = {
            'P-C': 0,  # Pair-wise Correct Prediction
            'P-V': 0,  # Pair-wise Vulnerable Prediction
            'P-B': 0,  # Pair-wise Benign Prediction
            'P-R': 0  # Pair-wise Reversed Prediction
        }
        self.total_pairs = 0

    def evaluate_pairs(self, predictions: List[int], true_labels: List[int]) -> Dict:
        """Evaluate pairwise prediction results"""
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true labels length mismatch")

        if len(predictions) % 2 != 0:
            logger.warning("Data length is not even, last sample will be ignored")

        self.total_pairs = len(predictions) // 2

        for i in range(0, len(predictions), 2):
            if i + 1 >= len(predictions):
                break

            pred1, pred2 = predictions[i], predictions[i + 1]
            true1, true2 = true_labels[i], true_labels[i + 1]

            # Classify pairwise prediction types
            if pred1 == true1 and pred2 == true2:
                # Correct prediction
                self.pair_stats['P-C'] += 1
            elif pred1 == 1 and pred2 == 1:
                # Both predicted as vulnerable
                self.pair_stats['P-V'] += 1
            elif pred1 == 0 and pred2 == 0:
                # Both predicted as benign
                self.pair_stats['P-B'] += 1
            elif (pred1 == true2 and pred2 == true1):
                # Reversed prediction
                self.pair_stats['P-R'] += 1
            # Note: other cases (partially correct) are not in these four categories

        # Calculate percentages
        pair_percentages = {}
        if self.total_pairs > 0:
            for key, count in self.pair_stats.items():
                pair_percentages[f"{key}_percent"] = (count / self.total_pairs) * 100

        return {
            'pair_counts': self.pair_stats.copy(),
            'pair_percentages': pair_percentages,
            'total_pairs': self.total_pairs,
            'coverage': sum(self.pair_stats.values()) / max(1, self.total_pairs) * 100
        }

    def get_summary(self) -> str:
        """Get pairwise evaluation summary"""
        if self.total_pairs == 0:
            return "No pairwise data for evaluation"

        summary = f"Pairwise evaluation results (total {self.total_pairs} pairs):\n"
        for key, count in self.pair_stats.items():
            percentage = (count / self.total_pairs) * 100
            summary += f"  {key}: {count:4d} ({percentage:5.1f}%)\n"

        coverage = sum(self.pair_stats.values()) / self.total_pairs * 100
        summary += f"  Coverage: {coverage:.1f}%"

        return summary


# ----------------- Trainer -----------------
class GATTrainer:
    def __init__(self, config: GATTrainingConfig):
        self.config = config
        self._setup_directories()

        # Initialize model
        self.model = PureGAT(config).to(config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Training history
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

        # Pairwise evaluator (only for balanced dataset)
        if config.enable_pairwise_evaluation:
            self.pairwise_evaluator = PairwiseEvaluator()

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config.output_dir, self.config.checkpoint_dir, self.config.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def load_datasets(self):
        """Load training, validation, and test datasets"""

        # Build file paths
        def get_file_paths(dataset_name):
            core_features_file = Path(self.config.core_features_dir) / f"{dataset_name}_complete_results.pkl"
            labels_file = Path(self.config.primevul_labels_dir) / self.config.label_files[dataset_name]
            return core_features_file, labels_file

        logger.info("Starting to load datasets...")

        # Load training set
        logger.info(f"Loading training set: {self.config.train_dataset}")
        train_core_file, train_labels_file = get_file_paths(self.config.train_dataset)
        self.train_dataset = CoreSubgraphDataset(train_core_file, train_labels_file, self.config)

        # Load validation set
        logger.info(f"Loading validation set: {self.config.valid_dataset}")
        val_core_file, val_labels_file = get_file_paths(self.config.valid_dataset)
        self.val_dataset = CoreSubgraphDataset(val_core_file, val_labels_file, self.config)

        # Load test set
        logger.info(f"Loading test set: {self.config.test_dataset}")
        test_core_file, test_labels_file = get_file_paths(self.config.test_dataset)
        self.test_dataset = CoreSubgraphDataset(test_core_file, test_labels_file, self.config)

        # Validate datasets are not empty
        if len(self.train_dataset) == 0:
            raise ValueError("Training set is empty")
        if len(self.val_dataset) == 0:
            raise ValueError("Validation set is empty")
        if len(self.test_dataset) == 0:
            raise ValueError("Test set is empty")

        # Create data loaders
        logger.info("Creating data loaders...")
        self.train_loader = GeoDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            follow_batch=None,
            exclude_keys=None
        )

        self.val_loader = GeoDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            follow_batch=None,
            exclude_keys=None
        )

        self.test_loader = GeoDataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            follow_batch=None,
            exclude_keys=None
        )

        logger.info(f"Dataset sizes:")
        logger.info(f"  Training set: {len(self.train_dataset)}")
        logger.info(f"  Validation set: {len(self.val_dataset)}")
        logger.info(f"  Test set: {len(self.test_dataset)}")

        # Test first batch
        logger.info("Validating data loaders...")
        try:
            for batch in self.train_loader:
                logger.info(f"First training batch:")
                logger.info(f"  Batch size: {batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 0}")
                logger.info(f"  Total nodes: {batch.x.size(0)}")
                logger.info(f"  Feature dimension: {batch.x.size(1)}")
                logger.info(f"  Number of edges: {batch.edge_index.size(1)}")
                logger.info(f"  Label shape: {batch.y.shape}")
                logger.info(f"  Label samples: {batch.y[:5] if batch.y.numel() >= 5 else batch.y}")
                break
        except Exception as e:
            logger.error(f"Data loader validation failed: {e}")
            raise

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            batch = batch.to(self.config.device)

            # Ensure batch.y has correct dimensions
            if batch.y.dim() == 0:
                batch.y = batch.y.unsqueeze(0)
            elif batch.y.dim() > 1:
                batch.y = batch.y.view(-1)

            logits = self.model(batch.x, batch.edge_index, batch.batch)

            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = batch.to(self.config.device)

                # Ensure batch.y has correct dimensions
                if batch.y.dim() == 0:
                    batch.y = batch.y.unsqueeze(0)
                elif batch.y.dim() > 1:
                    batch.y = batch.y.view(-1)

                logits = self.model(batch.x, batch.edge_index, batch.batch)

                loss = self.criterion(logits, batch.y)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                total_correct += (pred == batch.y).sum().item()
                total_samples += batch.y.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def test(self) -> Dict:
        """Test model and return detailed metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.config.device)

                # Ensure batch.y has correct dimensions
                if batch.y.dim() == 0:
                    batch.y = batch.y.unsqueeze(0)
                elif batch.y.dim() > 1:
                    batch.y = batch.y.view(-1)

                logits = self.model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=1)

                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Vulnerability class probability

        # Calculate standard metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Calculate VDS metrics
        logger.info("Calculating VDS metrics...")
        vds_results = calculate_vds(
            y_true=np.array(all_labels),
            y_probs=np.array(all_probs),
            fpr_tolerance=self.config.vds_fpr_tolerance
        )

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'vds_results': vds_results  # Add VDS results
        }

        # Add pairwise evaluation for balanced dataset
        if self.config.enable_pairwise_evaluation:
            pairwise_results = self.pairwise_evaluator.evaluate_pairs(all_preds, all_labels)
            results['pairwise_evaluation'] = pairwise_results

            logger.info("=" * 60)
            logger.info("Pairwise evaluation results:")
            logger.info(self.pairwise_evaluator.get_summary())
            logger.info("=" * 60)

        # Print VDS results
        logger.info("=" * 60)
        logger.info("VDS evaluation results:")
        logger.info(f"  VDS (FNR@FPR≤{self.config.vds_fpr_tolerance:.1%}): {vds_results['vds']:.4f}")
        logger.info(f"  Best threshold: {vds_results['vds_threshold']:.4f}")
        logger.info(f"  FPR at best threshold: {vds_results['vds_fpr']:.4f}")
        logger.info(f"  FNR at best threshold: {vds_results['vds_fnr']:.4f}")
        logger.info(f"  Satisfies FPR constraint: {'Yes' if vds_results['satisfies_fpr_constraint'] else 'No'}")
        vds_cm = vds_results['vds_confusion_matrix']
        logger.info(f"  Confusion matrix: TP={vds_cm['tp']}, FP={vds_cm['fp']}, TN={vds_cm['tn']}, FN={vds_cm['fn']}")
        logger.info("=" * 60)

        return results

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss
        }

        # Save current checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def train(self):
        """Complete training process"""
        logger.info("Starting training...")
        logger.info(f"Training configuration:")
        logger.info(f"  - Experiment type: {self.config.experiment_type}")
        logger.info(f"  - Model: Pure GAT")
        logger.info(f"  - Hidden dimension: {self.config.hidden_dim}")
        logger.info(f"  - Number of layers: {self.config.num_layers}")
        logger.info(f"  - Attention heads: {self.config.num_heads}")
        logger.info(f"  - Learning rate: {self.config.learning_rate}")
        logger.info(f"  - Batch size: {self.config.batch_size}")
        logger.info(f"  - Max epochs: {self.config.num_epochs}")
        logger.info(f"  - VDS FPR tolerance: {self.config.vds_fpr_tolerance:.1%}")
        logger.info(f"  - Pairwise evaluation: {'Enabled' if self.config.enable_pairwise_evaluation else 'Disabled'}")

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Record history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                        f"Time: {epoch_time:.2f}s")

            # Early stopping check
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if self.config.save_best_model:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Regular checkpoint saving
            if epoch % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed, total time: {total_time:.2f}s")

        # Test best model
        logger.info("Loading best model for testing...")
        best_model_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded best model")

        # Final testing
        test_results = self.test()

        logger.info("=" * 60)
        logger.info("Final test results:")
        logger.info(f"  Accuracy:  {test_results['accuracy']:.4f}")
        logger.info(f"  Precision: {test_results['precision']:.4f}")
        logger.info(f"  Recall:    {test_results['recall']:.4f}")
        logger.info(f"  F1-Score:  {test_results['f1_score']:.4f}")
        logger.info(f"  AUC:       {test_results['auc']:.4f}")
        logger.info(f"  VDS:       {test_results['vds_results']['vds']:.4f}")
        logger.info("=" * 60)

        # Save results
        self._save_results(test_results)

        return test_results

    def _save_results(self, test_results: Dict):
        """Save training results"""
        # Save test results
        results_file = Path(self.config.output_dir) / "test_results.json"

        # Create JSON-safe config copy
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)  # Convert device to string
            elif isinstance(value, Path):
                config_dict[key] = str(value)  # Convert Path to string
            elif isinstance(value, type):
                config_dict[key] = str(value)  # Convert type to string
            else:
                try:
                    # Test JSON serialization
                    json.dumps(value)
                    config_dict[key] = value
                except (TypeError, ValueError):
                    # If cannot serialize, convert to string
                    config_dict[key] = str(value)

        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'experiment_type': self.config.experiment_type,
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1_score': float(test_results['f1_score']),
            'auc': float(test_results['auc']),
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'config': config_dict
        }

        # Add VDS results
        vds_results = test_results['vds_results']
        json_results['vds'] = {
            'vds_score': float(vds_results['vds']),
            'vds_threshold': float(vds_results['vds_threshold']),
            'vds_fpr': float(vds_results['vds_fpr']),
            'vds_fnr': float(vds_results['vds_fnr']),
            'fpr_tolerance': float(vds_results['fpr_tolerance']),
            'vds_confusion_matrix': vds_results['vds_confusion_matrix'],
            'satisfies_fpr_constraint': bool(vds_results['satisfies_fpr_constraint'])
        }

        # Add pairwise evaluation results
        if 'pairwise_evaluation' in test_results:
            json_results['pairwise_evaluation'] = test_results['pairwise_evaluation']

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save training history
        history_file = Path(self.config.output_dir) / "training_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(self.train_history, f)

        # Save detailed prediction results (including VDS threshold analysis)
        predictions_file = Path(self.config.output_dir) / "predictions.pkl"
        with open(predictions_file, 'wb') as f:
            pickle.dump({
                'predictions': test_results['predictions'],
                'true_labels': test_results['true_labels'],
                'probabilities': test_results['probabilities'],
                'vds_threshold_analysis': vds_results['threshold_analysis']
            }, f)

        # Generate visualizations
        self._plot_training_curves()
        self._plot_confusion_matrix(test_results['confusion_matrix'])
        self._plot_vds_analysis(vds_results)  # New VDS analysis plot

        # Generate pairwise evaluation plot if available
        if 'pairwise_evaluation' in test_results:
            self._plot_pairwise_results(test_results['pairwise_evaluation'])

        logger.info(f"Results saved to: {self.config.output_dir}")

    def _plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.train_history['train_loss']) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Safe', 'Vulnerable'],
                    yticklabels=['Safe', 'Vulnerable'])
        plt.title(f'Confusion Matrix - {self.config.experiment_type.title()} (GAT)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_vds_analysis(self, vds_results: Dict):
        """Plot VDS analysis"""
        threshold_analysis = vds_results['threshold_analysis']

        if not threshold_analysis:
            logger.warning("No threshold analysis data, skipping VDS analysis plot")
            return

        # Extract data
        thresholds = [point['threshold'] for point in threshold_analysis]
        fprs = [point['fpr'] for point in threshold_analysis]
        fnrs = [point['fnr'] for point in threshold_analysis]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. FPR vs FNR scatter plot
        scatter = ax1.scatter(fprs, fnrs, c=thresholds, cmap='viridis', alpha=0.6)
        ax1.axvline(x=vds_results['fpr_tolerance'], color='red', linestyle='--',
                    label=f'FPR Tolerance ({vds_results["fpr_tolerance"]:.1%})')
        ax1.axhline(y=vds_results['vds'], color='red', linestyle='-',
                    label=f'VDS = {vds_results["vds"]:.4f}')
        ax1.scatter([vds_results['vds_fpr']], [vds_results['vds_fnr']],
                    color='red', s=100, marker='*', label='Optimal Point')
        ax1.set_xlabel('False Positive Rate (FPR)')
        ax1.set_ylabel('False Negative Rate (FNR)')
        ax1.set_title('FPR vs FNR Trade-off (GAT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Threshold')

        # 2. Threshold vs FPR
        ax2.plot(thresholds, fprs, 'b-', marker='o', markersize=3, alpha=0.7)
        ax2.axhline(y=vds_results['fpr_tolerance'], color='red', linestyle='--',
                    label=f'FPR Tolerance ({vds_results["fpr_tolerance"]:.1%})')
        ax2.axvline(x=vds_results['vds_threshold'], color='red', linestyle='-',
                    label=f'Optimal Threshold = {vds_results["vds_threshold"]:.4f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('False Positive Rate (FPR)')
        ax2.set_title('Threshold vs FPR (GAT)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Threshold vs FNR
        ax3.plot(thresholds, fnrs, 'g-', marker='o', markersize=3, alpha=0.7)
        ax3.axhline(y=vds_results['vds'], color='red', linestyle='-',
                    label=f'VDS = {vds_results["vds"]:.4f}')
        ax3.axvline(x=vds_results['vds_threshold'], color='red', linestyle='-',
                    label=f'Optimal Threshold = {vds_results["vds_threshold"]:.4f}')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('False Negative Rate (FNR)')
        ax3.set_title('Threshold vs FNR (GAT)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. VDS confusion matrix
        vds_cm = vds_results['vds_confusion_matrix']
        cm_matrix = np.array([[vds_cm['tn'], vds_cm['fp']],
                              [vds_cm['fn'], vds_cm['tp']]])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['Safe', 'Vulnerable'],
                    yticklabels=['Safe', 'Vulnerable'])
        ax4.set_title(f'VDS Confusion Matrix (GAT)\n(Threshold = {vds_results["vds_threshold"]:.4f})')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "vds_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pairwise_results(self, pairwise_results: Dict):
        """Plot pairwise evaluation results"""
        if 'pair_counts' not in pairwise_results:
            return

        counts = pairwise_results['pair_counts']
        percentages = pairwise_results['pair_percentages']

        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Count pie chart
        labels = list(counts.keys())
        values = list(counts.values())
        colors = ['green', 'red', 'orange', 'purple']

        wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax1.set_title('Pairwise Prediction Distribution (Counts) - GAT')

        # Percentage bar chart
        ax2.bar(labels, [percentages.get(f"{k}_percent", 0) for k in labels],
                color=colors)
        ax2.set_title('Pairwise Prediction Percentages - GAT')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)

        # Add value labels
        for i, (label, value) in enumerate(zip(labels, [percentages.get(f"{k}_percent", 0) for k in labels])):
            ax2.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "pairwise_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()


# ----------------- Experiment Manager -----------------
class ExperimentManager:
    def __init__(self):
        self.results = {}

    def run_experiment(self, experiment_type: str) -> Dict:
        """Run single experiment"""
        logger.info(f"Starting experiment: {experiment_type}")

        # Create configuration
        config = GATTrainingConfig(experiment_type)

        # Create trainer
        trainer = GATTrainer(config)

        # Load datasets
        trainer.load_datasets()

        # Train model
        results = trainer.train()

        return results

    def run_all_experiments(self):
        """Run all experiments (balanced and unbalanced)"""
        experiments = ["unbalanced", "balanced"]

        for exp_type in experiments:
            try:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Starting {exp_type.upper()} dataset experiment")
                logger.info(f"{'=' * 80}")

                results = self.run_experiment(exp_type)
                self.results[exp_type] = results

                logger.info(f"{exp_type.upper()} experiment completed!")

            except Exception as e:
                logger.error(f"{exp_type} experiment failed: {e}")
                import traceback
                logger.error(f"Detailed error info: {traceback.format_exc()}")
                continue

        # Generate comparison report
        self._generate_comparison_report()

    def _generate_comparison_report(self):
        """Generate comparison report"""
        if len(self.results) < 2:
            logger.warning("Insufficient results for comparison report")
            return

        logger.info("=" * 80)
        logger.info("GAT Experiment Comparison Report")
        logger.info("=" * 80)

        comparison = {}
        for exp_type, results in self.results.items():
            comparison[exp_type] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'auc': float(results['auc']),
                'vds': float(results['vds_results']['vds'])  # Add VDS to comparison
            }

            logger.info(f"{exp_type.upper()} dataset results:")
            logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall:    {results['recall']:.4f}")
            logger.info(f"  F1-Score:  {results['f1_score']:.4f}")
            logger.info(f"  AUC:       {results['auc']:.4f}")
            logger.info(f"  VDS:       {results['vds_results']['vds']:.4f}")

            # VDS details
            vds_res = results['vds_results']
            logger.info(f"  VDS details:")
            logger.info(f"    FPR tolerance: {vds_res['fpr_tolerance']:.1%}")
            logger.info(f"    Best threshold: {vds_res['vds_threshold']:.4f}")
            logger.info(f"    Actual FPR: {vds_res['vds_fpr']:.4f}")
            logger.info(f"    Actual FNR: {vds_res['vds_fnr']:.4f}")

            # If pairwise evaluation results exist
            if 'pairwise_evaluation' in results:
                pair_results = results['pairwise_evaluation']
                logger.info(f"  Pairwise evaluation:")
                for key, count in pair_results['pair_counts'].items():
                    percentage = pair_results['pair_percentages'].get(f"{key}_percent", 0)
                    logger.info(f"    {key}: {count} ({percentage:.1f}%)")

                # Add pairwise evaluation results to comparison
                comparison[exp_type]['pairwise_evaluation'] = {
                    'pair_counts': pair_results['pair_counts'],
                    'pair_percentages': pair_results['pair_percentages'],
                    'total_pairs': int(pair_results['total_pairs']),
                    'coverage': float(pair_results['coverage'])
                }
            logger.info("")

        # Save comparison results
        comparison_file = Path("./comparison_report_gat.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"GAT comparison report saved: {comparison_file}")


# ----------------- Main Function -----------------
def main():
    """Main function"""
    # Set environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    })

    import argparse
    parser = argparse.ArgumentParser(description='Pure GAT Vulnerability Detection Training System')
    parser.add_argument('--experiment', choices=['unbalanced', 'balanced', 'both'],
                        default='both', help='Choose experiment type')
    parser.add_argument('--vds-fpr-tolerance', type=float, default=0.005,
                        help='VDS FPR tolerance (default: 0.005, i.e., 0.5%)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Pure GAT Vulnerability Detection Training System - with VDS Metrics")
    logger.info("=" * 80)

    if args.experiment == 'both':
        # Run all experiments
        manager = ExperimentManager()
        manager.run_all_experiments()
    else:
        # Run single experiment
        logger.info(f"Running {args.experiment} experiment")
        manager = ExperimentManager()
        results = manager.run_experiment(args.experiment)
        logger.info(f"{args.experiment} experiment completed!")


if __name__ == "__main__":
    main()