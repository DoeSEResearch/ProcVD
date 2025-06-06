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
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv, global_mean_pool, global_max_pool
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


class Config:
    """Configuration class compatible with pretrained model"""

    def __init__(self):
        # GNN configuration
        self.gnn_hidden_dim = 128
        self.gnn_layers = 2
        # Node type mapping (consistent with pretrained model)
        self.node_types = {
            'IDENTIFIER': 1, 'LITERAL': 2, 'METHOD': 3, 'CALL': 4,
            'CONTROL_STRUCTURE': 5, 'OPERATOR': 6, 'UNKNOWN': 7
        }
        # Edge type mapping (consistent with pretrained model)
        self.edge_types = {'REACHING_DEF': 1, 'DATA_FLOW': 2, 'CONTROL_FLOW': 3, 'CALL': 4}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Utility functions
def make_json_safe(obj):
    """Recursively convert object to JSON-safe format"""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                new_key = "_".join(str(k) for k in key)
            else:
                new_key = str(key) if not isinstance(key, (str, int, float, bool)) and key is not None else key
            new_dict[new_key] = make_json_safe(value)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


# Logging setup
def setup_logging(log_file: str = "inference.log") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
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


class InferenceConfig:
    def __init__(self):
        # Pretrained model path
        self.pretrained_model_path = "../../../result/gnn_model_complete.pt"

        # GraphSON data paths and type mappings
        self.cpg_base_dir = Path("../../../data/primevul_graphson_files")
        self.primevul_labels_dir = "../../../data/primevul_process"

        # Dataset configuration for GraphSON directory structure
        self.datasets = {
            'train': 'PrimeVul_unbalanced_train_sampled',
            'test': 'PrimeVul_unbalanced_test_sampled',
            'valid': 'PrimeVul_unbalanced_valid_sampled',
            'train_paired': 'PrimeVul_balanced_train',
            'test_paired': 'PrimeVul_balanced_test',
            'valid_paired': 'PrimeVul_balanced_valid'
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

        # Inference configuration
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core feature extraction parameters
        self.min_core_nodes = 3
        self.max_core_nodes = 20
        self.top_k_ratio = 0.5

        # Quality validation parameters
        self.enable_quality_validation = True
        self.prediction_diff_threshold = 0.1
        self.min_prediction_preservation = 0.85
        self.embedding_similarity_threshold = 0.7

        # Output paths
        self.output_dir = "../../../result/primevul_core_all_graph_features"
        self.analysis_dir = "../../../result/analysis_all_graph_results"

        # Node and edge types (consistent with pretrained model for inference compatibility)
        self.node_types = {
            'IDENTIFIER': 1, 'LITERAL': 2, 'METHOD': 3, 'CALL': 4,
            'CONTROL_STRUCTURE': 5, 'OPERATOR': 6, 'UNKNOWN': 7
        }
        self.edge_types = {'REACHING_DEF': 1, 'DATA_FLOW': 2, 'CONTROL_FLOW': 3, 'CALL': 4}

        # New data format node and edge type mappings (for data processing)
        self.new_node_types = {
            'UNKNOWN': 0, 'METHOD': 1, 'BINDING': 2, 'LOCAL': 3, 'BLOCK': 4, 'CALL': 5,
            'CONTROL_STRUCTURE': 6, 'RETURN': 7, 'TYPE': 8, 'IDENTIFIER': 9,
            'LITERAL': 10, 'METHOD_RETURN': 11, 'FILE': 12, 'NAMESPACE_BLOCK': 13,
            'TYPE_DECL': 14, 'MEMBER': 15, 'MODIFIER': 16, 'ANNOTATION': 17,
        }
        self.new_edge_types = {
            'UNKNOWN': 0, 'REF': 1, 'AST': 2, 'EVAL_TYPE': 3, 'CALL': 4,
            'ARGUMENT': 5, 'CFG': 6, 'DOMINATE': 7, 'POST_DOMINATE': 8,
            'REACHING_DEF': 9, 'CONTAINS': 10, 'PARAMETER_LINK': 11,
            'SOURCE_FILE': 12,
        }


class EnhancedGNN(nn.Module):
    """Enhanced GNN model (identical to training version for compatibility)"""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.gnn_hidden_dim
        self.config = config

        # Keep consistent input dimension with pretrained model - 10 features
        self.input_proj = nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Multi-layer GNN
        self.layers = nn.ModuleList()
        self.attention_weights = []

        # GINE layer
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

        # GAT layer
        self.gat_layer = GATv2Conv(
            self.hidden_dim,
            self.hidden_dim,
            edge_dim=len(config.edge_types) + 1,
            heads=4,
            concat=False,
            dropout=0.1
        )
        self.layers.append(self.gat_layer)

        # Batch normalization
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
        self.attention_weights = []

        # Input projection
        x = self.input_proj(x)
        layer_outputs = [x]

        # GINE layer
        if edge_attr is not None and edge_attr.size(0) > 0 and edge_index.size(1) > 0:
            x_new = self.gine_layer(x, edge_index, edge_attr)
        else:
            if edge_index.size(1) > 0:
                default_edge_attr = torch.zeros((edge_index.size(1), len(self.config.edge_types) + 1),
                                                device=x.device, dtype=x.dtype)
                x_new = self.gine_layer(x, edge_index, default_edge_attr)
            else:
                x_new = x

        x_new = F.relu(x_new)
        skip_weight = torch.sigmoid(self.skip_weights[0])
        x = skip_weight * x + (1 - skip_weight) * x_new
        x = self.norms[0](x)
        layer_outputs.append(x)

        # GAT layer
        if edge_attr is not None and edge_attr.size(0) > 0 and edge_index.size(1) > 0:
            x_new, (edge_index_att, attention_weights) = self.gat_layer(
                x, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            if edge_index.size(1) > 0:
                default_edge_attr = torch.zeros((edge_index.size(1), len(self.config.edge_types) + 1),
                                                device=x.device, dtype=x.dtype)
                x_new, (edge_index_att, attention_weights) = self.gat_layer(
                    x, edge_index, default_edge_attr, return_attention_weights=True
                )
            else:
                x_new = x
                attention_weights = torch.empty((0,), device=x.device)

        self.attention_weights = attention_weights if 'attention_weights' in locals() else torch.empty((0,),
                                                                                                       device=x.device)
        x_new = F.relu(x_new)
        skip_weight = torch.sigmoid(self.skip_weights[1])
        x = skip_weight * x + (1 - skip_weight) * x_new
        x = self.norms[1](x)
        layer_outputs.append(x)

        # Fuse outputs
        final_output = sum(layer_outputs) / len(layer_outputs)
        node_importance_scores = self.node_importance(final_output)

        if return_attention:
            return final_output, node_importance_scores, self.attention_weights
        return final_output


class GNNOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gnn = EnhancedGNN(config)

        # Graph-level attention
        self.graph_attention = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.LeakyReLU(0.2)
        )

        # Classifier
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
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, graph: GeoData, device: torch.device, return_attention=False):
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
        batch_size = graph.batch.max().item() + 1 if graph.batch is not None else 1
        graph_emb_att_list = []

        for i in range(batch_size):
            if graph.batch is not None:
                mask = (graph.batch == i)
            else:
                mask = torch.ones(node_emb.size(0), dtype=torch.bool, device=node_emb.device)

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

        graph_emb = torch.cat([graph_emb_mean, graph_emb_max, graph_emb_att], dim=1)
        logits = self.classifier(graph_emb)

        if return_attention:
            return logits, node_importance, edge_attention
        return logits


class GraphSONDataset(Dataset):
    """Dataset loader for GraphSON format"""

    def __init__(self, graphson_files: List[Path], config: InferenceConfig):
        self.config = config
        self.data = []
        self.file_to_idx = {}
        self.failed_files = []
        self.failed_reasons = defaultdict(int)

        logger.info(f"Loading {len(graphson_files)} GraphSON files...")

        # Debug file formats
        self._debug_file_formats(graphson_files[:5])

        for idx, graphson_file in enumerate(tqdm(graphson_files, desc="Loading GraphSON data")):
            try:
                graph_data = self._load_graphson_file(graphson_file)
                if graph_data is None:
                    self.failed_files.append(str(graphson_file))
                    self.failed_reasons['load_failed'] += 1
                    continue

                if not self._validate_graph(graph_data):
                    self.failed_files.append(str(graphson_file))
                    self.failed_reasons['validation_failed'] += 1
                    continue

                # Extract idx from filename
                file_idx = graphson_file.stem

                self.data.append({
                    'graph': graph_data,
                    'file_path': str(graphson_file),
                    'file_name': graphson_file.stem,
                    'file_idx': file_idx,
                    'dataset_type': graphson_file.parent.name
                })
                self.file_to_idx[graphson_file.stem] = idx

            except Exception as e:
                logger.warning(f"Failed to load file {graphson_file}: {e}")
                self.failed_files.append(str(graphson_file))
                self.failed_reasons['exception'] += 1
                continue

        logger.info(f"Successfully loaded {len(self.data)} valid GraphSON graphs")

        # Output failure statistics
        logger.info(f"Failed files count: {len(self.failed_files)}")
        for reason, count in self.failed_reasons.items():
            logger.info(f"  - {reason}: {count}")

        if self.failed_files:
            logger.info(f"First 5 failed files: {self.failed_files[:5]}")

        self._log_statistics()

    def _debug_file_formats(self, sample_files: List[Path]):
        """Debug file format samples"""
        logger.info("Checking file format samples...")
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
                    logger.info(f"File {file_path.name} first 100 chars: {content[:100]}")

                    # Try parsing JSON
                    f.seek(0)
                    full_content = f.read()
                    if full_content.strip():
                        try:
                            data = json.loads(full_content)
                            logger.info(
                                f"  - JSON parse success, keys: {list(data.keys()) if isinstance(data, dict) else 'non-dict'}")
                            if isinstance(data, dict):
                                if 'vertices' in data:
                                    logger.info(f"  - vertices count: {len(data['vertices'])}")
                                if 'edges' in data:
                                    logger.info(f"  - edges count: {len(data['edges'])}")
                                # Check other possible keys
                                other_keys = [k for k in data.keys() if k not in ['vertices', 'edges']]
                                if other_keys:
                                    logger.info(f"  - other keys: {other_keys}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"  - JSON parse failed: {e}")
                    break  # Only check first valid file
            except Exception as e:
                logger.warning(f"Error checking file {file_path}: {e}")

    def _load_graphson_file(self, file_path: Path) -> Optional[Dict]:
        """Load GraphSON format file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"File {file_path} is empty")
                    return None

                # Try parsing JSON
                try:
                    graph_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed {file_path}: {e}")
                    return None

                # Check data format and convert
                converted_graph = self._convert_graphson_to_graph(graph_data, file_path)
                return converted_graph

        except Exception as e:
            logger.warning(f"Failed to load GraphSON file {file_path}: {e}")
            return None

    def _convert_graphson_to_graph(self, graphson_data: Dict, file_path: Path) -> Optional[Dict]:
        """Convert GraphSON format to graph data format"""
        nodes = []
        edges = []

        # Handle standard GraphSON format: {"@type": "tinker:graph", "@value": {"vertices": [...], "edges": [...]}}
        if isinstance(graphson_data, dict):
            # Check if it's standard GraphSON format
            if "@type" in graphson_data and "@value" in graphson_data:
                if graphson_data.get("@type") == "tinker:graph":
                    graph_value = graphson_data.get("@value", {})
                    vertices = graph_value.get("vertices", [])
                    edges_data = graph_value.get("edges", [])

                    logger.debug(
                        f"Parsing standard GraphSON format, vertices: {len(vertices)}, edges: {len(edges_data)}")
                else:
                    logger.warning(f"File {file_path} is not tinker:graph format, @type: {graphson_data.get('@type')}")
                    return None
            else:
                # Handle other possible formats
                vertices = graphson_data.get('vertices', graphson_data.get('nodes', []))
                edges_data = graphson_data.get('edges', [])
        else:
            logger.warning(f"File {file_path} contains non-dict data: {type(graphson_data)}")
            return None

        # If no vertices but edges exist, try to infer vertices from edges
        if not vertices and edges_data:
            logger.warning(f"File {file_path} has only edges, no vertices, trying to infer vertices from edges")
            # Extract vertex information from edges
            vertex_ids = set()
            vertex_labels = {}

            for edge in edges_data:
                if isinstance(edge, dict):
                    out_v = self._extract_graphson_id(edge.get('outV'))
                    in_v = self._extract_graphson_id(edge.get('inV'))
                    out_label = edge.get('outVLabel', 'UNKNOWN')
                    in_label = edge.get('inVLabel', 'UNKNOWN')

                    if out_v is not None:
                        vertex_ids.add(out_v)
                        vertex_labels[out_v] = out_label
                    if in_v is not None:
                        vertex_ids.add(in_v)
                        vertex_labels[in_v] = in_label

            # Create virtual vertices
            vertices = []
            for vertex_id in vertex_ids:
                vertices.append({
                    "@type": "g:Vertex",
                    "id": {"@type": "g:Int64", "@value": vertex_id},
                    "label": vertex_labels.get(vertex_id, 'UNKNOWN'),
                    "properties": {}
                })

            logger.info(f"Inferred {len(vertices)} vertices from edges")

        if not vertices:
            logger.warning(
                f"No node data found in file {file_path}, available keys: {list(graphson_data.keys()) if isinstance(graphson_data, dict) else 'non-dict'}")
            return None

        # Process vertices
        node_id_map = {}
        for i, vertex in enumerate(vertices):
            try:
                # Handle GraphSON format vertex
                if isinstance(vertex, dict):
                    # Extract ID
                    vertex_id = self._extract_graphson_id(vertex.get('id'))
                    if vertex_id is None:
                        vertex_id = i  # Fallback ID

                    # Extract label
                    node_label = vertex.get('label', 'UNKNOWN')

                    # Extract properties
                    properties = vertex.get('properties', {})

                    # Handle GraphSON format properties (possibly nested structure)
                    line_number = self._extract_graphson_property(properties, 'lineNumber', 0)
                    code = self._extract_graphson_property(properties, 'code', '')
                    ast_depth = self._extract_graphson_property(properties, 'astDepth', 0)
                    is_vulnerable = self._extract_graphson_property(properties, 'isVulnerable', 0)
                else:
                    # If vertex is not a dict, use default values
                    vertex_id = i
                    node_label = 'UNKNOWN'
                    line_number = 0
                    code = ''
                    ast_depth = 0
                    is_vulnerable = 0

                node = {
                    'id': vertex_id,
                    'label': node_label,
                    'lineNumber': line_number,
                    'code': code,
                    'astDepth': ast_depth,
                    'isVulnerable': is_vulnerable
                }

                nodes.append(node)
                node_id_map[vertex_id] = i

            except Exception as e:
                logger.debug(f"Error processing vertex {i}: {e}")
                continue

        # Process edges
        for edge in edges_data:
            try:
                if isinstance(edge, dict):
                    # Handle GraphSON format edge
                    src_id = self._extract_graphson_id(edge.get('outV'))
                    dst_id = self._extract_graphson_id(edge.get('inV'))
                    edge_label = edge.get('label', 'UNKNOWN')

                    if src_id is not None and dst_id is not None and src_id in node_id_map and dst_id in node_id_map:
                        edges.append({
                            'src': src_id,
                            'dst': dst_id,
                            'label': edge_label
                        })
            except Exception as e:
                logger.debug(f"Error processing edge: {e}")
                continue

        if not nodes:
            logger.warning(f"File {file_path} has no valid nodes")
            return None

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _extract_graphson_id(self, id_obj):
        """Extract GraphSON format ID"""
        if id_obj is None:
            return None

        if isinstance(id_obj, dict):
            # GraphSON format: {"@type": "g:Int64", "@value": 123}
            if "@value" in id_obj:
                return id_obj["@value"]
            elif "value" in id_obj:
                return id_obj["value"]
            else:
                return None
        else:
            # Direct ID value
            return id_obj

    def _extract_graphson_property(self, properties, prop_name, default_value):
        """Extract GraphSON format property"""
        if not isinstance(properties, dict):
            return default_value

        prop = properties.get(prop_name)
        if prop is None:
            return default_value

        # Handle GraphSON property format
        if isinstance(prop, list) and len(prop) > 0:
            # GraphSON property format: [{"@type": "...", "@value": ...}]
            prop_item = prop[0]
            if isinstance(prop_item, dict):
                if "@value" in prop_item:
                    return prop_item["@value"]
                elif "value" in prop_item:
                    return prop_item["value"]
                else:
                    return prop_item
            else:
                return prop_item
        elif isinstance(prop, dict):
            # Single property object
            if "@value" in prop:
                return prop["@value"]
            elif "value" in prop:
                return prop["value"]
            else:
                return default_value
        else:
            # Direct value
            return prop

    def _extract_property_value(self, prop):
        """Extract GraphSON property value"""
        if prop is None:
            return 0

        # Handle GraphSON standard format
        if isinstance(prop, list) and len(prop) > 0:
            # GraphSON property format: [{"id": ..., "value": ...}]
            if isinstance(prop[0], dict):
                return prop[0].get('value', 0)
            else:
                return prop[0]
        elif isinstance(prop, dict):
            return prop.get('value', prop.get('val', 0))
        else:
            # Direct value
            return prop

    def _map_new_to_old_node_type(self, new_label: str) -> str:
        """Map new node types to old node types"""
        mapping = {
            'METHOD': 'METHOD',
            'CALL': 'CALL',
            'CONTROL_STRUCTURE': 'CONTROL_STRUCTURE',
            'IDENTIFIER': 'IDENTIFIER',
            'LITERAL': 'LITERAL',
            'BINDING': 'IDENTIFIER',
            'LOCAL': 'IDENTIFIER',
            'BLOCK': 'CONTROL_STRUCTURE',
            'RETURN': 'CONTROL_STRUCTURE',
            'TYPE': 'LITERAL',
            'METHOD_RETURN': 'METHOD',
            'FILE': 'UNKNOWN',
            'NAMESPACE_BLOCK': 'UNKNOWN',
            'TYPE_DECL': 'LITERAL',
            'MEMBER': 'IDENTIFIER',
            'MODIFIER': 'LITERAL',
            'ANNOTATION': 'LITERAL',
            'UNKNOWN': 'UNKNOWN'
        }
        return mapping.get(new_label, 'UNKNOWN')

    def _map_new_to_old_edge_type(self, new_label: str) -> str:
        """Map new edge types to old edge types"""
        mapping = {
            'CALL': 'CALL',
            'CFG': 'CONTROL_FLOW',
            'REACHING_DEF': 'REACHING_DEF',
            'REF': 'DATA_FLOW',
            'AST': 'DATA_FLOW',
            'EVAL_TYPE': 'DATA_FLOW',
            'ARGUMENT': 'DATA_FLOW',
            'DOMINATE': 'CONTROL_FLOW',
            'POST_DOMINATE': 'CONTROL_FLOW',
            'CONTAINS': 'DATA_FLOW',
            'PARAMETER_LINK': 'DATA_FLOW',
            'SOURCE_FILE': 'DATA_FLOW',
            'UNKNOWN': 'DATA_FLOW'
        }
        return mapping.get(new_label, 'DATA_FLOW')

    def _validate_graph(self, graph_data: Dict) -> bool:
        """Validate graph data validity"""
        if not graph_data or 'nodes' not in graph_data:
            logger.debug("Graph data missing nodes field")
            return False

        nodes = graph_data.get('nodes', [])
        if len(nodes) < 1:
            logger.debug(f"Insufficient nodes: {len(nodes)}")
            return False

        # Check if nodes have basic fields
        valid_nodes = 0
        for node in nodes:
            if isinstance(node, dict) and 'id' in node:
                valid_nodes += 1

        if valid_nodes == 0:
            logger.debug("No valid nodes")
            return False

        return True

    def _log_statistics(self):
        """Log dataset statistics"""
        if not self.data:
            return

        dataset_counts = defaultdict(int)
        node_counts = []
        edge_counts = []

        for item in self.data:
            dataset_counts[item['dataset_type']] += 1
            nodes = item['graph'].get('nodes', [])
            edges = item['graph'].get('edges', [])
            node_counts.append(len(nodes))
            edge_counts.append(len(edges))

        logger.info(f"Dataset distribution: {dict(dataset_counts)}")
        if node_counts:
            logger.info(f"Average nodes: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}")
            logger.info(f"Average edges: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}")
            logger.info(f"Node range: {min(node_counts)} - {max(node_counts)}")
            logger.info(f"Edge range: {min(edge_counts)} - {max(edge_counts)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        graph_data = item['graph']

        # Process node features (consistent with training)
        nodes = graph_data['nodes']
        edges = graph_data.get('edges', [])

        id2idx = {n.get('id', i): i for i, n in enumerate(nodes)}

        # Build node feature matrix - maintain 10 features consistent with pretrained model
        x = []
        for n in nodes:
            # Original node label
            original_node_label = n.get('label', 'UNKNOWN')
            # Map to old node type
            mapped_node_label = self._map_new_to_old_node_type(original_node_label)
            node_type = self.config.node_types.get(mapped_node_label, self.config.node_types['UNKNOWN'])

            code = str(n.get('code', ''))
            code_len = len(code)
            line_number = float(n.get('lineNumber', 0))
            ast_depth = float(n.get('astDepth', 0))
            is_vulnerable = float(n.get('isVulnerable', 0))

            # Vulnerability pattern detection (consistent with original model)
            vuln_patterns = ['strcpy', 'malloc', 'free', 'memcpy', 'sprintf', 'gets', 'scanf', 'strcat']
            has_vuln_pattern = int(any(p in code.lower() for p in vuln_patterns))

            safe_patterns = ['__libc_enable_secure', 'strncpy', 'snprintf', 'fgets', 'strncat']
            has_safe_pattern = int(any(p in code for p in safe_patterns))

            has_origin = int('ORIGIN' in code or 'PLATFORM' in code)
            is_literal = int(mapped_node_label == 'LITERAL')
            is_identifier = int(mapped_node_label == 'IDENTIFIER')

            # 10 features (completely consistent with pretrained model)
            features = [
                line_number / 100.0,
                0.0,  # Original columnNumber, not in new data, set to 0
                node_type / len(self.config.node_types),
                min(code_len / 50.0, 1.0),
                has_vuln_pattern,
                has_safe_pattern,
                has_origin,
                is_literal,
                is_identifier,
                float(code.count('=') > 0),
            ]
            x.append(features)

        x = torch.tensor(x, dtype=torch.float)

        # Process edges
        if edges:
            src_list, dst_list, edge_features = [], [], []

            for e in edges:
                src_id = e.get('src')
                dst_id = e.get('dst')

                if src_id in id2idx and dst_id in id2idx:
                    src_list.append(id2idx[src_id])
                    dst_list.append(id2idx[dst_id])

                    # Original edge label
                    original_edge_label = e.get('label', 'UNKNOWN')
                    # Map to old edge type
                    mapped_edge_label = self._map_new_to_old_edge_type(original_edge_label)
                    edge_type = self.config.edge_types.get(mapped_edge_label, 0)
                    edge_feat = one_hot(torch.tensor(edge_type),
                                        num_classes=len(self.config.edge_types) + 1)
                    edge_features.append(edge_feat)

            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                edge_attr = torch.stack(edge_features).float()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.config.edge_types) + 1), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config.edge_types) + 1), dtype=torch.float)

        # Return graph data and metadata
        graph = GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph, item


def custom_collate_fn(batch):
    """Custom collate function for torch_geometric.data.Data objects"""
    graphs, items = zip(*batch)
    return list(graphs), list(items)


class PrimeVulLabelLoader:
    """Optimized label data loader"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.labels_cache = {}

    def load_labels(self, dataset_name: str) -> Dict[str, int]:
        """Load labels for specified dataset"""
        if dataset_name in self.labels_cache:
            return self.labels_cache[dataset_name]

        if dataset_name not in self.config.label_files:
            logger.warning(f"Label file configuration not found for dataset {dataset_name}")
            return {}

        label_file = Path(self.config.primevul_labels_dir) / self.config.label_files[dataset_name]
        if not label_file.exists():
            logger.error(f"Label file does not exist: {label_file}")
            return {}

        labels = {}
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Use idx as key, which corresponds to graph data filename
                        idx = str(data.get('idx', ''))
                        target = data.get('target', 0)

                        if idx:
                            labels[idx] = target
                        else:
                            logger.warning(f"Line {line_num} missing idx: {line[:100]}...")

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse failed at line {line_num}: {e}")
                        continue

            logger.info(f"Successfully loaded {len(labels)} labels from {label_file}")
            self.labels_cache[dataset_name] = labels
            return labels

        except Exception as e:
            logger.error(f"Failed to load label file {label_file}: {e}")
            return {}

    def get_label_by_idx(self, dataset_name: str, file_idx: str) -> Optional[int]:
        """Get label by file index"""
        labels = self.load_labels(dataset_name)

        # Directly find label by idx
        label = labels.get(file_idx, None)

        if label is None:
            logger.debug(f"Label not found for file index {file_idx}")

        return label


class CoreFeatureExtractor:
    def __init__(self, model: GNNOnlyModel, config: InferenceConfig):
        self.model = model
        self.config = config
        self.model.eval()
        self.label_loader = PrimeVulLabelLoader(config)

        # Quality validation statistics
        self.quality_stats = {
            'total_processed': 0,
            'high_quality_count': 0,
            'medium_quality_count': 0,
            'low_quality_count': 0,
            'prediction_preservation_scores': [],
            'embedding_similarity_scores': [],
            'compression_efficiency_scores': [],
            'label_consistency_scores': [],
            'accuracy_preservation_scores': [],
            'label_match_count': 0,
            'label_mismatch_count': 0,
        }

    def evaluate_core_quality_with_labels(self, original_graph: GeoData, core_graph: GeoData,
                                          true_label: int, dataset_name: str,
                                          original_embedding: torch.Tensor = None) -> Dict:
        """Evaluate core subgraph quality using true labels"""
        if not self.config.enable_quality_validation:
            return {'quality_score': 1.0, 'quality_level': 'not_evaluated'}

        quality_metrics = {}

        try:
            with torch.no_grad():
                # Get predictions for original and core graphs
                orig_logits = self.model(original_graph, self.config.device)
                core_logits = self.model(core_graph, self.config.device)

                orig_probs = torch.softmax(orig_logits, dim=1)
                core_probs = torch.softmax(core_logits, dim=1)

                # Get predicted labels
                orig_pred = torch.argmax(orig_probs, dim=1).item()
                core_pred = torch.argmax(core_probs, dim=1).item()

                # Prediction preservation evaluation (KL divergence)
                kl_div = torch.nn.functional.kl_div(
                    torch.log(core_probs + 1e-8), orig_probs, reduction='batchmean'
                ).item()
                prediction_preservation = max(0, 1 - kl_div)
                quality_metrics['prediction_preservation'] = prediction_preservation

                # Label consistency evaluation
                label_consistency = 1.0 if orig_pred == core_pred else 0.0
                quality_metrics['label_consistency'] = label_consistency

                # Accuracy preservation evaluation
                orig_correct = 1.0 if orig_pred == true_label else 0.0
                core_correct = 1.0 if core_pred == true_label else 0.0

                if orig_correct == core_correct:
                    accuracy_preservation = 1.0
                else:
                    accuracy_preservation = 0.0

                quality_metrics['accuracy_preservation'] = accuracy_preservation
                quality_metrics['original_accuracy'] = orig_correct
                quality_metrics['core_accuracy'] = core_correct

                # Embedding similarity evaluation
                if original_embedding is not None:
                    core_embedding = self.get_core_embedding(core_graph)
                    embedding_similarity = torch.nn.functional.cosine_similarity(
                        original_embedding.unsqueeze(0),
                        core_embedding.unsqueeze(0)
                    ).item()
                    quality_metrics['embedding_similarity'] = max(0, embedding_similarity)
                else:
                    quality_metrics['embedding_similarity'] = 0.5

                # Compression efficiency evaluation
                compression_ratio = core_graph.x.size(0) / original_graph.x.size(0)
                compression_efficiency = (1 - compression_ratio) * prediction_preservation
                quality_metrics['compression_efficiency'] = compression_efficiency

                # Comprehensive quality score
                weights = {
                    'prediction_preservation': 0.25,
                    'label_consistency': 0.25,
                    'accuracy_preservation': 0.30,
                    'embedding_similarity': 0.10,
                    'compression_efficiency': 0.10
                }

                quality_score = sum(
                    quality_metrics[metric] * weight
                    for metric, weight in weights.items()
                )

                # Quality level classification
                if quality_score >= 0.8:
                    quality_level = 'high'
                    self.quality_stats['high_quality_count'] += 1
                elif quality_score >= 0.6:
                    quality_level = 'medium'
                    self.quality_stats['medium_quality_count'] += 1
                else:
                    quality_level = 'low'
                    self.quality_stats['low_quality_count'] += 1

                # Update statistics
                self.quality_stats['total_processed'] += 1
                self.quality_stats['prediction_preservation_scores'].append(prediction_preservation)
                self.quality_stats['embedding_similarity_scores'].append(quality_metrics['embedding_similarity'])
                self.quality_stats['compression_efficiency_scores'].append(compression_efficiency)
                self.quality_stats['label_consistency_scores'].append(label_consistency)
                self.quality_stats['accuracy_preservation_scores'].append(accuracy_preservation)

                # Add detailed information
                quality_metrics.update({
                    'quality_score': quality_score,
                    'quality_level': quality_level,
                    'kl_divergence': kl_div,
                    'compression_ratio': compression_ratio,
                    'true_label': true_label,
                    'original_prediction': orig_pred,
                    'core_prediction': core_pred,
                    'original_confidence': orig_probs.max().item(),
                    'core_confidence': core_probs.max().item()
                })

                return quality_metrics

        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return {
                'quality_score': 0.5,
                'quality_level': 'evaluation_failed',
                'error': str(e)
            }

    def adaptive_core_extraction_with_labels(self, graph: GeoData, true_label: int,
                                             dataset_name: str, return_details=False):
        """Adaptive core subgraph extraction with label validation"""
        best_core_graph = None
        best_quality = 0
        best_details = None

        # Try different top_k_ratio values
        ratios_to_try = [0.6, 0.5, 0.4, 0.3] if graph.x.size(0) > 15 else [0.7, 0.6, 0.5]

        original_embedding = None
        if self.config.enable_quality_validation:
            # Get original graph embedding as baseline
            original_embedding = self.get_original_embedding(graph)

        for ratio in ratios_to_try:
            try:
                # Temporarily modify configuration
                original_ratio = self.config.top_k_ratio
                self.config.top_k_ratio = ratio

                # Extract core subgraph
                core_graph, details = self.extract_attention_based_core(graph, return_details=True)

                # Restore original configuration
                self.config.top_k_ratio = original_ratio

                if core_graph.x.size(0) == 0:
                    continue

                # Quality evaluation using true labels
                if self.config.enable_quality_validation:
                    quality_metrics = self.evaluate_core_quality_with_labels(
                        graph, core_graph, true_label, dataset_name, original_embedding
                    )
                    details['quality_metrics'] = quality_metrics

                    # Update best result if quality is better
                    if quality_metrics['quality_score'] > best_quality:
                        best_quality = quality_metrics['quality_score']
                        best_core_graph = core_graph
                        best_details = details

                    # Early stopping if high quality is achieved
                    if quality_metrics['quality_score'] >= 0.8:
                        break
                else:
                    # Use first valid result when quality validation is disabled
                    best_core_graph = core_graph
                    best_details = details
                    break

            except Exception as e:
                logger.warning(f"Adaptive extraction failed (ratio={ratio}): {e}")
                continue

        # Use default method if all attempts fail
        if best_core_graph is None:
            logger.warning("Adaptive extraction failed, using default method")
            return self.extract_attention_based_core(graph, return_details)

        if return_details:
            return best_core_graph, best_details
        return best_core_graph

    def get_original_embedding(self, graph: GeoData) -> torch.Tensor:
        """Get original graph embedding vector"""
        with torch.no_grad():
            graph = graph.to(self.config.device)

            # Ensure correct batch information
            if graph.batch is None:
                graph.batch = torch.zeros(graph.x.size(0),
                                          dtype=torch.long,
                                          device=graph.x.device)

            try:
                # Get node embeddings
                node_emb = self.model.gnn(
                    graph.x, graph.edge_index, graph.edge_attr, graph.batch
                )

                # Use same pooling strategy as model
                graph_emb_mean = global_mean_pool(node_emb, graph.batch)
                graph_emb_max = global_max_pool(node_emb, graph.batch)

                # Attention-weighted pooling
                att_scores = self.model.graph_attention(node_emb)
                att_weights = F.softmax(att_scores, dim=0)
                graph_emb_att = (node_emb * att_weights).sum(dim=0, keepdim=True)

                # Combine embeddings
                final_embedding = torch.cat([graph_emb_mean, graph_emb_max, graph_emb_att], dim=1)

                return final_embedding.squeeze().cpu()

            except Exception as e:
                logger.warning(f"Failed to get original embedding: {e}")
                # Fallback: simple average
                try:
                    node_emb = self.model.gnn(
                        graph.x, graph.edge_index, graph.edge_attr, graph.batch
                    )
                    if node_emb.size(0) > 0:
                        # Simple average
                        avg_emb = node_emb.mean(dim=0)
                        # Replicate 3 times to match expected dimension
                        final_emb = torch.cat([avg_emb, avg_emb, avg_emb], dim=0)
                        return final_emb.cpu()
                    else:
                        return torch.zeros(384, device='cpu')  # 128 * 3
                except Exception as e2:
                    logger.error(f"Fallback method failed: {e2}")
                    return torch.zeros(384, device='cpu')

    def get_quality_summary(self) -> Dict:
        """Get quality validation summary"""
        if self.quality_stats['total_processed'] == 0:
            return {'message': 'No quality validation performed'}

        total = self.quality_stats['total_processed']

        summary = {
            'total_processed': total,
            'label_statistics': {
                'matched_labels': self.quality_stats['label_match_count'],
                'unmatched_labels': self.quality_stats['label_mismatch_count'],
                'match_rate': (self.quality_stats['label_match_count'] /
                               max(1, self.quality_stats['label_match_count'] + self.quality_stats[
                                   'label_mismatch_count'])) * 100
            },
            'quality_distribution': {
                'high_quality': {
                    'count': self.quality_stats['high_quality_count'],
                    'percentage': self.quality_stats['high_quality_count'] / total * 100
                },
                'medium_quality': {
                    'count': self.quality_stats['medium_quality_count'],
                    'percentage': self.quality_stats['medium_quality_count'] / total * 100
                },
                'low_quality': {
                    'count': self.quality_stats['low_quality_count'],
                    'percentage': self.quality_stats['low_quality_count'] / total * 100
                }
            },
            'average_metrics': {
                'prediction_preservation': np.mean(self.quality_stats['prediction_preservation_scores']),
                'embedding_similarity': np.mean(self.quality_stats['embedding_similarity_scores']),
                'compression_efficiency': np.mean(self.quality_stats['compression_efficiency_scores']),
                'label_consistency': np.mean(self.quality_stats['label_consistency_scores']),
                'accuracy_preservation': np.mean(self.quality_stats['accuracy_preservation_scores'])
            },
            'metric_std': {
                'prediction_preservation': np.std(self.quality_stats['prediction_preservation_scores']),
                'embedding_similarity': np.std(self.quality_stats['embedding_similarity_scores']),
                'compression_efficiency': np.std(self.quality_stats['compression_efficiency_scores']),
                'label_consistency': np.std(self.quality_stats['label_consistency_scores']),
                'accuracy_preservation': np.std(self.quality_stats['accuracy_preservation_scores'])
            }
        }

        return summary

    def extract_attention_based_core(self, graph: GeoData, return_details=False):
        """Extract core subgraph based on attention mechanism"""
        with torch.no_grad():
            try:
                # Get complete forward pass results
                logits, node_importance, edge_attention = self.model(
                    graph, self.config.device, return_attention=True
                )
            except Exception as e:
                logger.warning(f"Failed to get attention weights: {e}, using default method")
                # Fallback: don't use attention weights
                logits = self.model(graph, self.config.device)
                node_importance = None
                edge_attention = None

            num_nodes = graph.x.size(0)

            # Method 1: Based on node importance scores
            if (node_importance is not None and
                    node_importance.numel() > 0):

                importance_scores = node_importance.squeeze().cpu()

                # Ensure correct dimension
                if importance_scores.dim() == 0:
                    importance_scores = importance_scores.unsqueeze(0)

                # Handle dimension mismatch
                if len(importance_scores) != num_nodes:
                    logger.warning(f"Node importance dimension mismatch: {len(importance_scores)} vs {num_nodes}")
                    if len(importance_scores) > num_nodes:
                        importance_scores = importance_scores[:num_nodes]
                    else:
                        # Pad with average value
                        mean_val = importance_scores.mean() if len(importance_scores) > 0 else 0.5
                        pad_size = num_nodes - len(importance_scores)
                        padding = torch.full((pad_size,), mean_val, dtype=importance_scores.dtype)
                        importance_scores = torch.cat([importance_scores, padding])
            else:
                # Fallback: uniform distribution
                importance_scores = torch.ones(num_nodes) * 0.5

            # Method 2: Based on edge attention weights
            edge_scores = {}
            if (edge_attention is not None and
                    graph.edge_index.size(1) > 0):

                try:
                    edge_attention_cpu = edge_attention.cpu()
                    num_edges = graph.edge_index.size(1)

                    # Safer edge attention processing
                    for i, (src, dst) in enumerate(graph.edge_index.t()):
                        # Ensure no out-of-bounds
                        if i >= min(len(edge_attention_cpu), num_edges):
                            break

                        try:
                            edge_key = f"{src.item()}_{dst.item()}"
                            att_tensor = edge_attention_cpu[i]

                            # Handle different attention weight formats
                            if att_tensor.dim() > 0:
                                # Multi-head attention: take average
                                att_value = float(att_tensor.mean().item())
                            else:
                                # Single value attention
                                att_value = float(att_tensor.item())

                            # Ensure value in reasonable range
                            att_value = max(0.0, min(1.0, att_value))
                            edge_scores[edge_key] = att_value

                        except (RuntimeError, ValueError, IndexError):
                            # Skip problematic edges
                            continue

                except Exception as e:
                    logger.debug(f"Error processing edge attention: {e}")
                    edge_scores = {}

            # Combined scoring: node importance + edge attention
            final_node_scores = importance_scores.clone()

            # Adjust node scores based on edge attention
            edge_boost = 0.2
            for edge_key, att_weight in edge_scores.items():
                try:
                    src_str, dst_str = edge_key.split('_')
                    src, dst = int(src_str), int(dst_str)

                    # Ensure valid indices
                    if (0 <= src < len(final_node_scores) and
                            0 <= dst < len(final_node_scores)):
                        final_node_scores[src] += att_weight * edge_boost
                        final_node_scores[dst] += att_weight * edge_boost
                except (ValueError, IndexError):
                    continue

            # Safe normalization
            min_score = final_node_scores.min()
            max_score = final_node_scores.max()
            if max_score > min_score:
                final_node_scores = (final_node_scores - min_score) / (max_score - min_score)

            # Select core nodes
            k = max(self.config.min_core_nodes,
                    min(self.config.max_core_nodes,
                        int(num_nodes * self.config.top_k_ratio)))
            k = min(k, num_nodes)  # Ensure not exceeding total nodes

            if k > 0:
                try:
                    _, core_node_indices = torch.topk(final_node_scores, k)
                    core_node_indices = core_node_indices.sort()[0]
                except RuntimeError:
                    # Fallback when topk fails
                    logger.warning("topk failed, using threshold method")
                    threshold = final_node_scores.median()
                    core_node_indices = torch.where(final_node_scores >= threshold)[0]
                    if len(core_node_indices) == 0:
                        core_node_indices = torch.tensor([0])  # Select at least one node
            else:
                core_node_indices = torch.tensor([0])

            # Build core subgraph
            core_graph = self._build_core_subgraph(graph, core_node_indices)

            if return_details:
                return core_graph, {
                    'original_nodes': num_nodes,
                    'core_nodes': len(core_node_indices),
                    'core_node_indices': core_node_indices.tolist(),
                    'node_importance': importance_scores.tolist(),
                    'edge_attention': edge_scores,
                    'final_scores': final_node_scores.tolist(),
                    'prediction': torch.softmax(logits, dim=1).cpu().tolist()[0],
                    'selection_ratio': len(core_node_indices) / max(1, num_nodes)
                }

            return core_graph

    def _build_core_subgraph(self, original_graph: GeoData, core_node_indices: torch.Tensor) -> GeoData:
        """Build core subgraph"""
        device = original_graph.x.device

        # Create node mapping
        node_mask = torch.zeros(original_graph.x.size(0), dtype=torch.bool, device=device)
        node_mask[core_node_indices] = True

        # Extract core node features
        core_x = original_graph.x[core_node_indices]

        # Process edges: keep only edges where both endpoints are in core nodes
        if original_graph.edge_index.size(1) > 0:
            edge_mask = node_mask[original_graph.edge_index[0]] & node_mask[original_graph.edge_index[1]]
            core_edge_index = original_graph.edge_index[:, edge_mask]

            # Remap node indices
            if core_edge_index.size(1) > 0:
                node_mapping = torch.full((original_graph.x.size(0),), -1, dtype=torch.long, device=device)
                node_mapping[core_node_indices] = torch.arange(len(core_node_indices), device=device)
                core_edge_index = node_mapping[core_edge_index]

                # Edge attributes
                if original_graph.edge_attr is not None:
                    core_edge_attr = original_graph.edge_attr[edge_mask]
                else:
                    core_edge_attr = None
            else:
                core_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                core_edge_attr = None
        else:
            core_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            core_edge_attr = None

        return GeoData(x=core_x, edge_index=core_edge_index, edge_attr=core_edge_attr)

    def get_core_embedding(self, core_graph: GeoData) -> torch.Tensor:
        """Get core subgraph embedding vector"""
        with torch.no_grad():
            core_graph = core_graph.to(self.config.device)

            # Check if graph is empty
            if core_graph.x.size(0) == 0:
                logger.warning("Core subgraph is empty, returning zero vector")
                return torch.zeros(384, device='cpu')  # 128 * 3

            try:
                # Create correct batch for single graph
                core_graph.batch = torch.zeros(core_graph.x.size(0),
                                               dtype=torch.long,
                                               device=core_graph.x.device)

                # Get node embeddings
                node_emb = self.model.gnn(
                    core_graph.x, core_graph.edge_index,
                    core_graph.edge_attr, core_graph.batch
                )

                # Use same pooling strategy as model
                graph_emb_mean = global_mean_pool(node_emb, core_graph.batch)
                graph_emb_max = global_max_pool(node_emb, core_graph.batch)

                # Correct attention-weighted pooling
                att_scores = self.model.graph_attention(node_emb)
                # For single graph (all batch are 0), directly softmax all nodes
                att_weights = F.softmax(att_scores, dim=0)
                graph_emb_att = (node_emb * att_weights).sum(dim=0, keepdim=True)

                # Combine final embedding (exactly consistent with training)
                final_embedding = torch.cat([graph_emb_mean, graph_emb_max, graph_emb_att], dim=1)

                return final_embedding.squeeze().cpu()

            except Exception as e:
                logger.error(f"Error computing core embedding: {e}")
                # Fallback: use simple node embedding average
                try:
                    node_emb = self.model.gnn(
                        core_graph.x, core_graph.edge_index,
                        core_graph.edge_attr, core_graph.batch if hasattr(core_graph, 'batch') else None
                    )
                    if node_emb.size(0) > 0:
                        # Simple average, then expand to correct dimension
                        simple_emb = node_emb.mean(dim=0)
                        # Replicate 3 times to match expected dimension
                        final_emb = torch.cat([simple_emb, simple_emb, simple_emb], dim=0)
                        return final_emb.cpu()
                    else:
                        return torch.zeros(384, device='cpu')
                except Exception as e2:
                    logger.error(f"Fallback method also failed: {e2}")
                    return torch.zeros(384, device='cpu')


class PrimeVulInferencePipeline:
    """Main inference pipeline"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._setup_directories()
        self._load_pretrained_model()
        self.feature_extractor = CoreFeatureExtractor(self.model, config)

    def _setup_directories(self):
        """Create output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.analysis_dir, exist_ok=True)

    def _load_pretrained_model(self):
        """Load pretrained model"""
        if not os.path.exists(self.config.pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model file does not exist: {self.config.pretrained_model_path}")

        logger.info(f"Loading pretrained model: {self.config.pretrained_model_path}")

        try:
            # Allow unsafe loading but only for trusted sources
            checkpoint = torch.load(self.config.pretrained_model_path,
                                    map_location=self.config.device,
                                    weights_only=False)  # Allow loading Config objects

            # Create compatible Config object for model initialization
            model_config = Config()

            # Initialize model
            self.model = GNNOnlyModel(model_config).to(self.config.device)

            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Successfully loaded model weights (from checkpoint)")
            else:
                # If directly saved as state dict
                self.model.load_state_dict(checkpoint)
                logger.info("Successfully loaded model weights (direct state dict)")

            self.model.eval()

            # Verify model loading
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try alternative loading method
            try:
                logger.info("Trying alternative loading method...")
                # Add safe global variables first
                torch.serialization.add_safe_globals([Config])

                state_dict = torch.load(self.config.pretrained_model_path,
                                        map_location=self.config.device,
                                        weights_only=True)

                model_config = Config()
                self.model = GNNOnlyModel(model_config).to(self.config.device)

                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                else:
                    self.model.load_state_dict(state_dict)

                self.model.eval()
                logger.info("Alternative method succeeded")

            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                raise RuntimeError(f"Cannot load pretrained model: {e}, {e2}")

    def load_dataset(self, dataset_name: str) -> GraphSONDataset:
        """Load specified dataset"""
        if dataset_name not in self.config.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_dir = self.config.cpg_base_dir / self.config.datasets[dataset_name]
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        # Get all GraphSON files (usually .json or .graphson extension)
        graphson_files = list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.graphson"))
        logger.info(f"Found {len(graphson_files)} GraphSON files in {dataset_dir}")

        return GraphSONDataset(graphson_files, self.config)

    def process_dataset(self, dataset_name: str, save_results: bool = True) -> Dict:
        """Process entire dataset"""
        logger.info(f"Starting to process dataset: {dataset_name}")

        # Load dataset
        dataset = self.load_dataset(dataset_name)

        # Return empty result if no data
        if len(dataset) == 0:
            logger.warning(f"Dataset {dataset_name} has no valid data, skipping processing")
            return {
                'dataset_name': dataset_name,
                'total_graphs': 0,
                'core_embeddings': [],
                'analysis_details': [],
                'statistics': {
                    'avg_core_nodes': [],
                    'avg_original_nodes': [],
                    'compression_ratios': [],
                    'predictions': []
                },
                'final_statistics': {
                    'total_samples': 0,
                    'avg_original_nodes': 0,
                    'avg_core_nodes': 0,
                    'avg_compression_ratio': 0,
                    'std_compression_ratio': 0,
                    'avg_vuln_probability': 0
                }
            }

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        results = {
            'dataset_name': dataset_name,
            'total_graphs': len(dataset),
            'core_embeddings': [],
            'analysis_details': [],
            'statistics': {
                'avg_core_nodes': [],
                'avg_original_nodes': [],
                'compression_ratios': [],
                'predictions': []
            }
        }

        logger.info(f"Starting to extract core features from {len(dataset)} subgraphs...")

        for idx, (graphs, items) in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
            try:
                # Single graph processing
                graph = graphs[0]
                item_info = items[0]

                # Get true label (using file index)
                true_label = self.feature_extractor.label_loader.get_label_by_idx(
                    dataset_name, item_info['file_idx']
                )

                if true_label is not None:
                    # Found label, update statistics
                    self.feature_extractor.quality_stats['label_match_count'] += 1

                    # Use adaptive core feature extraction with label validation
                    if self.config.enable_quality_validation:
                        core_graph, details = self.feature_extractor.adaptive_core_extraction_with_labels(
                            graph, true_label, dataset_name, return_details=True
                        )
                    else:
                        core_graph, details = self.feature_extractor.extract_attention_based_core(
                            graph, return_details=True
                        )
                else:
                    # Label not found, update statistics
                    self.feature_extractor.quality_stats['label_mismatch_count'] += 1
                    logger.debug(f"Label not found for file {item_info['file_idx']}, skipping quality validation")

                    # Use normal method to extract core features
                    core_graph, details = self.feature_extractor.extract_attention_based_core(
                        graph, return_details=True
                    )

                # Get core embedding
                core_embedding = self.feature_extractor.get_core_embedding(core_graph)

                # Save results
                result_item = {
                    'file_name': item_info['file_name'],
                    'file_idx': item_info['file_idx'],  # File index
                    'dataset_type': item_info['dataset_type'],
                    'core_embedding': core_embedding.numpy().tolist(),
                    'true_label': true_label,  # True label (may be None)
                    'analysis': {
                        'original_nodes': details['original_nodes'],
                        'core_nodes': details['core_nodes'],
                        'core_node_indices': details['core_node_indices'],
                        'node_importance': details['node_importance'],
                        'edge_attention': details['edge_attention'],
                        'prediction': details['prediction'],
                        'quality_metrics': details.get('quality_metrics', {})  # Quality metrics
                    }
                }

                results['core_embeddings'].append({
                    'file_name': item_info['file_name'],
                    'file_idx': item_info['file_idx'],
                    'embedding': core_embedding.numpy()
                })

                results['analysis_details'].append(result_item)

                # Statistics
                results['statistics']['avg_original_nodes'].append(details['original_nodes'])
                results['statistics']['avg_core_nodes'].append(details['core_nodes'])
                results['statistics']['compression_ratios'].append(
                    details['core_nodes'] / details['original_nodes']
                )
                results['statistics']['predictions'].append(details['prediction'])

                # Periodic saving (every 100 samples)
                if save_results and (idx + 1) % 100 == 0:
                    self._save_intermediate_results(results, dataset_name, idx + 1)

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                import traceback
                logger.error(f"Detailed error info: {traceback.format_exc()}")
                continue

        # Calculate final statistics
        stats = results['statistics']
        if len(stats['avg_original_nodes']) > 0:
            final_stats = {
                'total_samples': len(stats['avg_original_nodes']),
                'avg_original_nodes': np.mean(stats['avg_original_nodes']),
                'avg_core_nodes': np.mean(stats['avg_core_nodes']),
                'avg_compression_ratio': np.mean(stats['compression_ratios']),
                'std_compression_ratio': np.std(stats['compression_ratios']),
                'avg_vuln_probability': np.mean([p[1] for p in stats['predictions']])
                # Assume index 1 is vulnerability probability
            }

            # Add quality validation statistics
            if self.config.enable_quality_validation:
                quality_summary = self.feature_extractor.get_quality_summary()
                final_stats['quality_validation'] = quality_summary
        else:
            final_stats = {
                'total_samples': 0,
                'avg_original_nodes': 0,
                'avg_core_nodes': 0,
                'avg_compression_ratio': 0,
                'std_compression_ratio': 0,
                'avg_vuln_probability': 0
            }

        results['final_statistics'] = final_stats

        # Output processing results
        logger.info(f"Dataset {dataset_name} processing completed:")
        logger.info(f"  - Total samples: {final_stats['total_samples']}")
        logger.info(f"  - Average original nodes: {final_stats['avg_original_nodes']:.1f}")
        logger.info(f"  - Average core nodes: {final_stats['avg_core_nodes']:.1f}")
        logger.info(f"  - Average compression ratio: {final_stats['avg_compression_ratio']:.3f}")
        logger.info(f"  - Average vulnerability probability: {final_stats['avg_vuln_probability']:.3f}")

        # Output quality validation information
        if self.config.enable_quality_validation and 'quality_validation' in final_stats:
            quality = final_stats['quality_validation']
            logger.info(f"  - Label match rate: {quality['label_statistics']['match_rate']:.1f}% "
                        f"({quality['label_statistics']['matched_labels']}/{quality['label_statistics']['matched_labels'] + quality['label_statistics']['unmatched_labels']})")

            if 'quality_distribution' in quality:
                logger.info(f"  - Quality distribution:")
                logger.info(f"    * High quality: {quality['quality_distribution']['high_quality']['count']} "
                            f"({quality['quality_distribution']['high_quality']['percentage']:.1f}%)")
                logger.info(f"    * Medium quality: {quality['quality_distribution']['medium_quality']['count']} "
                            f"({quality['quality_distribution']['medium_quality']['percentage']:.1f}%)")
                logger.info(f"    * Low quality: {quality['quality_distribution']['low_quality']['count']} "
                            f"({quality['quality_distribution']['low_quality']['percentage']:.1f}%)")
                logger.info(
                    f"  - Average prediction preservation: {quality['average_metrics']['prediction_preservation']:.3f}")
                logger.info(
                    f"  - Average embedding similarity: {quality['average_metrics']['embedding_similarity']:.3f}")
                logger.info(f"  - Average label consistency: {quality['average_metrics']['label_consistency']:.3f}")
                logger.info(
                    f"  - Average accuracy preservation: {quality['average_metrics']['accuracy_preservation']:.3f}")

        # Save final results
        if save_results:
            self._save_final_results(results, dataset_name)

        return results

    def _save_intermediate_results(self, results: Dict, dataset_name: str, processed_count: int):
        """Save intermediate results"""
        filename = f"{dataset_name}_intermediate_{processed_count}.pkl"
        filepath = Path(self.config.output_dir) / filename

        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"Intermediate results saved: {filepath}")

    def _save_final_results(self, results: Dict, dataset_name: str):
        """Save final results"""
        # Save complete results
        results_file = Path(self.config.output_dir) / f"{dataset_name}_complete_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        # Save core embedding vectors (for downstream tasks)
        embeddings_file = Path(self.config.output_dir) / f"{dataset_name}_core_embeddings.pkl"
        embeddings_data = {
            'embeddings': np.array([item['embedding'] for item in results['core_embeddings']]),
            'file_names': [item['file_name'] for item in results['core_embeddings']],
            'file_indices': [item['file_idx'] for item in results['core_embeddings']],  # File indices
            'dataset_name': dataset_name
        }

        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)

        # Save statistical analysis
        analysis_file = Path(self.config.analysis_dir) / f"{dataset_name}_analysis.json"
        analysis_data = {
            'dataset_name': dataset_name,
            'final_statistics': results['final_statistics'],
            'sample_details': results['analysis_details'][:10]  # Only save first 10 samples' detailed info
        }

        # Ensure data is JSON-safe
        safe_analysis_data = make_json_safe(analysis_data)

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(safe_analysis_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved:")
        logger.info(f"  - Complete results: {results_file}")
        logger.info(f"  - Embedding vectors: {embeddings_file}")
        logger.info(f"  - Analysis report: {analysis_file}")

    def generate_visualization_report(self, dataset_name: str):
        """Generate visualization report"""
        # Load results
        results_file = Path(self.config.output_dir) / f"{dataset_name}_complete_results.pkl"
        if not results_file.exists():
            logger.error(f"Results file does not exist: {results_file}")
            return

        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # Check if there's valid data
        if not results['statistics']['avg_original_nodes']:
            logger.warning(f"Dataset {dataset_name} has no valid statistical data, skipping visualization")
            return

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'PrimeVul {dataset_name} Dataset Analysis', fontsize=16)

        stats = results['statistics']

        # 1. Node count distribution
        axes[0, 0].hist(stats['avg_original_nodes'], bins=30, alpha=0.7, label='Original', color='blue')
        axes[0, 0].hist(stats['avg_core_nodes'], bins=30, alpha=0.7, label='Core', color='red')
        axes[0, 0].set_xlabel('Number of Nodes')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Node Count Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Compression ratio distribution
        axes[0, 1].hist(stats['compression_ratios'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Compression Ratio (Core/Original)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Compression Ratio Distribution')
        axes[0, 1].axvline(np.mean(stats['compression_ratios']), color='red', linestyle='--',
                           label=f'Mean: {np.mean(stats["compression_ratios"]):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Vulnerability probability distribution
        vuln_probs = [p[1] for p in stats['predictions']]
        axes[1, 0].hist(vuln_probs, bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Vulnerability Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Vulnerability Probability Distribution')
        axes[1, 0].axvline(np.mean(vuln_probs), color='red', linestyle='--',
                           label=f'Mean: {np.mean(vuln_probs):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Core nodes vs original nodes scatter plot
        axes[1, 1].scatter(stats['avg_original_nodes'], stats['avg_core_nodes'],
                           alpha=0.6, color='purple', s=20)
        axes[1, 1].plot([0, max(stats['avg_original_nodes'])], [0, max(stats['avg_original_nodes'])],
                        'r--', alpha=0.8, label='y=x')
        axes[1, 1].set_xlabel('Original Nodes')
        axes[1, 1].set_ylabel('Core Nodes')
        axes[1, 1].set_title('Core vs Original Nodes')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        viz_file = Path(self.config.analysis_dir) / f"{dataset_name}_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization report saved: {viz_file}")

    def generate_quality_analysis_report(self, dataset_name: str):
        """Generate quality analysis report"""
        if not self.config.enable_quality_validation:
            return

        # Load results
        results_file = Path(self.config.output_dir) / f"{dataset_name}_complete_results.pkl"
        if not results_file.exists():
            logger.error(f"Results file does not exist: {results_file}")
            return

        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # Extract quality metrics
        quality_scores = []
        prediction_preservations = []
        embedding_similarities = []
        compression_efficiencies = []
        label_consistencies = []
        accuracy_preservations = []

        for detail in results['analysis_details']:
            quality_metrics = detail['analysis'].get('quality_metrics', {})
            if quality_metrics:
                quality_scores.append(quality_metrics.get('quality_score', 0))
                prediction_preservations.append(quality_metrics.get('prediction_preservation', 0))
                embedding_similarities.append(quality_metrics.get('embedding_similarity', 0))
                compression_efficiencies.append(quality_metrics.get('compression_efficiency', 0))
                label_consistencies.append(quality_metrics.get('label_consistency', 0))
                accuracy_preservations.append(quality_metrics.get('accuracy_preservation', 0))

        if not quality_scores:
            logger.warning(f"Dataset {dataset_name} has no quality evaluation data")
            return

        # Create quality analysis visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle(f'Quality Analysis Report - {dataset_name}', fontsize=16)

        # Quality score distribution
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(quality_scores):.3f}')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Quality Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Prediction preservation distribution
        axes[0, 1].hist(prediction_preservations, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(prediction_preservations), color='red', linestyle='--',
                           label=f'Mean: {np.mean(prediction_preservations):.3f}')
        axes[0, 1].set_xlabel('Prediction Preservation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Preservation Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Embedding similarity distribution
        axes[0, 2].hist(embedding_similarities, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(np.mean(embedding_similarities), color='red', linestyle='--',
                           label=f'Mean: {np.mean(embedding_similarities):.3f}')
        axes[0, 2].set_xlabel('Embedding Similarity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Embedding Similarity Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Label consistency distribution
        axes[0, 3].hist(label_consistencies, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 3].axvline(np.mean(label_consistencies), color='red', linestyle='--',
                           label=f'Mean: {np.mean(label_consistencies):.3f}')
        axes[0, 3].set_xlabel('Label Consistency')
        axes[0, 3].set_ylabel('Frequency')
        axes[0, 3].set_title('Label Consistency Distribution')
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)

        # Quality vs compression ratio scatter plot
        compression_ratios = results['statistics']['compression_ratios'][:len(quality_scores)]
        axes[1, 0].scatter(compression_ratios, quality_scores, alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('Compression Ratio')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_title('Quality vs Compression Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Accuracy preservation distribution
        axes[1, 1].hist(accuracy_preservations, bins=10, alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 1].axvline(np.mean(accuracy_preservations), color='red', linestyle='--',
                           label=f'Mean: {np.mean(accuracy_preservations):.3f}')
        axes[1, 1].set_xlabel('Accuracy Preservation')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Accuracy Preservation Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Quality level pie chart
        quality_levels = {'high': 0, 'medium': 0, 'low': 0}
        for score in quality_scores:
            if score >= 0.8:
                quality_levels['high'] += 1
            elif score >= 0.6:
                quality_levels['medium'] += 1
            else:
                quality_levels['low'] += 1

        axes[1, 2].pie(quality_levels.values(), labels=quality_levels.keys(), autopct='%1.1f%%',
                       colors=['green', 'yellow', 'red'], startangle=90)
        axes[1, 2].set_title('Quality Level Distribution')

        # Metrics correlation heatmap
        correlation_data = np.array([
            quality_scores, prediction_preservations, embedding_similarities,
            compression_efficiencies[:len(quality_scores)], label_consistencies,
            accuracy_preservations
        ]).T

        correlation_df = pd.DataFrame(correlation_data,
                                      columns=['Quality', 'Pred_Preservation', 'Emb_Similarity',
                                               'Comp_Efficiency', 'Label_Consistency', 'Acc_Preservation'])

        sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[1, 3])
        axes[1, 3].set_title('Quality Metrics Correlation')

        plt.tight_layout()

        # Save quality analysis report
        quality_viz_file = Path(self.config.analysis_dir) / f"{dataset_name}_quality_analysis.png"
        plt.savefig(quality_viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Quality analysis report saved: {quality_viz_file}")

        # Save detailed quality statistics
        quality_stats = {
            'dataset_name': dataset_name,
            'quality_summary': {
                'mean_quality_score': np.mean(quality_scores),
                'std_quality_score': np.std(quality_scores),
                'mean_prediction_preservation': np.mean(prediction_preservations),
                'std_prediction_preservation': np.std(prediction_preservations),
                'mean_embedding_similarity': np.mean(embedding_similarities),
                'std_embedding_similarity': np.std(embedding_similarities),
                'mean_label_consistency': np.mean(label_consistencies),
                'std_label_consistency': np.std(label_consistencies),
                'mean_accuracy_preservation': np.mean(accuracy_preservations),
                'std_accuracy_preservation': np.std(accuracy_preservations),
                'quality_distribution': quality_levels,
                'high_quality_percentage': quality_levels['high'] / len(quality_scores) * 100,
                'label_consistency_rate': np.mean(label_consistencies) * 100,
                'accuracy_preservation_rate': np.mean(accuracy_preservations) * 100
            }
        }

        quality_stats_file = Path(self.config.analysis_dir) / f"{dataset_name}_quality_stats.json"
        with open(quality_stats_file, 'w') as f:
            json.dump(make_json_safe(quality_stats), f, indent=2)

        return quality_stats

    def process_all_datasets(self):
        """Process all datasets"""
        logger.info("Starting to process all PrimeVul datasets...")

        all_results = {}

        for dataset_name in self.config.datasets.keys():
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing dataset: {dataset_name}")
                logger.info(f"{'=' * 60}")

                results = self.process_dataset(dataset_name, save_results=True)
                all_results[dataset_name] = results

                # Generate visualization report
                self.generate_visualization_report(dataset_name)

                # Generate quality analysis report
                if self.config.enable_quality_validation:
                    self.generate_quality_analysis_report(dataset_name)

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                logger.error(f"Detailed error info: {traceback.format_exc()}")
                continue

        # Generate summary report
        self._generate_summary_report(all_results)

        return all_results

    def _generate_summary_report(self, all_results: Dict):
        """Generate summary report"""
        summary = {
            'total_datasets': len(all_results),
            'dataset_summaries': {},
            'overall_statistics': {
                'total_samples': 0,
                'avg_compression_ratio': [],
                'avg_vuln_probability': []
            }
        }

        for dataset_name, results in all_results.items():
            if 'final_statistics' in results:
                stats = results['final_statistics']
                summary['dataset_summaries'][dataset_name] = stats
                summary['overall_statistics']['total_samples'] += stats['total_samples']
                if stats['total_samples'] > 0:
                    summary['overall_statistics']['avg_compression_ratio'].append(stats['avg_compression_ratio'])
                    summary['overall_statistics']['avg_vuln_probability'].append(stats['avg_vuln_probability'])

        # Calculate overall statistics
        if summary['overall_statistics']['avg_compression_ratio']:
            summary['overall_statistics']['mean_compression_ratio'] = np.mean(
                summary['overall_statistics']['avg_compression_ratio']
            )
            summary['overall_statistics']['mean_vuln_probability'] = np.mean(
                summary['overall_statistics']['avg_vuln_probability']
            )

        # Save summary report
        summary_file = Path(self.config.analysis_dir) / "summary_report.json"
        safe_summary = make_json_safe(summary)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(safe_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'=' * 60}")
        logger.info("Summary Report")
        logger.info(f"{'=' * 60}")
        logger.info(f"Processed datasets: {summary['total_datasets']}")
        logger.info(f"Total samples: {summary['overall_statistics']['total_samples']}")
        if 'mean_compression_ratio' in summary['overall_statistics']:
            logger.info(f"Average compression ratio: {summary['overall_statistics']['mean_compression_ratio']:.3f}")
            logger.info(
                f"Average vulnerability probability: {summary['overall_statistics']['mean_vuln_probability']:.3f}")
        logger.info(f"Summary report saved: {summary_file}")


class CoreFeatureAnalyzer:
    """Core feature analysis tool"""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def compare_datasets(self, dataset1: str, dataset2: str):
        """Compare core features of two datasets"""
        logger.info(f"Comparing datasets: {dataset1} vs {dataset2}")

        # Load embeddings from both datasets
        emb1_file = Path(self.config.output_dir) / f"{dataset1}_core_embeddings.pkl"
        emb2_file = Path(self.config.output_dir) / f"{dataset2}_core_embeddings.pkl"

        if not emb1_file.exists() or not emb2_file.exists():
            logger.error("Missing required embedding files")
            return

        with open(emb1_file, 'rb') as f:
            data1 = pickle.load(f)
        with open(emb2_file, 'rb') as f:
            data2 = pickle.load(f)

        emb1 = data1['embeddings']
        emb2 = data2['embeddings']

        # Check if data is empty
        if len(emb1) == 0 or len(emb2) == 0:
            logger.warning(f"Dataset {dataset1} or {dataset2} is empty, cannot compare")
            return

        # Calculate distance distribution
        try:
            from scipy.spatial.distance import cdist

            # Intra-dataset distances
            intra_dist1 = cdist(emb1, emb1, metric='cosine')
            intra_dist2 = cdist(emb2, emb2, metric='cosine')

            # Inter-dataset distances
            inter_dist = cdist(emb1, emb2, metric='cosine')

            # Statistical analysis
            comparison = {
                'dataset1': dataset1,
                'dataset2': dataset2,
                'intra_distance_1': {
                    'mean': np.mean(intra_dist1[intra_dist1 > 0]),
                    'std': np.std(intra_dist1[intra_dist1 > 0])
                },
                'intra_distance_2': {
                    'mean': np.mean(intra_dist2[intra_dist2 > 0]),
                    'std': np.std(intra_dist2[intra_dist2 > 0])
                },
                'inter_distance': {
                    'mean': np.mean(inter_dist),
                    'std': np.std(inter_dist)
                }
            }

            # Save comparison results
            comp_file = Path(self.config.analysis_dir) / f"comparison_{dataset1}_vs_{dataset2}.json"
            safe_comparison = make_json_safe(comparison)
            with open(comp_file, 'w') as f:
                json.dump(safe_comparison, f, indent=2)

            logger.info(f"Dataset comparison completed, results saved to: {comp_file}")
            return comparison

        except ImportError:
            logger.error("scipy library required for distance calculation")
            return None
        except Exception as e:
            logger.error(f"Error comparing datasets: {e}")
            return None

    def find_similar_graphs(self, dataset_name: str, query_file: str, top_k: int = 10):
        """Find most similar graphs to query graph in dataset"""
        logger.info(f"Searching for graphs similar to {query_file} in {dataset_name}")

        emb_file = Path(self.config.output_dir) / f"{dataset_name}_core_embeddings.pkl"
        if not emb_file.exists():
            logger.error(f"Embedding file does not exist: {emb_file}")
            return

        with open(emb_file, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        file_names = data['file_names']

        if len(embeddings) == 0:
            logger.warning(f"Dataset {dataset_name} is empty")
            return

        # Find query file index
        query_idx = None
        for i, name in enumerate(file_names):
            if query_file in name:
                query_idx = i
                break

        if query_idx is None:
            logger.error(f"Query file {query_file} not found")
            return

        try:
            # Calculate similarity
            from scipy.spatial.distance import cosine

            query_emb = embeddings[query_idx]
            similarities = []

            for i, emb in enumerate(embeddings):
                if i != query_idx:
                    sim = 1 - cosine(query_emb, emb)
                    similarities.append((i, file_names[i], sim))

            # Sort and get top-k
            similarities.sort(key=lambda x: x[2], reverse=True)
            top_similar = similarities[:top_k]

            result = {
                'query_file': query_file,
                'dataset': dataset_name,
                'top_similar': [
                    {'rank': i + 1, 'file_name': item[1], 'similarity': item[2]}
                    for i, item in enumerate(top_similar)
                ]
            }

            # Save results
            sim_file = Path(self.config.analysis_dir) / f"similarity_search_{query_file}_{dataset_name}.json"
            safe_result = make_json_safe(result)
            with open(sim_file, 'w') as f:
                json.dump(safe_result, f, indent=2)

            logger.info(f"Similarity search completed, results saved to: {sim_file}")
            return result

        except ImportError:
            logger.error("scipy library required for similarity calculation")
            return None
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return None


def main():
    """Main function"""
    # Set environment variables (if needed)
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    })

    # Initialize configuration
    config = InferenceConfig()

    logger.info("=" * 80)
    logger.info("PrimeVul GNN Inference & Core Feature Extraction System - GraphSON Format Optimized")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Pretrained model: {config.pretrained_model_path}")
    logger.info(f"GraphSON data directory: {config.cpg_base_dir}")
    logger.info(f"Label data directory: {config.primevul_labels_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Quality validation: {'Enabled' if config.enable_quality_validation else 'Disabled'}")

    # Initialize inference pipeline
    try:
        pipeline = PrimeVulInferencePipeline(config)
        logger.info("Inference pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Inference pipeline initialization failed: {e}")
        import traceback
        logger.error(f"Detailed error info: {traceback.format_exc()}")
        return

    # Process all datasets
    try:
        all_results = pipeline.process_all_datasets()
        logger.info("All datasets processed successfully")

        # Perform dataset comparison analysis
        analyzer = CoreFeatureAnalyzer(config)

        # Compare training and test sets
        if 'train' in all_results and 'test' in all_results:
            analyzer.compare_datasets('train', 'test')

        # Compare balanced and unbalanced datasets
        if 'train' in all_results and 'train_paired' in all_results:
            analyzer.compare_datasets('train', 'train_paired')

        logger.info("Analysis completed!")
        logger.info("=" * 80)
        logger.info("GraphSON format core feature extraction completed with main fixes:")
        logger.info("1. Fixed GraphSON file format parsing")
        logger.info("2. Enhanced error handling and debugging info")
        logger.info("3. Support for multiple GraphSON formats")
        logger.info("4. Improved file validation logic")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(f"Detailed error info: {traceback.format_exc()}")


if __name__ == "__main__":
    main()