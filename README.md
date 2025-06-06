# LLM-Guided Vulnerability Detection with Proper Context

Implementation code for the paper "LLM-Guided Vulnerability Detection with Proper Context".

## Quick Start

### Installation
```bash
git clone <repository-url>
cd finetuningVD
pip install -r requirements.txt

# Install Joern for CPG analysis
# Download from: https://joern.io/
export JOERN_PATH=/path/to/joern-parse
```

## LLM Configuration

### Supported Models for Stage 1 (Key Node Identification)

**Multi-LLM Consensus (Recommended):**
- **Qwen2.5-Coder**: Optimized for coding tasks from the Qwen series
- **Gemini-2.0-Flash**: Fast multimodal model from Google with strong coding abilities  
- **DeepSeek-V3**: Open-source Mixture-of-Experts (MoE) code language model

**Consensus Mechanism:**
- Uses voting-based agreement across multiple LLMs
- Configurable agreement threshold (default: 0.7)
- Reduces bias and improves critical node identification accuracy

### Supported Models for Stage 4 (Feature Fusion)

**LLM Integration Options:**
- **DeepSeek-R1-Distill-Qwen-32B**: Combines DeepSeek-R1 reasoning with Qwen architecture
- **CodeLlama-13B**: Baseline model for comparison and alternative configuration

## Usage Pipeline

### Stage 1: LLM-based Key Node Identification

Our approach uses a three-round Chain-of-Thought (CoT) prompting strategy for systematic vulnerability analysis:

**Round 1: Functional Overview**
```
I. Functional Overview:
Briefly describe the main purpose and core behavior of the following function 
(e.g., key computations, state changes, external interactions):

[code_function]
```

**Round 2: Security Analysis and Critical Node Annotation**
```
II. Security Analysis and Critical Node Annotation:
Based on the function's purpose, proceed to deeply analyze its security characteristics, 
identify potential vulnerability patterns, and annotate the fine-grained critical nodes 
that contribute most significantly to its security posture.

When identifying these nodes, pay close attention to: dangerous API calls, sensitive 
data operations (arrays, pointers, etc.), critical arithmetic operations, important 
control logic affecting security, handling of external inputs, and protections at 
trust boundaries.

For each identified critical node, provide:
**Type:** Select from V_ (Variable/Data), F_ (Function/Operation), P_ (Parameter), 
C_ (Control/Condition), SM_ (Security Mechanism), M_ (Missing Element).
**Name:** the node name in the code.
**Code Reference:** The relevant variable/function name in the code.
**Line Numbers:** Start and end line numbers.
```

**Round 3: JSON Output with Confidence Scores**
```
III. Node Confidence and Rationale:
For each identified critical node from the previous analysis, provide its confidence 
score and annotation rationale in the following JSON format:

{
  "critical_nodes": [
    {
      "type": "...",
      "name": "...", 
      "code_reference": "...",
      "line_start": 0,
      "line_end": 0,
      "confidence": 0.0,
      "rationale": "..."
    }
  ]
}

Where confidence (0.0-1.0) represents the LLM's belief in: 1) The identified code 
element is indeed a critical node relevant to software security; 2) The assigned 
type is accurate; 3) The rationale adequately explains the node's security criticality.
```

**Implementation:**

```bash
cd src/step1_multi_llm

# Multi-LLM consensus (recommended)
python triple_consensus_analyzer.py \
    --deepseek-key "your_deepseek_key" \
    --gemini-key "your_gemini_key" \
    --qwen-key "your_qwen_key" \
    --input-data "../../data/raw/input.jsonl" \
    --output-data "../../data/primevul_consensus_key_node/output.jsonl" \
    --agreement-threshold 0.7 \
    --limit 10

# Single LLM analysis
python single_llm.py \
    --api-key "your_key" \
    --input-data "../../data/raw/input.jsonl" \
    --output-data "../../data/primevul_consensus_key_node/single_llm.jsonl" \
    --max-retries 5 \
    --retry-delay 15
```

### Stage 2: CPG Extraction and Subgraph Slicing

```bash
cd src/step2_subgraph

# CPG extraction
python cpg_extractor.py \
    --input-dir "../../data/raw/" \
    --output-dir "../../data/primevul_cpg_bin_files/" \
    --joern-path /usr/local/bin/joern-parse \
    --memory 32G \
    --processes 8

# Subgraph slicing
python joern_slice.py \
    --cpg-bin-dir "../../data/primevul_cpg_bin_files/" \
    --input-jsonl "../../data/primevul_consensus_key_node/output.jsonl" \
    --output-dir "../../data/primevul_subgraph/" \
    --slice-depth 5 \
    --memory "20G"
```

### Stage 3: GNN-based Feature Extraction

```bash
cd src/step3_gnn

# Pretraining
python pretraining.py

# GAT optimization
python gat_optimization.py
```

### Stage 4: Feature Fusion and Classification

```bash
cd src/step4_fusion

# Joint fine-tuning with CodeLlama
python joint_finetuning_gnn_embedding_codellama.py

# Joint fine-tuning with DeepSeek-R1
python joint_finetuning_gnn_embedding_deepseek_r1.py
```

## Data Processing

### PrimeVul Dataset Processing
```bash
cd src/data

# Sample PrimeVul dataset
python primevul_sampler.py \
    --input-dir ../../data/raw/PrimeVul \
    --output-dir ../../data/primevul_process/

# Process for consensus key nodes
python process_primevul.py
```

### CVEFixes Dataset Processing
```bash
cd src/data
python process_data.py
```

## Baseline Methods

### Running Individual Baselines
```bash
# Devign
cd experiments/baselines/devign
python main.py -c -e -p

# SySeVR
cd experiments/baselines/sysevr
python main.py

# GNN-ReGVD
cd experiments/baselines/GNN-ReGVD
python run.py \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train --do_eval --do_test \
    --train_data_file=../../../data/primevul_process/train.jsonl \
    --eval_data_file=../../../data/primevul_process/valid.jsonl \
    --test_data_file=../../../data/primevul_process/test.jsonl
```

### Other Transformer Baselines
```bash
cd src/baseline

# CodeBERT
python codebert.py

# UnixCoder  
python unixcoder.py

# GraphCodeBERT
python graphbert.py
```

## Experiments

### RQ2 Context Refinement Analysis
```bash
cd experiments/rq2_comparisons
python compare_full_vs_subgraph.py
```

### Ablation Studies
```bash
cd experiments/ablation_studies

# Full model
python run_full.py

# Feature fusion analysis
python run_fusion.py

# No embedding baseline
python run_no_embedding.py

# Single LLM analysis
python run_single.py
```

## Data Format

**Input JSONL Format** (stored in `data/raw/`)
```json
{
    "idx": 0,
    "project": "openssl", 
    "commit_id": "ca989269a2876bae79393bd54c3e72d49975fc75",
    "target": 1,
    "func": "long ssl_get_algorithm2(SSL *s) { ... }",
    "cwe": ["CWE-310"],
    "cve": "CVE-2013-6449"
}
```

**Processed Data Locations:**
- Raw data: `data/raw/`
- Consensus key nodes: `data/primevul_consensus_key_node/`
- CPG binaries: `data/primevul_cpg_bin_files/`
- GraphSON files: `data/primevul_graphson_files/`
- Subgraphs: `data/primevul_subgraph/`
- Gemini subgraphs: `data/gemini_subgraph/`

## Project Structure

```
finetuningVD/
├── data/                                    # Data directory
│   ├── raw/                                # Raw input data
│   ├── primevul_consensus_key_node/        # LLM consensus results
│   ├── primevul_cpg_bin_files/            # CPG binary files
│   ├── primevul_graphson_files/           # GraphSON format
│   ├── primevul_subgraph/                 # Extracted subgraphs
│   ├── gemini_subgraph/                   # Gemini-specific subgraphs
│   ├── new_graphson_files/                # Updated GraphSON files
│   ├── primevul_process/                  # Processed PrimeVul data
│   └── subgraphs/                         # Additional subgraphs
├── experiments/                           # Experimental analysis
│   ├── rq2_comparisons/                   # RQ2 analysis
│   ├── ablation_studies/                  # Ablation study experiments
│   │   ├── full/                          # Full model results
│   │   ├── fusion/                        # Feature fusion analysis
│   │   ├── no_embedding/                  # No embedding baseline
│   │   └── single/                        # Single LLM results
│   └── baselines/                         # Baseline implementations
│       ├── devign/                        # Devign baseline
│       ├── GNN-ReGVD/                     # ReGVD baseline
│       └── sysevr/                        # SySeVR baseline
├── model/                                 # Pre-trained models
│   ├── CodeLlama-13b-hf/                  # CodeLlama model
│   └── DeepSeek-R1-Distill-Qwen-32B/     # DeepSeek model
├── result/                                # Experimental results
├── src/                                   # Source code
│   ├── data/                              # Data processing scripts
│   ├── baseline/                          # Baseline method implementations
│   ├── step1_multi_llm/                   # Stage 1: LLM consensus
│   ├── step2_subgraph/                    # Stage 2: Subgraph extraction
│   ├── step3_gnn/                         # Stage 3: GNN processing
│   └── step4_fusion/                      # Stage 4: Feature fusion
└── requirements.txt
```

## Model Files

The project uses pre-trained models stored in `model/`:
- **CodeLlama-13b-hf/**: CodeLlama model for code understanding
- **DeepSeek-R1-Distill-Qwen-32B/**: DeepSeek-R1 distilled model

## Evaluation

```bash
# Run evaluation on test results
python src/evaluator/evaluator.py \
    -a data/primevul_process/test.jsonl \
    -p result/predictions.txt
```

**Prediction Format:**
```
idx	prediction	confidence
0	0	0.450
1	1	0.900
2	1	0.501
```

## Requirements

- Python 3.8+
- PyTorch 2.5+
- Transformers 4.51+
- Joern (for static analysis)
- GPU recommended for training
- API keys for LLM services (DeepSeek, Gemini, Qwen)

## Quick Reproduction

To reproduce the main results:

1. **Prepare data:**
   ```bash
   cd src/data
   python primevul_sampler.py --input-dir ../../data/raw/PrimeVul --output-dir ../../data/primevul_process/
   ```

2. **Run full pipeline:**
   ```bash
   # Stage 1: LLM consensus
   cd src/step1_multi_llm
   python triple_consensus_analyzer.py --deepseek-key "key" --gemini-key "key" --qwen-key "key" --input-data "../../data/raw/input.jsonl" --output-data "../../data/primevul_consensus_key_node/output.jsonl"
   
   # Stage 2: Subgraph extraction
   cd ../step2_subgraph
   python cpg_extractor.py --input-dir "../../data/raw/" --output-dir "../../data/primevul_cpg_bin_files/"
   python joern_slice.py --cpg-bin-dir "../../data/primevul_cpg_bin_files/" --input-jsonl "../../data/primevul_consensus_key_node/output.jsonl" --output-dir "../../data/primevul_subgraph/"
   
   # Stage 3: GNN training
   cd ../step3_gnn
   python pretraining.py
   python gat_optimization.py
   
   # Stage 4: Joint fine-tuning
   cd ../step4_fusion
   python joint_finetuning_gnn_embedding_codellama.py
   ```

3. **Run baselines for comparison:**
   ```bash
   cd experiments/baselines/devign && python main.py -c -e -p
   cd ../sysevr && python main.py
   cd ../GNN-ReGVD && python run.py --do_train --do_eval --do_test
   ```