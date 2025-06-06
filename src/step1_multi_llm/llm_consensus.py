import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import time
from openai import OpenAI
import google.generativeai as genai
from collections import defaultdict
import statistics
import re

# Constants
DEFAULT_AGREEMENT_THRESHOLD = 0.7
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
NODE_LINE_TOLERANCE = 3
MAX_TOKENS = 2000
TEMPERATURE = 0.1

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_model_security_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BaseSecurityAnalyzer:
    """Base security analyzer for code vulnerability detection"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.max_retries = DEFAULT_MAX_RETRIES
        self.retry_delay = DEFAULT_RETRY_DELAY
        self.system_prompt = """
        You are an expert code security analyst specializing in identifying critical nodes in code that affect security posture. 
        Your role is to perform comprehensive security analysis following a structured approach, identifying fine-grained 
        critical nodes that contribute significantly to software security (either as risk points or protective measures).

        Focus on providing detailed, accurate analysis with high confidence in your assessments.
        """

    def execute_security_analysis(self, code_function: str) -> Optional[Dict[str, Any]]:
        """Execute complete three-round security analysis"""
        try:
            # Round 1: Functional overview
            round1_prompt = "I. Functional Overview:\nBriefly describe the main purpose and core behavior of the following function (e.g., key computations, state changes, external interactions):\n\n" + code_function

            round1_response = self._call_model_with_retry([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": round1_prompt}
            ])

            separator = "=" * 40
            logger.info("\n%s\n%s Round 1 Output:\n%s\n%s", separator, self.model_name, round1_response, separator)

            # Round 2: Security analysis and node annotation
            round2_prompt = """II. Security Analysis and Critical Node Annotation:
Based on the function's purpose, proceed to deeply analyze its security characteristics, identify potential vulnerability patterns, and annotate the fine-grained critical nodes that contribute most significantly to its security posture (either as risk points or protective measures).

When identifying these nodes, pay close attention to: dangerous API calls, sensitive data operations (arrays, pointers, etc., noting boundary, type, and lifecycle issues), critical arithmetic operations, important control logic affecting security (watch for missing checks), handling of external inputs, and protections at trust boundaries. Also, understand the flow of critical data (e.g., inputs, pointer values, operation results) and how it impacts sensitive operations or decisions, to determine if it constitutes potential vulnerability links (trigger, propagation, exploitation) or effective defenses.

For each identified critical node, provide:
**Type:** Select from V_ (Variable/Data), F_ (Function/Operation), P_ (Parameter), C_ (Control/Condition), SM_ (Security Mechanism), M_ (Missing Element).
**Name:** the node name in the code.
**Code Reference:** The relevant variable/function name in the code.
**Line Numbers:** Start and end line numbers."""

            round2_response = self._call_model_with_retry([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": round1_prompt},
                {"role": "assistant", "content": round1_response},
                {"role": "user", "content": round2_prompt}
            ])

            logger.info("\n%s\n%s Round 2 Output:\n%s\n%s", separator, self.model_name, round2_response, separator)

            # Round 3: JSON output with confidence scores
            round3_prompt = """III. Node Confidence and Rationale:
For each identified critical node from the previous analysis, provide its confidence score and annotation rationale in the following JSON format:

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

Where:
- type: Select from V_ (Variable/Data), F_ (Function/Operation), P_ (Parameter), C_ (Control/Condition), SM_ (Security Mechanism), M_ (Missing Element)
- name: the node name in the code
- code_reference: The relevant variable/function name in the code
- line_start/line_end: Start and end line numbers (use 0 if not applicable)
- confidence: Confidence Score (0.0-1.0): This score represents your (LLM's) belief in the following: 1) The identified code element is indeed a "critical node" relevant to software security; 2) The "type" assigned to this node is accurate; 3) The "rationale" provided below adequately and correctly explains the node's security criticality. A higher score indicates greater certainty in this overall annotation.
- rationale: Concisely explain the node's direct link to specific security patterns,data/control flow, or its role in the trigger, propagation, or exploitation phases of potential vulnerabilities, or as a defensive mechanism.

Please output ONLY the JSON, no additional text or explanations."""

            round3_response = self._call_model_with_retry([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": round1_prompt},
                {"role": "assistant", "content": round1_response},
                {"role": "user", "content": round2_prompt},
                {"role": "assistant", "content": round2_response},
                {"role": "user", "content": round3_prompt}
            ])

            logger.info("\n%s\n%s Round 3 Output:\n%s\n%s", separator, self.model_name, round3_response, separator)

            parsed_critical_nodes = self._parse_json_response(round3_response)

            analysis_result = {
                'model_name': self.model_name,
                'functional_analysis': round1_response,
                'security_analysis': round2_response,
                'critical_nodes': parsed_critical_nodes
            }

            return analysis_result

        except Exception as e:
            separator = "=" * 40
            logger.error("\n%s\n%s security analysis failed: %s\n%s", separator, self.model_name, str(e), separator)
            return None

    def execute_cross_validation(self, code_function: str, other_models_nodes: List[Dict[str, Any]]) -> Optional[
        Dict[str, Any]]:
        """Execute cross-validation on nodes from other models only"""
        try:
            if not other_models_nodes:
                logger.info("%s: No nodes from other models to validate", self.model_name)
                return {
                    'model_name': self.model_name,
                    'validation_results': []
                }

            validation_prompt = self._build_validation_prompt(code_function, other_models_nodes)

            response = self._call_model_with_retry([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": validation_prompt}
            ])

            separator = "=" * 40
            logger.info("\n%s\n%s Cross-Validation:\n%s\n%s", separator, self.model_name, response, separator)

            validation_result = self._parse_validation_response(response)

            return {
                'model_name': self.model_name,
                'validation_results': validation_result
            }

        except Exception as e:
            logger.error("%s cross-validation failed: %s", self.model_name, str(e))
            return None

    def _build_validation_prompt(self, code_function: str, other_models_nodes: List[Dict[str, Any]]) -> str:
        """Build cross-validation prompt"""
        if not other_models_nodes:
            return "No nodes to validate."

        nodes_summary = "Other analysts have identified the following critical security nodes:\n\n"

        for i, node_info in enumerate(other_models_nodes, 1):
            node = node_info['node']
            source_model = node_info['source_model']
            nodes_summary += "{}. **Node ID: {}**\n".format(i, i)
            nodes_summary += "   - From: {}\n".format(source_model)
            nodes_summary += "   - Type: {}\n".format(node.get('type', 'N/A'))
            nodes_summary += "   - Name: {}\n".format(node.get('name', 'N/A'))
            nodes_summary += "   - Code Reference: {}\n".format(node.get('code_reference', 'N/A'))
            nodes_summary += "   - Lines: {}-{}\n".format(node.get('line_start', 0), node.get('line_end', 0))
            nodes_summary += "   - Original Confidence: {}\n".format(node.get('confidence', 'N/A'))
            nodes_summary += "   - Rationale: {}\n\n".format(node.get('rationale', 'N/A'))

        validation_prompt = """Cross-Validation Task:

{}

For each node listed above, please evaluate whether you agree with its identification as a security-critical node. 

Respond with JSON format containing ONLY validation results:

{{
  "validations": [
    {{
      "node_id": 1,
      "validation": "AGREE|DISAGREE",
      "confidence": 0.0,
      "reasoning": "Brief explanation of your assessment"
    }}
  ]
}}

Rules:
- Use "AGREE" if you believe the node is indeed security-critical
- Use "DISAGREE" if you believe the node is not security-critical or incorrectly identified
- Provide confidence score (0.0-1.0) for your validation
- Give brief reasoning for your decision

Code for reference:
{}

Output ONLY the JSON, no additional text.""".format(nodes_summary, code_function)

        return validation_prompt

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response with error handling"""
        try:
            response_clean = response.strip()

            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]

            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()
            response_clean = self._fix_json_issues(response_clean)

            parsed_data = json.loads(response_clean)
            return parsed_data.get('critical_nodes', [])

        except json.JSONDecodeError as e:
            logger.warning("%s JSON parsing failed: %s", self.model_name, str(e))
            return self._fallback_json_parse(response)
        except Exception as e:
            logger.warning("%s Unexpected error in JSON parsing: %s", self.model_name, str(e))
            return []

    def _parse_validation_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse validation JSON response"""
        try:
            response_clean = response.strip()

            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]

            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()
            response_clean = self._fix_json_issues(response_clean)

            parsed_data = json.loads(response_clean)
            return parsed_data.get('validations', [])

        except json.JSONDecodeError as e:
            logger.warning("%s Failed to parse validation JSON: %s", self.model_name, str(e))
            return self._fallback_validation_parse(response)
        except Exception as e:
            logger.warning("%s Unexpected error parsing validation response: %s", self.model_name, str(e))
            return []

    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON format issues"""
        if not json_str or not json_str.strip():
            return "{}"

        lines = json_str.split('\n')
        clean_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            clean_lines.append(line)
        json_str = '\n'.join(clean_lines)

        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        json_str = json_str.replace('\\n', ' ').replace('\\t', ' ')
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        return json_str

    def _fallback_json_parse(self, response: str) -> List[Dict[str, Any]]:
        """Fallback JSON parsing method"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_part = response[start_idx:end_idx + 1]
                json_part = self._fix_json_issues(json_part)
                parsed_data = json.loads(json_part)
                return parsed_data.get('critical_nodes', [])
        except Exception:
            pass

        logger.warning("%s All JSON parsing attempts failed", self.model_name)
        return []

    def _fallback_validation_parse(self, response: str) -> List[Dict[str, Any]]:
        """Fallback validation parsing method"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_part = response[start_idx:end_idx + 1]
                json_part = self._fix_json_issues(json_part)
                parsed_data = json.loads(json_part)
                return parsed_data.get('validations', [])
        except Exception:
            pass

        logger.warning("%s All validation parsing attempts failed", self.model_name)
        return []

    def _call_model_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Call model with retry mechanism - to be implemented by subclasses"""
        raise NotImplementedError


class DeepSeekSecurityAnalyzer(BaseSecurityAnalyzer):
    """DeepSeek model security analyzer"""

    def __init__(self, api_key: str):
        super().__init__("DeepSeek")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def _call_model_with_retry(self, messages: List[Dict[str, str]]) -> str:
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning("%s API call failed (attempt %d/%d): %s", self.model_name, retries, self.max_retries,
                               str(e))

                # Exponential backoff for rate limits
                if "rate limit" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.warning("Rate limit detected, waiting %ds before retry", wait_time)
                    time.sleep(wait_time)
                else:
                    time.sleep(self.retry_delay)

        raise Exception(
            "{} API call failed after {} retries: {}".format(self.model_name, self.max_retries, str(last_error)))


class GeminiSecurityAnalyzer(BaseSecurityAnalyzer):
    """Gemini 2.0 Flash model security analyzer"""

    def __init__(self, api_key: str):
        super().__init__("Gemini-2.0-Flash")
        genai.configure(api_key=api_key)

    def _call_model_with_retry(self, messages: List[Dict[str, str]]) -> str:
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Convert messages to single prompt
                prompt_parts = []
                for msg in messages:
                    if msg["role"] == "system":
                        prompt_parts.append("System: " + msg['content'])
                    elif msg["role"] == "user":
                        prompt_parts.append("User: " + msg['content'])
                    elif msg["role"] == "assistant":
                        prompt_parts.append("Assistant: " + msg['content'])

                full_prompt = "\n\n".join(prompt_parts)

                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning("%s API call failed (attempt %d/%d): %s", self.model_name, retries, self.max_retries,
                               str(e))

                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.warning("Rate limit detected, waiting %ds before retry", wait_time)
                    time.sleep(wait_time)
                else:
                    time.sleep(self.retry_delay)

        raise Exception(
            "{} API call failed after {} retries: {}".format(self.model_name, self.max_retries, str(last_error)))


class QwenSecurityAnalyzer(BaseSecurityAnalyzer):
    """Qwen 2.5 Coder 32B model security analyzer"""

    def __init__(self, api_key: str):
        super().__init__("Qwen2.5-Coder-32B")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _call_model_with_retry(self, messages: List[Dict[str, str]]) -> str:
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="qwen2.5-coder-32b-instruct",
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    extra_body={"enable_thinking": False}
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning("%s API call failed (attempt %d/%d): %s", self.model_name, retries, self.max_retries,
                               str(e))

                if "throttling" in str(e).lower() or "rate" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.warning("Rate limit detected, waiting %ds before retry", wait_time)
                    time.sleep(wait_time)
                else:
                    time.sleep(self.retry_delay)

        raise Exception(
            "{} API call failed after {} retries: {}".format(self.model_name, self.max_retries, str(last_error)))


class TripleConsensusCalculator:
    """Triple consensus calculator - each node validated by other 2 models"""

    def __init__(self, agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD):
        self.agreement_threshold = agreement_threshold

    def calculate_triple_consensus(self, individual_results: List[Dict[str, Any]],
                                   validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate triple model consensus"""

        if len(individual_results) != 3:
            error_msg = "Expected exactly 3 individual results, got {}".format(len(individual_results))
            logger.error(error_msg)
            return {'consensus_nodes': [], 'statistics': {'error': error_msg}}

        if len(validation_results) != 3:
            error_msg = "Expected exactly 3 validation results, got {}".format(len(validation_results))
            logger.error(error_msg)
            return {'consensus_nodes': [], 'statistics': {'error': error_msg}}

        logger.info("Starting triple consensus calculation")

        try:
            all_nodes_with_source = self._collect_all_nodes(individual_results)
            logger.info("Collected %d total nodes from individual analysis", len(all_nodes_with_source))

            if not all_nodes_with_source:
                logger.warning("No valid nodes found in individual analysis results")
                return {
                    'consensus_nodes': [],
                    'validation_matrix': {},
                    'statistics': {
                        'total_individual_nodes': 0,
                        'total_unique_nodes': 0,
                        'total_consensus_nodes': 0,
                        'consensus_rate': 0.0,
                        'error': 'No valid nodes found'
                    }
                }

            validation_matrix = self._build_validation_matrix(all_nodes_with_source, validation_results)
            logger.info("Built validation matrix for %d unique nodes", len(validation_matrix))

            consensus_nodes = self._apply_triple_consensus_rules(validation_matrix)
            logger.info("Found %d nodes with triple consensus", len(consensus_nodes))

            statistics = self._generate_statistics(all_nodes_with_source, validation_matrix, consensus_nodes)

            return {
                'consensus_nodes': consensus_nodes,
                'validation_matrix': validation_matrix,
                'statistics': statistics
            }

        except Exception as e:
            error_msg = "Error during consensus calculation: {}".format(str(e))
            logger.error(error_msg)
            logger.error("Exception details:", exc_info=True)
            return {
                'consensus_nodes': [],
                'validation_matrix': {},
                'statistics': {'error': error_msg}
            }

    def _collect_all_nodes(self, individual_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect all nodes from individual analysis"""
        all_nodes = []

        for result in individual_results:
            model_name = result.get('model_name', 'Unknown')
            nodes = result.get('critical_nodes', [])

            logger.info("Processing %d nodes from %s", len(nodes), model_name)

            for i, node in enumerate(nodes):
                try:
                    if self._is_valid_node(node):
                        enhanced_node = {
                            'original_node': node,
                            'source_model': model_name,
                            'node_id': len(all_nodes) + 1,
                            'node_key': self._generate_node_key(node, model_name)
                        }
                        all_nodes.append(enhanced_node)
                        logger.debug("Added valid node %d from %s: %s", i + 1, model_name,
                                     node.get('code_reference', 'N/A'))
                    else:
                        logger.warning("Invalid node from %s: %s", model_name, node)
                        self._diagnose_invalid_node(node, model_name)
                except Exception as e:
                    logger.error("Error processing node %d from %s: %s", i + 1, model_name, str(e))
                    logger.error("Problematic node data: %s", node)

        logger.info("Successfully collected %d valid nodes total", len(all_nodes))
        return all_nodes

    def _diagnose_invalid_node(self, node: Dict[str, Any], model_name: str):
        """Diagnose invalid node issues"""
        issues = []

        required_fields = ['type', 'code_reference', 'confidence']
        for field in required_fields:
            if field not in node or node[field] is None:
                issues.append("Missing required field: {}".format(field))

        confidence = node.get('confidence')
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                issues.append("Confidence is not a number: {} (type: {})".format(confidence, type(confidence)))
            elif not (0.0 <= confidence <= 1.0):
                issues.append("Confidence out of range [0,1]: {}".format(confidence))

        node_type = node.get('type', '')
        if node_type:
            valid_types = ['V_', 'F_', 'P_', 'C_', 'SM_', 'M_']
            type_parts = [t.strip() for t in str(node_type).split(',')]
            invalid_types = [t for t in type_parts if t not in valid_types]
            if invalid_types:
                issues.append("Invalid type parts: {} (valid: {})".format(invalid_types, valid_types))

        if issues:
            logger.warning("Node validation issues for %s: %s", model_name, "; ".join(issues))

    def _is_valid_node(self, node: Dict[str, Any]) -> bool:
        """Validate node data"""
        required_fields = ['type', 'code_reference', 'confidence']

        for field in required_fields:
            if field not in node or node[field] is None:
                return False

        confidence = node.get('confidence')
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False

        node_type = node.get('type', '').strip()
        if not node_type:
            return False

        valid_types = ['V_', 'F_', 'P_', 'C_', 'SM_', 'M_']
        type_parts = [t.strip() for t in node_type.split(',')]
        for type_part in type_parts:
            if type_part not in valid_types:
                return False

        return True

    def _generate_node_key(self, node: Dict[str, Any], source_model: str) -> str:
        """Generate unique node key with improved precision"""
        node_type = node.get('type', '').strip()
        if ',' in node_type:
            primary_type = node_type.split(',')[0].strip()
        else:
            primary_type = node_type

        code_ref = node.get('code_reference', '')
        if code_ref is None:
            code_ref = ''
        code_ref = str(code_ref).strip().lower()
        normalized_ref = re.sub(r'[^\w_(),.\[\]]', '', code_ref)

        name = node.get('name', '')
        if name is None:
            name = ''
        name = str(name).strip().lower()
        normalized_name = re.sub(r'[^\w_]', '', name)

        # Smaller line tolerance for better precision
        line_start = node.get('line_start', 0)
        line_end = node.get('line_end', 0)

        try:
            line_start = int(line_start) if line_start is not None else 0
            line_end = int(line_end) if line_end is not None else 0
        except (ValueError, TypeError):
            line_start = 0
            line_end = 0

        line_group_start = (line_start // NODE_LINE_TOLERANCE) * NODE_LINE_TOLERANCE
        line_group_end = (line_end // NODE_LINE_TOLERANCE) * NODE_LINE_TOLERANCE
        line_range = "{}{}{}".format(line_group_start, "-", line_group_end)

        return "{}:{}:{}:{}:{}".format(source_model, primary_type, normalized_ref, normalized_name, line_range)

    def _build_validation_matrix(self, all_nodes: List[Dict[str, Any]],
                                 validation_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build validation matrix for each node"""
        validation_matrix = {}

        for node_info in all_nodes:
            node_key = node_info['node_key']
            validation_matrix[node_key] = {
                'original_node': node_info['original_node'],
                'source_model': node_info['source_model'],
                'node_id': node_info['node_id'],
                'validations': {},
                'validation_summary': {}
            }

        logger.info("Initialized validation matrix for %d unique node keys", len(validation_matrix))

        node_id_to_info = {info['node_id']: info for info in all_nodes}
        logger.debug("Node ID to info mapping: %d entries", len(node_id_to_info))

        total_validations_processed = 0

        for validation_result in validation_results:
            validator_model = validation_result.get('model_name', 'Unknown')
            validations = validation_result.get('validation_results', [])

            logger.info("Processing %d validations from %s", len(validations), validator_model)

            for validation in validations:
                try:
                    node_id = validation.get('node_id')
                    if node_id is None:
                        logger.warning("Validation missing node_id from %s: %s", validator_model, validation)
                        continue

                    if node_id not in node_id_to_info:
                        logger.warning("Unknown node_id %s from %s", node_id, validator_model)
                        continue

                    node_info = node_id_to_info[node_id]
                    node_key = node_info['node_key']
                    source_model = node_info['source_model']

                    # Prevent self-validation
                    if validator_model == source_model:
                        logger.warning("Self-validation detected: %s validating own node %s, skipping",
                                       validator_model, node_id)
                        continue

                    if node_key not in validation_matrix:
                        logger.warning("Node key %s not found in validation matrix", node_key)
                        continue

                    validation_status = validation.get('validation', 'UNKNOWN')
                    confidence = validation.get('confidence', 0.0)

                    try:
                        confidence = float(confidence)
                        if not (0.0 <= confidence <= 1.0):
                            logger.warning("Invalid confidence %s from %s for node %s, using 0.0",
                                           confidence, validator_model, node_id)
                            confidence = 0.0
                    except (ValueError, TypeError):
                        logger.warning("Non-numeric confidence %s from %s for node %s, using 0.0",
                                       confidence, validator_model, node_id)
                        confidence = 0.0

                    validation_matrix[node_key]['validations'][validator_model] = {
                        'validation': validation_status,
                        'confidence': confidence,
                        'reasoning': validation.get('reasoning', '')
                    }

                    total_validations_processed += 1

                except Exception as e:
                    logger.error("Error processing validation from %s: %s", validator_model, str(e))
                    logger.error("Problematic validation: %s", validation)

        logger.info("Processed %d validations total", total_validations_processed)

        # Summarize validations for each node
        nodes_with_validations = 0
        for node_key, node_data in validation_matrix.items():
            try:
                node_data['validation_summary'] = self._summarize_validations(node_data['validations'])
                if node_data['validations']:
                    nodes_with_validations += 1
            except Exception as e:
                logger.error("Error summarizing validations for node %s: %s", node_key, str(e))
                node_data['validation_summary'] = {
                    'total_validators': 0,
                    'agree_count': 0,
                    'disagree_count': 0,
                    'high_confidence_agree': 0,
                    'agreement_rate': 0.0,
                    'high_confidence_agreement_rate': 0.0,
                    'validator_details': {}
                }

        logger.info("Validation matrix complete: %d nodes have validations", nodes_with_validations)
        return validation_matrix

    def _summarize_validations(self, validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize validations for a single node"""
        total_validators = len(validations)
        agree_count = 0
        disagree_count = 0
        high_confidence_agree = 0

        validator_details = {}

        for validator_model, validation_data in validations.items():
            validation_result = validation_data.get('validation', 'UNKNOWN')
            confidence = validation_data.get('confidence', 0.0)

            try:
                confidence = float(confidence)
                if not (0.0 <= confidence <= 1.0):
                    logger.warning("Invalid confidence %s from %s, using 0.0", confidence, validator_model)
                    confidence = 0.0
            except (ValueError, TypeError):
                logger.warning("Non-numeric confidence %s from %s, using 0.0", confidence, validator_model)
                confidence = 0.0

            validator_details[validator_model] = {
                'result': validation_result,
                'confidence': confidence,
                'high_confidence': confidence >= self.agreement_threshold
            }

            if validation_result == 'AGREE':
                agree_count += 1
                if confidence >= self.agreement_threshold:
                    high_confidence_agree += 1
            elif validation_result == 'DISAGREE':
                disagree_count += 1

        return {
            'total_validators': total_validators,
            'agree_count': agree_count,
            'disagree_count': disagree_count,
            'high_confidence_agree': high_confidence_agree,
            'agreement_rate': agree_count / max(total_validators, 1),
            'high_confidence_agreement_rate': high_confidence_agree / max(total_validators, 1),
            'validator_details': validator_details
        }

    def _apply_triple_consensus_rules(self, validation_matrix: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply corrected triple consensus rules - each node validated by other 2 models"""
        consensus_nodes = []
        total_candidates = len(validation_matrix)

        logger.info("Applying corrected triple consensus rules to %d node candidates", total_candidates)

        expected_validators = 2  # Other 2 models validate each node

        nodes_with_full_validation = 0
        nodes_with_both_agree = 0
        nodes_with_high_confidence = 0

        for node_key, node_data in validation_matrix.items():
            try:
                summary = node_data['validation_summary']
                source_model = node_data['source_model']

                total_validators = summary.get('total_validators', 0)
                agree_count = summary.get('agree_count', 0)
                high_confidence_agree = summary.get('high_confidence_agree', 0)

                if total_validators == expected_validators:
                    nodes_with_full_validation += 1

                if agree_count == expected_validators:
                    nodes_with_both_agree += 1

                if high_confidence_agree == expected_validators:
                    nodes_with_high_confidence += 1

                # Corrected rule: both other models must agree with high confidence
                if (agree_count == expected_validators and
                        high_confidence_agree == expected_validators and
                        total_validators == expected_validators):

                    try:
                        original_node = node_data['original_node']
                        validations = node_data['validations']

                        # Calculate average validation confidence
                        validation_confidences = []
                        for v in validations.values():
                            conf = v.get('confidence', 0.0)
                            try:
                                conf = float(conf)
                                if 0.0 <= conf <= 1.0:
                                    validation_confidences.append(conf)
                            except (ValueError, TypeError):
                                logger.warning("Invalid validation confidence: %s", conf)

                        if not validation_confidences:
                            logger.warning("No valid validation confidences for node %s", node_key)
                            continue

                        avg_validation_confidence = statistics.mean(validation_confidences)

                        original_confidence = original_node.get('confidence', 0.0)
                        try:
                            original_confidence = float(original_confidence)
                            if not (0.0 <= original_confidence <= 1.0):
                                original_confidence = 0.0
                        except (ValueError, TypeError):
                            original_confidence = 0.0

                        consensus_node = {
                            'type': original_node.get('type', ''),
                            'name': original_node.get('name', ''),
                            'code_reference': str(original_node.get('code_reference', '')),
                            'line_start': int(original_node.get('line_start', 0)) if original_node.get(
                                'line_start') is not None else 0,
                            'line_end': int(original_node.get('line_end', 0)) if original_node.get(
                                'line_end') is not None else 0,
                            'rationale': str(original_node.get('rationale', '')),
                            'original_confidence': original_confidence,
                            'consensus_type': 'TRIPLE_UNANIMOUS',
                            'source_model': source_model,
                            'validation_confidence': round(avg_validation_confidence, 3),
                            'other_models_agree': True,
                            'validators_count': total_validators,
                            'validation_details': validations,
                            'quality_score': round((original_confidence + avg_validation_confidence) / 2, 3),
                            'node_key': node_key
                        }

                        consensus_nodes.append(consensus_node)
                        logger.debug("Triple consensus achieved for node: %s (quality: %.3f, source: %s)",
                                     node_key, consensus_node['quality_score'], source_model)

                    except Exception as e:
                        logger.error("Error creating consensus node for %s: %s", node_key, str(e))
                        continue

                else:
                    logger.debug(
                        "Node %s failed consensus: agree=%d/%d, high_conf=%d/%d, validators=%d/%d (source: %s)",
                        node_key, agree_count, expected_validators, high_confidence_agree, expected_validators,
                        total_validators, expected_validators, source_model)

            except Exception as e:
                logger.error("Error processing consensus for node %s: %s", node_key, str(e))
                continue

        # Sort by quality score
        consensus_nodes.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)

        logger.info("Corrected consensus statistics:")
        logger.info("  Total candidates: %d", total_candidates)
        logger.info("  Nodes with full validation (2/2): %d", nodes_with_full_validation)
        logger.info("  Nodes with both validators agree: %d", nodes_with_both_agree)
        logger.info("  Nodes with high confidence agreement: %d", nodes_with_high_confidence)
        logger.info("  Final consensus nodes: %d", len(consensus_nodes))

        return consensus_nodes

    def _generate_statistics(self, all_nodes: List[Dict[str, Any]],
                             validation_matrix: Dict[str, Dict[str, Any]],
                             consensus_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate consensus statistics"""
        total_unique_nodes = len(validation_matrix)
        total_consensus_nodes = len(consensus_nodes)

        # Analyze validation distribution for 2-validator scenario
        validation_stats = {
            'both_agree': 0,
            'one_agree': 0,
            'both_disagree': 0,
            'incomplete_validation': 0
        }

        for node_data in validation_matrix.values():
            summary = node_data['validation_summary']
            agree_count = summary.get('agree_count', 0)
            total_validators = summary.get('total_validators', 0)

            if total_validators < 2:
                validation_stats['incomplete_validation'] += 1
            elif agree_count == 2:
                validation_stats['both_agree'] += 1
            elif agree_count == 1:
                validation_stats['one_agree'] += 1
            else:
                validation_stats['both_disagree'] += 1

        return {
            'total_individual_nodes': len(all_nodes),
            'total_unique_nodes': total_unique_nodes,
            'total_consensus_nodes': total_consensus_nodes,
            'consensus_rate': round(total_consensus_nodes / max(total_unique_nodes, 1), 3),
            'validation_distribution': validation_stats,
            'agreement_threshold': self.agreement_threshold,
            'expected_validators_per_node': 2,
            'quality_metrics': {
                'average_quality_score': round(
                    statistics.mean([node['quality_score'] for node in consensus_nodes])
                    if consensus_nodes else 0.0, 3
                ),
                'min_quality_score': min(
                    [node['quality_score'] for node in consensus_nodes]) if consensus_nodes else 0.0,
                'max_quality_score': max(
                    [node['quality_score'] for node in consensus_nodes]) if consensus_nodes else 0.0
            }
        }


class TripleModelSecurityAnalyzer:
    """Main triple model security analyzer with corrected cross-validation"""

    def __init__(self, deepseek_key: str, gemini_key: str, qwen_key: str,
                 agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD):
        self.agreement_threshold = agreement_threshold
        self.analyzers = []

        try:
            self.analyzers.append(DeepSeekSecurityAnalyzer(deepseek_key))
            logger.info("DeepSeek analyzer initialized")
        except Exception as e:
            logger.error("Failed to initialize DeepSeek: %s", str(e))
            raise

        try:
            self.analyzers.append(GeminiSecurityAnalyzer(gemini_key))
            logger.info("Gemini analyzer initialized")
        except Exception as e:
            logger.error("Failed to initialize Gemini: %s", str(e))
            raise

        try:
            self.analyzers.append(QwenSecurityAnalyzer(qwen_key))
            logger.info("Qwen analyzer initialized")
        except Exception as e:
            logger.error("Failed to initialize Qwen: %s", str(e))
            raise

        if len(self.analyzers) != 3:
            raise Exception("Must have exactly 3 analyzers for triple consensus")

        self.consensus_calculator = TripleConsensusCalculator(agreement_threshold)

    def analyze_with_triple_consensus(self, code_function: str) -> Optional[Dict[str, Any]]:
        """Execute corrected triple model consensus analysis"""

        separator_line = "=" * 70
        logger.info(separator_line)
        logger.info("TRIPLE MODEL SECURITY ANALYSIS - CORRECTED UNANIMOUS CONSENSUS")
        logger.info(separator_line)

        # Phase 1: Independent analysis
        logger.info("PHASE 1: Independent Analysis by All Three Models")
        individual_results = []

        for i, analyzer in enumerate(self.analyzers, 1):
            try:
                logger.info("  %d/3: %s performing independent analysis...", i, analyzer.model_name)
                result = analyzer.execute_security_analysis(code_function)

                if result and result.get('critical_nodes'):
                    node_count = len(result.get('critical_nodes', []))
                    individual_results.append(result)
                    logger.info("  ✓ %s: %d critical nodes identified", analyzer.model_name, node_count)
                else:
                    logger.warning("  ✗ %s: No valid nodes found", analyzer.model_name)

            except Exception as e:
                logger.error("  ✗ %s: Analysis failed - %s", analyzer.model_name, str(e))

        if len(individual_results) != 3:
            logger.error("CRITICAL ERROR: Need exactly 3 successful analyses, got %d", len(individual_results))
            return None

        logger.info("PHASE 1 COMPLETE: All 3 models completed independent analysis")

        # Phase 2: Corrected cross-validation
        logger.info("\nPHASE 2: Building Cross-Validation Lists (Corrected)")
        validation_results = []

        for i, analyzer in enumerate(self.analyzers, 1):
            try:
                logger.info("  %d/3: %s preparing cross-validation...", i, analyzer.model_name)

                # Only collect nodes from other models (excluding self)
                other_models_nodes = []
                for result in individual_results:
                    if result.get('model_name') != analyzer.model_name:
                        model_name = result.get('model_name')
                        nodes = result.get('critical_nodes', [])

                        for node in nodes:
                            other_models_nodes.append({
                                'node': node,
                                'source_model': model_name
                            })

                logger.info("    %s will validate %d nodes from other models",
                            analyzer.model_name, len(other_models_nodes))

                validation_result = analyzer.execute_cross_validation(code_function, other_models_nodes)

                if validation_result and validation_result.get('validation_results') is not None:
                    validation_count = len(validation_result.get('validation_results', []))
                    validation_results.append(validation_result)
                    logger.info("  ✓ %s: %d validations completed", analyzer.model_name, validation_count)
                else:
                    logger.warning("  ✗ %s: Cross-validation failed or no results", analyzer.model_name)

            except Exception as e:
                logger.error("  ✗ %s: Cross-validation failed - %s", analyzer.model_name, str(e))

        if len(validation_results) != 3:
            logger.error("CRITICAL ERROR: Need exactly 3 successful validations, got %d", len(validation_results))
            return None

        logger.info("PHASE 2 COMPLETE: All 3 models completed cross-validation")

        # Phase 3: Calculate corrected consensus
        logger.info("\nPHASE 3: Corrected Triple Consensus Calculation")
        try:
            consensus_results = self.consensus_calculator.calculate_triple_consensus(
                individual_results,
                validation_results
            )

            consensus_count = len(consensus_results.get('consensus_nodes', []))
            stats = consensus_results.get('statistics', {})

            logger.info("  ✓ Corrected triple consensus calculation completed")
            logger.info("  ✓ %d nodes achieved unanimous agreement from other models", consensus_count)
            logger.info("  ✓ Consensus rate: %s", stats.get('consensus_rate', 0.0))

        except Exception as e:
            logger.error("  ✗ Triple consensus calculation failed: %s", str(e))
            return None

        # Build final result
        final_result = {
            'analysis_metadata': {
                'analysis_type': 'CORRECTED_TRIPLE_UNANIMOUS_CONSENSUS',
                'models_used': [analyzer.model_name for analyzer in self.analyzers],
                'agreement_threshold': self.agreement_threshold,
                'validation_rule': 'Each node validated by 2 other models, both must agree with high confidence',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'phase_results': {
                'independent_analysis': individual_results,
                'cross_validation': validation_results,
                'consensus_calculation': consensus_results
            },
            'final_summary': {
                'total_individual_nodes': sum(len(r.get('critical_nodes', [])) for r in individual_results),
                'total_unique_candidates': len(consensus_results.get('validation_matrix', {})),
                'unanimous_consensus_nodes': len(consensus_results.get('consensus_nodes', [])),
                'consensus_quality': consensus_results.get('statistics', {}).get('quality_metrics', {})
            },
            'unanimous_consensus_nodes': consensus_results.get('consensus_nodes', [])
        }

        # Final summary
        logger.info("\n" + separator_line)
        logger.info("CORRECTED TRIPLE CONSENSUS ANALYSIS COMPLETE")
        logger.info(separator_line)
        logger.info("Individual Nodes Found: %d", final_result['final_summary']['total_individual_nodes'])
        logger.info("Unique Node Candidates: %d", final_result['final_summary']['total_unique_candidates'])
        logger.info("UNANIMOUS CONSENSUS NODES: %d", final_result['final_summary']['unanimous_consensus_nodes'])

        quality_metrics = final_result['final_summary']['consensus_quality']
        if quality_metrics:
            logger.info("Average Quality Score: %s", quality_metrics.get('average_quality_score', 'N/A'))

        logger.info(separator_line)

        return final_result


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL format data file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse line %d: %s", line_num, str(e))
    return data


def save_jsonl_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL format file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_dataset(deepseek_key: str, gemini_key: str, qwen_key: str,
                    input_data_path: str, output_data_path: str,
                    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
                    limit: Optional[int] = None) -> None:
    """Process dataset with corrected triple model consensus analysis"""

    input_file = Path(input_data_path)
    if not input_file.exists():
        raise FileNotFoundError("Input file not found: {}".format(input_data_path))

    output_file = Path(output_data_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", input_data_path)
    try:
        data = load_jsonl_data(input_data_path)
        logger.info("Loaded %d items from dataset", len(data))
    except Exception as e:
        raise Exception("Failed to load input data: {}".format(str(e)))

    if limit is not None and limit > 0:
        data = data[:limit]
        logger.info("Limited processing to first %d items", limit)

    logger.info("Initializing corrected triple model analyzer (threshold: %s)", agreement_threshold)

    try:
        triple_analyzer = TripleModelSecurityAnalyzer(
            deepseek_key=deepseek_key,
            gemini_key=gemini_key,
            qwen_key=qwen_key,
            agreement_threshold=agreement_threshold
        )
    except Exception as e:
        raise Exception("Failed to initialize triple model analyzer: {}".format(str(e)))

    processed_data = []
    success_count = 0
    total_consensus_nodes = 0

    for i, item in enumerate(tqdm(data, desc="Processing with corrected triple consensus")):
        try:
            code_function = item.get('func', '')
            if not code_function.strip():
                logger.warning("Item %d has empty 'func' field, skipping", i + 1)
                continue

            separator = "=" * 50
            logger.info("\n%s", separator)
            logger.info("Processing Item %d/%d", i + 1, len(data))
            logger.info(separator)

            analysis_result = triple_analyzer.analyze_with_triple_consensus(code_function)

            if analysis_result is not None:
                enhanced_result = {
                    'item_id': i + 1,
                    'original_item': item,
                    'corrected_triple_consensus_analysis': analysis_result
                }
                processed_data.append(enhanced_result)
                success_count += 1

                consensus_nodes_count = len(analysis_result.get('unanimous_consensus_nodes', []))
                total_consensus_nodes += consensus_nodes_count

                logger.info("Item %d SUCCESS: %d corrected unanimous consensus nodes", i + 1, consensus_nodes_count)
            else:
                logger.warning("Item %d FAILED: Could not complete corrected triple consensus analysis", i + 1)

        except Exception as e:
            logger.error("Item %d ERROR: %s", i + 1, str(e))
            continue

    logger.info("\nSaving %d processed results to %s", len(processed_data), output_data_path)
    try:
        save_jsonl_data(processed_data, output_data_path)
        logger.info("Results saved successfully")
    except Exception as e:
        raise Exception("Failed to save output data: {}".format(str(e)))

    # Final statistics
    final_separator = "=" * 80
    logger.info("\n" + final_separator)
    logger.info("CORRECTED DATASET PROCESSING FINAL SUMMARY")
    logger.info(final_separator)
    logger.info("Total Items Processed: %d", len(data))
    logger.info("Successful Analyses: %d", success_count)
    logger.info("Success Rate: %.1f%%", round(success_count / len(data) * 100, 1))
    logger.info("Total Corrected Unanimous Consensus Nodes: %d", total_consensus_nodes)
    logger.info("Average Consensus Nodes per Item: %.2f", round(total_consensus_nodes / max(success_count, 1), 2))
    logger.info(final_separator)


def main():
    parser = argparse.ArgumentParser(description='Corrected Triple Model Unanimous Consensus Security Analysis')
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key')
    parser.add_argument('--gemini-key', required=True, help='Gemini API key')
    parser.add_argument('--qwen-key', required=True, help='Qwen API key')
    parser.add_argument('--input-data', required=True, help='Input JSONL file path')
    parser.add_argument('--output-data', required=True, help='Output JSONL file path')
    parser.add_argument('--agreement-threshold', type=float, default=DEFAULT_AGREEMENT_THRESHOLD,
                        help='Validation confidence threshold (default: 0.7)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of items to process')

    args = parser.parse_args()

    try:
        process_dataset(
            deepseek_key=args.deepseek_key,
            gemini_key=args.gemini_key,
            qwen_key=args.qwen_key,
            input_data_path=args.input_data,
            output_data_path=args.output_data,
            agreement_threshold=args.agreement_threshold,
            limit=args.limit
        )
    except Exception as e:
        logger.error("CRITICAL ERROR: %s", str(e))
        raise


if __name__ == '__main__':
    main()