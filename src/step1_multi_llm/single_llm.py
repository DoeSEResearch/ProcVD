import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import time
import google.generativeai as genai
import re

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.1


# Logging configuration
def setup_logging(log_file: str = 'gemini_security_analysis.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class GeminiSecurityAnalyzer:
    """Gemini 2.0 Flash model security analyzer for single model analysis"""

    def __init__(self, api_key: str, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: int = DEFAULT_RETRY_DELAY):
        self.model_name = "Gemini-2.0-Flash"
        genai.configure(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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

When identifying these nodes, pay close attention to: dangerous API calls, sensitive data operations (arrays, pointers, etc, noting boundary, type, and lifecycle issues), critical arithmetic operations, important control logic affecting security (watch for missing checks), handling of external inputs, and protections at trust boundaries. Also, understand the flow of critical data (e.g., inputs, pointer values, operation results) and how it impacts sensitive operations or decisions, to determine if it constitutes potential vulnerability links (trigger, propagation, exploitation) or effective defenses.

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

    def _call_model_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Call model with improved retry mechanism and exponential backoff"""
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Convert messages to single prompt for Gemini
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
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS,
                    )
                )
                return response.text

            except Exception as e:
                last_error = e
                retries += 1
                logger.warning("%s API call failed (attempt %d/%d): %s", self.model_name, retries, self.max_retries,
                               str(e))

                # Implement exponential backoff with jitter for different error types
                wait_time = self._calculate_wait_time(e, retries)
                if retries < self.max_retries:
                    logger.info("Waiting %ds before retry...", wait_time)
                    time.sleep(wait_time)

        raise Exception(
            "{} API call failed after {} retries: {}".format(self.model_name, self.max_retries, str(last_error)))

    def _calculate_wait_time(self, error: Exception, retry_count: int) -> int:
        """Calculate wait time based on error type and retry count"""
        error_str = str(error).lower()

        # Check for rate limiting errors
        if any(keyword in error_str for keyword in ['quota', 'rate limit', 'throttling', 'too many requests']):
            # Exponential backoff for rate limits: 10, 20, 40 seconds
            return self.retry_delay * (2 ** (retry_count - 1))

        # Check for server errors (5xx)
        elif any(keyword in error_str for keyword in ['internal error', 'server error', '500', '503']):
            # Moderate backoff for server errors: 5, 10, 15 seconds
            return max(5, self.retry_delay // 2 * retry_count)

        # Default backoff for other errors
        else:
            return self.retry_delay

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response with robust error handling"""
        try:
            response_clean = response.strip()

            # Remove code block markers
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

    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON format issues"""
        if not json_str or not json_str.strip():
            return "{}"

        # Remove inline comments
        lines = json_str.split('\n')
        clean_lines = []
        for line in lines:
            if '//' in line:
                line = line[:line.index('//')]
            clean_lines.append(line)
        json_str = '\n'.join(clean_lines)

        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Fix quote issues
        json_str = json_str.replace('\\n', ' ').replace('\\t', ' ')

        # Remove control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        return json_str

    def _fallback_json_parse(self, response: str) -> List[Dict[str, Any]]:
        """Fallback JSON parsing method"""
        try:
            # Try to extract JSON part
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


def process_dataset(api_key: str, input_data_path: str, output_data_path: str,
                    limit: Optional[int] = None, max_retries: int = DEFAULT_MAX_RETRIES,
                    retry_delay: int = DEFAULT_RETRY_DELAY, log_file: str = 'gemini_security_analysis.log') -> None:
    """Process dataset with single Gemini model analysis"""

    # Setup logging with custom log file
    global logger
    logger = setup_logging(log_file)

    # Validate input file
    input_file = Path(input_data_path)
    if not input_file.exists():
        raise FileNotFoundError("Input file not found: {}".format(input_data_path))

    # Create output directory
    output_file = Path(output_data_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading dataset from %s", input_data_path)
    try:
        data = load_jsonl_data(input_data_path)
        logger.info("Loaded %d items from dataset", len(data))
    except Exception as e:
        raise Exception("Failed to load input data: {}".format(str(e)))

    # Apply data limit
    if limit is not None and limit > 0:
        data = data[:limit]
        logger.info("Limited processing to first %d items", limit)

    # Initialize analyzer
    logger.info("Initializing Gemini analyzer (retries: %d, delay: %ds)", max_retries, retry_delay)
    try:
        analyzer = GeminiSecurityAnalyzer(
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    except Exception as e:
        raise Exception("Failed to initialize Gemini analyzer: {}".format(str(e)))

    # Process dataset
    processed_data = []
    success_count = 0
    total_nodes_found = 0

    for i, item in enumerate(tqdm(data, desc="Processing with Gemini")):
        try:
            code_function = item.get('func', '')
            if not code_function.strip():
                logger.warning("Item %d has empty 'func' field, skipping", i + 1)
                continue

            separator = "=" * 50
            logger.info("\n%s", separator)
            logger.info("Processing Item %d/%d with Gemini", i + 1, len(data))
            logger.info(separator)

            # Execute analysis
            analysis_result = analyzer.execute_security_analysis(code_function)

            if analysis_result is not None:
                # Add original data information
                enhanced_result = {
                    'item_id': i + 1,
                    'original_item': item,
                    'gemini_analysis': analysis_result
                }
                processed_data.append(enhanced_result)
                success_count += 1

                nodes_count = len(analysis_result.get('critical_nodes', []))
                total_nodes_found += nodes_count
                logger.info("Item %d SUCCESS: %d critical nodes found", i + 1, nodes_count)
            else:
                logger.warning("Item %d FAILED: Could not complete analysis", i + 1)

        except Exception as e:
            logger.error("Item %d ERROR: %s", i + 1, str(e))
            continue

    # Save results
    logger.info("\nSaving %d processed results to %s", len(processed_data), output_data_path)
    try:
        save_jsonl_data(processed_data, output_data_path)
        logger.info("Results saved successfully")
    except Exception as e:
        raise Exception("Failed to save output data: {}".format(str(e)))

    # Final statistics
    final_separator = "=" * 60
    logger.info("\n" + final_separator)
    logger.info("GEMINI SINGLE MODEL ANALYSIS COMPLETE")
    logger.info(final_separator)
    logger.info("Total Items Processed: %d", len(data))
    logger.info("Successful Analyses: %d", success_count)
    logger.info("Success Rate: %.1f%%", round(success_count / len(data) * 100, 1))
    logger.info("Total Critical Nodes Found: %d", total_nodes_found)
    logger.info("Average Nodes per Success: %.2f", round(total_nodes_found / max(success_count, 1), 2))
    logger.info(final_separator)


def main():
    parser = argparse.ArgumentParser(description='Gemini Single Model Security Analysis')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--input-data', required=True, help='Input JSONL file path')
    parser.add_argument('--output-data', required=True, help='Output JSONL file path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of items to process')
    parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES,
                        help='Maximum number of retries for API calls')
    parser.add_argument('--retry-delay', type=int, default=DEFAULT_RETRY_DELAY,
                        help='Base delay between retries in seconds')
    parser.add_argument('--log-file', default='gemini_security_analysis.log',
                        help='Log file path')

    args = parser.parse_args()

    try:
        process_dataset(
            api_key=args.api_key,
            input_data_path=args.input_data,
            output_data_path=args.output_data,
            limit=args.limit,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            log_file=args.log_file
        )
    except Exception as e:
        logger.error("CRITICAL ERROR: %s", str(e))
        raise


if __name__ == '__main__':
    main()