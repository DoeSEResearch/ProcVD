import json
import os
import re
import subprocess
import logging
import argparse
import glob
import shutil
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
from dataclasses import dataclass

# Constants
DEFAULT_JOERN_SLICE_CMD = "joern-slice"
DEFAULT_SLICE_DEPTH = 4
DEFAULT_MEMORY = "10G"
DEFAULT_TIMEOUT = 180
DEFAULT_LOG_LEVEL = logging.INFO


# Configuration dataclass
@dataclass
class SliceConfig:
    """Configuration for CPG slice processing"""
    joern_slice_cmd: str = DEFAULT_JOERN_SLICE_CMD
    slice_depth: int = DEFAULT_SLICE_DEPTH
    memory_limit: str = DEFAULT_MEMORY
    timeout_seconds: int = DEFAULT_TIMEOUT
    log_level: int = DEFAULT_LOG_LEVEL


class CPGSliceProcessor:
    """Process CPG files to generate subgraphs based on consensus nodes"""

    def __init__(self,
                 cpg_bin_base_dir: str,
                 input_jsonl_path: str,
                 output_base_dir: str,
                 config: SliceConfig = None,
                 num_processes: Optional[int] = None):
        """
        Initialize processor with paths and configuration

        Args:
            cpg_bin_base_dir: Base directory containing CPG binary files
            input_jsonl_path: Path to input JSONL file with consensus nodes
            output_base_dir: Output directory for generated subgraph JSON files
            config: Configuration object for slice processing
            num_processes: Number of parallel processes
        """
        self.cpg_bin_base_dir = Path(cpg_bin_base_dir).resolve()
        self.input_jsonl_path = Path(input_jsonl_path) if input_jsonl_path else None
        self.output_base_dir = Path(output_base_dir)
        self.config = config or SliceConfig()
        self.num_processes = num_processes or max(1, cpu_count() - 1)

        self.logger = self._setup_logging()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'empty_slices': 0,
            'start_time': datetime.now()
        }

        self.logger.info(f"Initialized CPG Slice Processor with {self.num_processes} processes")
        self.logger.info(f"Configuration: depth={self.config.slice_depth}, "
                         f"memory={self.config.memory_limit}, timeout={self.config.timeout_seconds}s")

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging system"""
        logger = logging.getLogger('CPGSliceProcessor')
        logger.setLevel(self.config.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create logs directory
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"cpg_slice_{timestamp}.log"

        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.log_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _validate_environment(self) -> None:
        """Validate all required environment components"""
        self.logger.info("Validating environment...")

        # Check input paths
        if self.input_jsonl_path and not self.input_jsonl_path.exists():
            raise FileNotFoundError(f"Input JSONL file not found: {self.input_jsonl_path}")

        if not self.cpg_bin_base_dir.exists():
            raise FileNotFoundError(f"CPG binaries directory not found: {self.cpg_bin_base_dir}")

        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory validated: {self.output_base_dir}")

        # Verify joern-slice availability
        self._validate_joern_slice()

    def _validate_joern_slice(self) -> None:
        """Validate joern-slice installation"""
        try:
            result = subprocess.run(
                [self.config.joern_slice_cmd, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                version_info = result.stdout.strip() or "version info not available"
                self.logger.info(f"Joern-slice validated: {version_info}")
            else:
                raise RuntimeError(f"Joern-slice validation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Joern-slice validation timed out")
        except FileNotFoundError:
            raise RuntimeError(f"Joern-slice command not found: {self.config.joern_slice_cmd}")
        except Exception as e:
            raise RuntimeError(f"Joern-slice validation failed: {str(e)}")

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load and validate input JSONL dataset"""
        if not self.input_jsonl_path:
            raise ValueError("Input JSONL path not provided")

        self.logger.info(f"Loading dataset from {self.input_jsonl_path}")

        data = []
        try:
            with open(self.input_jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON at line {line_num}: {str(e)}")
                        continue

            if not data:
                raise ValueError("No valid JSON objects found in JSONL file")

            self.logger.info(f"Loaded {len(data)} records from dataset")
            return data

        except Exception as e:
            self.logger.error(f"Dataset loading failed: {str(e)}")
            raise

    def _find_cpg_file(self, idx: str) -> Optional[Path]:
        """
        Find CPG file by idx with improved search strategy

        Args:
            idx: Identifier to search for

        Returns:
            Path to CPG file if found, None otherwise
        """
        # Normalize idx
        idx = str(idx).strip()

        # Direct path patterns (most common cases)
        direct_patterns = [
            self.cpg_bin_base_dir / f"PrimeVul_balanced_train_{idx}" / f"PrimeVul_balanced_train_{idx}.bin",
            self.cpg_bin_base_dir / f"PrimeVul_balanced_test_{idx}" / f"PrimeVul_balanced_test_{idx}.bin",
            self.cpg_bin_base_dir / f"{idx}" / f"{idx}.bin"
        ]

        # Check direct patterns first
        for pattern in direct_patterns:
            if pattern.exists():
                self.logger.debug(f"Found CPG file for idx {idx}: {pattern}")
                return pattern

        # Glob patterns for more flexible matching
        glob_patterns = [
            f"*_{idx}/*_{idx}.bin",
            f"*_{idx}/{idx}.bin",
            f"{idx}_{idx}/{idx}.bin"
        ]

        for pattern in glob_patterns:
            matches = list(self.cpg_bin_base_dir.glob(pattern))
            if matches:
                # Validate the match is exact
                for match in matches:
                    if self._validate_cpg_match(match, idx):
                        self.logger.debug(f"Found CPG file via glob for idx {idx}: {match}")
                        return match

        # Recursive search as last resort
        self.logger.debug(f"Trying recursive search for idx {idx}...")
        for cpg_file in self.cpg_bin_base_dir.rglob("*.bin"):
            if self._validate_cpg_match(cpg_file, idx):
                self.logger.debug(f"Found CPG file via recursive search for idx {idx}: {cpg_file}")
                return cpg_file

        self.logger.warning(f"CPG file not found for idx {idx}")
        return None

    def _validate_cpg_match(self, cpg_path: Path, idx: str) -> bool:
        """
        Validate that a CPG file path matches the given idx

        Args:
            cpg_path: Path to CPG file
            idx: Index to validate against

        Returns:
            True if the path matches the idx exactly
        """
        file_name = cpg_path.name
        dir_name = cpg_path.parent.name

        # Check if filename matches exactly
        expected_filenames = [f"{idx}.bin", f"*_{idx}.bin"]
        filename_match = file_name == f"{idx}.bin" or file_name.endswith(f"_{idx}.bin")

        # Check if directory name contains idx appropriately
        dir_match = (dir_name == idx or
                     dir_name.endswith(f"_{idx}") or
                     f"_{idx}_" in dir_name)

        return filename_match and dir_match

    def _build_sink_pattern(self, node_name: str) -> str:
        """
        Construct regex pattern from node name with validation

        Args:
            node_name: Name of the node to create pattern for

        Returns:
            Regex pattern string

        Raises:
            ValueError: If node_name is invalid or creates invalid regex
        """
        if not node_name or not isinstance(node_name, str):
            raise ValueError("Invalid node_name: must be non-empty string")

        node_name = node_name.strip()
        if not node_name:
            raise ValueError("Invalid node_name: cannot be empty after stripping")

        try:
            # Escape special regex characters but preserve meaningful structure
            escaped = re.escape(node_name)
            pattern = f".*{escaped}.*"

            # Test pattern compilation
            re.compile(pattern)

            self.logger.debug(f"Built sink pattern for '{node_name}': {pattern}")
            return pattern

        except re.error as e:
            error_msg = f"Invalid regex pattern for node_name '{node_name}': {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _execute_joern_slice(self,
                             idx: str,
                             cpg_path: Path,
                             output_path: Path,
                             sink_filter: str) -> Tuple[bool, str]:
        """
        Execute joern-slice command and return success status and message

        Args:
            idx: Identifier for logging
            cpg_path: Path to CPG binary file
            output_path: Path for output JSON file
            sink_filter: Regex pattern for sink filtering

        Returns:
            Tuple of (success, message)
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.config.joern_slice_cmd,
            "data-flow",
            f"--slice-depth={self.config.slice_depth}",
            f"--sink-filter={sink_filter}",
            "-o", str(output_path),
            f"-J-Xmx{self.config.memory_limit}",
            str(cpg_path)
        ]

        self.logger.debug(f"Executing command: {' '.join(cmd)}")
        self.logger.debug(f"Sink filter: {sink_filter}")

        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.timeout_seconds
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Log command output for debugging
            if result.stdout:
                self.logger.debug(f"Command stdout: {result.stdout.strip()}")
            if result.stderr:
                self.logger.debug(f"Command stderr: {result.stderr.strip()}")

            if result.returncode == 0:
                # Check for empty slice conditions
                stdout_lower = result.stdout.lower()
                if any(phrase in stdout_lower for phrase in ['empty slice', 'no file generated', 'no matches']):
                    self.logger.info(f"Empty slice for {idx} - no matching nodes found")
                    return False, "Empty slice - no matching nodes found"

                # Check if output file was created and is valid
                if output_path.exists() and output_path.stat().st_size > 0:
                    self.logger.info(f"Successfully generated slice for {idx} in {duration:.2f}s")
                    return True, f"Generated subgraph: {output_path}"
                else:
                    self.logger.warning(f"Command succeeded but output file missing/empty: {output_path}")
                    return False, "Command succeeded but no valid output file"
            else:
                error_msg = f"Command failed (exit {result.returncode}): {result.stderr.strip()}"
                self.logger.error(f"Slice generation failed for {idx}: {error_msg}")
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"Timeout after {self.config.timeout_seconds} seconds"
            self.logger.error(f"Slice generation timed out for {idx}: {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Slice generation error for {idx}: {error_msg}", exc_info=True)
            return False, error_msg

    def _merge_json_files(self, json_files: List[Path], output_path: Path) -> bool:
        """
        Merge multiple JSON files into one array

        Args:
            json_files: List of JSON file paths to merge
            output_path: Output path for merged file

        Returns:
            True if successful, False otherwise
        """
        try:
            merged_data = []

            for json_file in json_files:
                if not json_file.exists():
                    self.logger.warning(f"JSON file not found: {json_file}")
                    continue

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            merged_data.extend(data)
                        else:
                            merged_data.append(data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON in file {json_file}: {str(e)}")
                    continue

            # Write merged data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Merged {len(json_files)} JSON files into {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to merge JSON files: {str(e)}")
            return False

    def _process_single_item(self, item: Dict[str, Any]) -> Tuple[str, bool, str]:
        """
        Process a single item with consensus nodes

        Args:
            item: Item dictionary containing idx and consensus_nodes

        Returns:
            Tuple of (idx, success, message)
        """
        idx = str(item.get('idx', 'UNKNOWN_IDX'))
        consensus_nodes = item.get('consensus_nodes', [])

        self.logger.info(f"Processing item idx {idx} with {len(consensus_nodes)} consensus nodes")

        # Validate consensus nodes
        if not consensus_nodes:
            return idx, False, "No consensus nodes found"

        # Find CPG file
        cpg_path = self._find_cpg_file(idx)
        if not cpg_path:
            return idx, False, "CPG file not found"

        # Check if output already exists
        final_output_path = self.output_base_dir / f"{idx}.json"
        if final_output_path.exists():
            return idx, False, "Output file already exists"

        try:
            if len(consensus_nodes) == 1:
                # Single node processing
                return self._process_single_node(idx, consensus_nodes[0], cpg_path, final_output_path)
            else:
                # Multiple nodes processing
                return self._process_multiple_nodes(idx, consensus_nodes, cpg_path, final_output_path)

        except Exception as e:
            self.logger.error(f"Error processing idx {idx}: {str(e)}", exc_info=True)
            return idx, False, f"Unexpected error: {str(e)}"

    def _process_single_node(self,
                             idx: str,
                             node: Dict[str, Any],
                             cpg_path: Path,
                             output_path: Path) -> Tuple[str, bool, str]:
        """Process single consensus node"""
        node_name = node.get('name')
        if not node_name:
            return idx, False, "Empty node name"

        self.logger.info(f"Processing single consensus node '{node_name}' for idx {idx}")

        try:
            sink_filter = self._build_sink_pattern(node_name)
        except ValueError as e:
            return idx, False, f"Invalid sink pattern: {str(e)}"

        success, message = self._execute_joern_slice(idx, cpg_path, output_path, sink_filter)

        if success:
            return idx, True, "Successfully processed single node"
        else:
            return idx, False, message

    def _process_multiple_nodes(self,
                                idx: str,
                                nodes: List[Dict[str, Any]],
                                cpg_path: Path,
                                output_path: Path) -> Tuple[str, bool, str]:
        """Process multiple consensus nodes"""
        self.logger.info(f"Processing {len(nodes)} consensus nodes for idx {idx}")

        valid_temp_files = []
        temp_dir = self.output_base_dir / f"temp_{idx}"

        try:
            for node_idx, node in enumerate(nodes):
                node_name = node.get('name')
                if not node_name:
                    self.logger.warning(f"Skipping node {node_idx} for idx {idx}: Empty name")
                    continue

                self.logger.debug(f"Processing node '{node_name}' ({node_idx + 1}/{len(nodes)})")

                temp_output_path = temp_dir / f"node_{node_idx}.json"

                try:
                    sink_filter = self._build_sink_pattern(node_name)
                except ValueError as e:
                    self.logger.warning(f"Invalid sink pattern for '{node_name}': {str(e)}")
                    continue

                success, message = self._execute_joern_slice(
                    f"{idx}_node_{node_idx}", cpg_path, temp_output_path, sink_filter
                )

                if success and temp_output_path.exists() and temp_output_path.stat().st_size > 0:
                    valid_temp_files.append(temp_output_path)
                    self.logger.debug(f"Successfully processed node '{node_name}'")
                else:
                    self.logger.debug(f"No output for node '{node_name}': {message}")

            # Process results based on number of valid outputs
            return self._finalize_multiple_nodes_result(idx, valid_temp_files, output_path)

        finally:
            # Cleanup temporary directory
            self._cleanup_temp_directory(temp_dir)

    def _finalize_multiple_nodes_result(self,
                                        idx: str,
                                        valid_temp_files: List[Path],
                                        output_path: Path) -> Tuple[str, bool, str]:
        """Finalize processing of multiple nodes"""
        if not valid_temp_files:
            return idx, False, "No valid subgraphs generated"
        elif len(valid_temp_files) == 1:
            # Single valid output, use directly
            shutil.copy2(valid_temp_files[0], output_path)
            self.logger.info(f"Using single valid subgraph for idx {idx}")
            return idx, True, "Successfully processed with single valid subgraph"
        else:
            # Multiple valid outputs, merge them
            merge_success = self._merge_json_files(valid_temp_files, output_path)
            if merge_success:
                self.logger.info(f"Successfully merged {len(valid_temp_files)} subgraphs for idx {idx}")
                return idx, True, f"Successfully merged {len(valid_temp_files)} subgraphs"
            else:
                return idx, False, "Failed to merge subgraphs"

    def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up temporary directory"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp directory {temp_dir}: {str(e)}")

    def process_items(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Process multiple items and return statistics

        Args:
            items: List of items to process

        Returns:
            Dictionary with processing statistics
        """
        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'empty_slices': 0}

        if self.num_processes > 1:
            # Multiprocessing
            self.logger.info(f"Processing {len(items)} items with {self.num_processes} processes")

            with Pool(processes=self.num_processes) as pool:
                process_func = partial(
                    process_item_worker,
                    str(self.cpg_bin_base_dir),
                    str(self.output_base_dir),
                    self.config
                )

                results = pool.map(process_func, items)

                # Aggregate results
                for idx, success, message in results:
                    if success:
                        stats['success'] += 1
                    elif "already exists" in message or "No consensus nodes" in message:
                        stats['skipped'] += 1
                    elif "Empty slice" in message:
                        stats['empty_slices'] += 1
                    else:
                        stats['failed'] += 1
        else:
            # Single process
            self.logger.info("Processing items in single-process mode")
            for item in items:
                idx, success, message = self._process_single_item(item)
                if success:
                    stats['success'] += 1
                elif "already exists" in message or "No consensus nodes" in message:
                    stats['skipped'] += 1
                elif "Empty slice" in message:
                    stats['empty_slices'] += 1
                else:
                    stats['failed'] += 1

        return stats

    def run(self) -> None:
        """Main execution method"""
        try:
            self.logger.info("Starting CPG Slice Processing Pipeline")
            self._validate_environment()

            dataset = self._load_dataset()
            self.stats['total'] = len(dataset)

            self.logger.info(f"Processing {len(dataset)} items with {self.num_processes} processes")

            # Process all items
            processing_stats = self.process_items(dataset)

            # Update final statistics
            self.stats.update(processing_stats)

            # Generate final report
            self._generate_final_report()

        except Exception as e:
            self.logger.critical(f"Fatal error in processing pipeline: {str(e)}", exc_info=True)
            raise

    def _generate_final_report(self) -> None:
        """Generate and log final processing report"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("CPG SLICE PROCESSING COMPLETED - FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total items:           {self.stats['total']}")
        self.logger.info(f"Successfully processed: {self.stats['success']}")
        self.logger.info(f"Failed:                {self.stats['failed']}")
        self.logger.info(f"Skipped:               {self.stats['skipped']}")
        self.logger.info(f"Empty slices:          {self.stats.get('empty_slices', 0)}")
        self.logger.info(f"Total duration:        {duration:.2f} seconds")

        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            avg_time = duration / self.stats['total']
            self.logger.info(f"Success rate:          {success_rate:.1f}%")
            self.logger.info(f"Average time per item: {avg_time:.2f} seconds")

        self.logger.info("=" * 60 + "\n")


def process_item_worker(cpg_bin_base_dir: str,
                        output_base_dir: str,
                        config: SliceConfig,
                        item: Dict[str, Any]) -> Tuple[str, bool, str]:
    """
    Worker function for multiprocessing

    Args:
        cpg_bin_base_dir: Base directory for CPG files
        output_base_dir: Output directory
        config: Slice configuration
        item: Item to process

    Returns:
        Tuple of (idx, success, message)
    """
    # Create temporary processor instance for single item
    processor = CPGSliceProcessor(cpg_bin_base_dir, "", output_base_dir, config, 1)
    return processor._process_single_item(item)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CPG Slice Processor for generating subgraphs from consensus nodes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--cpg-bin-dir', required=True,
                        help='Base directory containing CPG bin files in subdirectories')
    parser.add_argument('--input-jsonl', required=True,
                        help='Path to input JSONL file with consensus nodes')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for generated subgraph JSON files')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes (default: CPU cores - 1)')

    # Configuration options
    parser.add_argument('--joern-slice-cmd', default=DEFAULT_JOERN_SLICE_CMD,
                        help='Joern slice command name or path')
    parser.add_argument('--slice-depth', type=int, default=DEFAULT_SLICE_DEPTH,
                        help='Slice depth for joern-slice')
    parser.add_argument('--memory', default=DEFAULT_MEMORY,
                        help='Memory limit (e.g., 10G)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help='Timeout for each slice operation in seconds')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')

    return parser.parse_args()


def main() -> int:
    """Main function with command line argument parsing"""
    try:
        args = parse_arguments()

        # Validate input arguments
        if not os.path.exists(args.cpg_bin_dir):
            print(f"Error: CPG bin directory does not exist: {args.cpg_bin_dir}")
            return 1

        if not os.path.exists(args.input_jsonl):
            print(f"Error: Input JSONL file does not exist: {args.input_jsonl}")
            return 1

        # Create configuration
        config = SliceConfig(
            joern_slice_cmd=args.joern_slice_cmd,
            slice_depth=args.slice_depth,
            memory_limit=args.memory,
            timeout_seconds=args.timeout,
            log_level=getattr(logging, args.log_level.upper())
        )

        # Create and run processor
        processor = CPGSliceProcessor(
            args.cpg_bin_dir,
            args.input_jsonl,
            args.output_dir,
            config,
            args.processes
        )

        processor.run()
        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())