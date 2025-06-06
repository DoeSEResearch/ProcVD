import os
import json
import subprocess
import logging
import argparse
import glob
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Constants
DEFAULT_JOERN_PARSE = "Your joern-parse executable path"
DEFAULT_MEMORY = "20G"
DEFAULT_TIMEOUT = 300
SUPPORTED_C_EXTENSIONS = {'.c', '.h'}
SUPPORTED_CPP_EXTENSIONS = {'.cpp', '.cxx', '.cc', '.c++', '.hpp', '.hxx', '.h++'}
CPP_INDICATORS = {'cpp', 'cxx', 'cc', '++'}


# Configuration
class CPGConfig:
    """Configuration class for CPG generation"""

    def __init__(self,
                 joern_parse_path: str = DEFAULT_JOERN_PARSE,
                 input_dir: str = "../../data/primevul_process",
                 output_base_dir: str = "../../data/primevul_cpg_bin_files",
                 failed_json: str = "../../result/cpg_extractor_failed_items.json",
                 log_file: str = "../../result/cpg_generation_jsonl.log",
                 memory_limit: str = DEFAULT_MEMORY,
                 timeout: int = DEFAULT_TIMEOUT):
        self.joern_parse_path = joern_parse_path
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.failed_json = Path(failed_json)
        self.log_file = Path(log_file)
        self.memory_limit = memory_limit
        self.timeout = timeout

        # Create necessary directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.failed_json.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def init_logger(logger_queue):
    """Initialize logger for multiprocessing workers"""
    while not logger_queue.empty():
        record = logger_queue.get()
        logger = logging.getLogger()
        logger.handle(record)


def get_file_extension(file_name: str) -> str:
    """
    Determine appropriate file extension based on file name

    Args:
        file_name: Original file name

    Returns:
        Appropriate extension (.c or .cpp)
    """
    if not file_name:
        return ".c"

    file_name_lower = file_name.lower()

    # Check for C++ extensions
    if any(file_name_lower.endswith(ext) for ext in SUPPORTED_CPP_EXTENSIONS):
        return ".cpp"

    # Check for C extensions
    if any(file_name_lower.endswith(ext) for ext in SUPPORTED_C_EXTENSIONS):
        if file_name_lower.endswith('.h'):
            # Header file heuristic: check for C++ indicators
            if any(indicator in file_name_lower for indicator in CPP_INDICATORS):
                return ".cpp"
        return ".c"

    # Default to C
    return ".c"


def validate_joern_installation(joern_parse_path: str) -> bool:
    """
    Validate Joern installation

    Args:
        joern_parse_path: Path to joern-parse executable

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(joern_parse_path):
        return False

    try:
        result = subprocess.run(
            [joern_parse_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False


def load_jsonl_files(input_dir: Path) -> List[Path]:
    """
    Load all JSONL files from the input directory

    Args:
        input_dir: Input directory path

    Returns:
        List of JSONL file paths
    """
    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    return sorted(jsonl_files)


def load_jsonl_data(jsonl_file: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load data from a single JSONL file

    Args:
        jsonl_file: Path to JSONL file
        logger: Logger instance

    Returns:
        List of loaded records
    """
    data = []
    file_basename = jsonl_file.stem

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    # Add metadata for tracking
                    record['source_file'] = file_basename
                    record['line_number'] = line_num
                    # Create unique ID combining file and original idx
                    original_idx = record.get('idx', line_num)
                    record['unique_id'] = f"{file_basename}_{original_idx}"
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON in {jsonl_file.name} line {line_num}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error reading {jsonl_file}: {e}")
        return []

    logger.info(f"Loaded {len(data)} records from {jsonl_file.name}")
    return data


def generate_cpg_bin(unique_id: str,
                     code: str,
                     source_file: str,
                     file_name: str,
                     config: CPGConfig,
                     logger_queue=None) -> bool:
    """
    Generate CPG binary file using Joern

    Args:
        unique_id: Unique identifier for the code sample
        code: Source code content
        source_file: Source file identifier
        file_name: Original file name
        config: Configuration object
        logger_queue: Queue for multiprocessing logging

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory structure: base_dir/source_file/unique_id/
        output_dir = config.output_base_dir / source_file
        folder = output_dir / unique_id
        folder.mkdir(parents=True, exist_ok=True)

        # Create source file with appropriate extension
        ext = get_file_extension(file_name)
        code_file = folder / f"code_{unique_id}{ext}"

        # Write code to file with error handling
        try:
            with open(code_file, "w", encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            _log_message(logger_queue, logging.ERROR,
                         f"[{unique_id}] ❌ Failed to write code file: {str(e)}")
            return False

        cpg_bin = folder / f"{unique_id}.bin"

        # Build joern-parse command
        parse_cmd = [
            config.joern_parse_path,
            str(code_file),
            "--output", str(cpg_bin),
            f"-J-Xmx{config.memory_limit}"
        ]

        # Execute joern-parse
        result = subprocess.run(
            parse_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.timeout
        )

        # Validate output
        if not cpg_bin.exists() or cpg_bin.stat().st_size == 0:
            _log_message(logger_queue, logging.WARNING,
                         f"[{unique_id}] ⚠️ CPG bin file not created or is empty")
            return False

        _log_message(logger_queue, logging.INFO,
                     f"[{unique_id}] ✅ Successfully generated CPG")
        return True

    except subprocess.TimeoutExpired:
        _log_message(logger_queue, logging.ERROR,
                     f"[{unique_id}] ⏱️ Timeout ({config.timeout}s) while generating CPG")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        _log_message(logger_queue, logging.ERROR,
                     f"[{unique_id}] ❌ Failed to generate CPG: {error_msg}")
        return False
    except Exception as e:
        _log_message(logger_queue, logging.ERROR,
                     f"[{unique_id}] ❌ Unexpected error: {str(e)}")
        return False


def _log_message(logger_queue, level: int, message: str):
    """Helper function to log messages in multiprocessing context"""
    if logger_queue:
        logger_queue.put(logging.makeLogRecord({
            'msg': message,
            'levelno': level,
            'levelname': logging.getLevelName(level)
        }))


def process_record(row: Dict[str, Any], config: CPGConfig, logger_queue=None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Process a single record

    Args:
        row: Data record containing code and metadata
        config: Configuration object
        logger_queue: Queue for multiprocessing logging

    Returns:
        Tuple of (success, failure_info)
    """
    unique_id = row.get('unique_id')
    source_file = row.get('source_file')
    code = row.get("func")
    file_name = row.get("file_name", "")

    # Validate input
    if not isinstance(code, str) or not code.strip():
        _log_message(logger_queue, logging.WARNING,
                     f"[{unique_id}] ❌ Empty or non-string code")
        return False, {
            "unique_id": unique_id,
            "source_file": source_file,
            "reason": "Empty or non-string code",
            "file_name": file_name,
            "row": dict(row)
        }

    # Generate CPG
    success = generate_cpg_bin(unique_id, code, source_file, file_name, config, logger_queue)

    if not success:
        return False, {
            "unique_id": unique_id,
            "source_file": source_file,
            "reason": "CPG generation failed",
            "file_name": file_name,
            "code": code.strip()[:500] + "..." if len(code.strip()) > 500 else code.strip(),
            "row": dict(row)
        }

    return True, None


def process_chunk(chunk: List[Dict[str, Any]], config: CPGConfig, logger_queue) -> List[
    Tuple[bool, Optional[Dict[str, Any]]]]:
    """
    Process a chunk of records in a worker process

    Args:
        chunk: List of records to process
        config: Configuration object
        logger_queue: Queue for multiprocessing logging

    Returns:
        List of processing results
    """
    results = []
    for row in chunk:
        results.append(process_record(row, config, logger_queue))
    return results


def split_into_chunks(data: List[Any], num_chunks: int) -> List[List[Any]]:
    """
    Split data into roughly equal chunks for multiprocessing

    Args:
       data: List of data to split
        num_chunks: Number of chunks to create

    Returns:
        List of data chunks
    """
    if not data:
        return []

    chunk_size = (len(data) + num_chunks - 1) // num_chunks
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def generate_summary_stats(all_data: List[Dict[str, Any]],
                           failed_records: List[Dict[str, Any]],
                           logger: logging.Logger):
    """
    Generate and log summary statistics

    Args:
        all_data: All processed data
        failed_records: List of failed records
        logger: Logger instance
    """
    source_files = set(row.get('source_file') for row in all_data)
    c_count = sum(1 for row in all_data if get_file_extension(row.get('file_name', '')) == '.c')
    cpp_count = sum(1 for row in all_data if get_file_extension(row.get('file_name', '')) == '.cpp')

    logger.info("📊 Summary Statistics:")
    logger.info(f"  Total samples processed: {len(all_data)}")
    logger.info(f"  C code samples: {c_count}")
    logger.info(f"  C++ code samples: {cpp_count}")
    logger.info(f"  Source files: {len(source_files)}")

    logger.info("📊 Results by source file:")
    for source_file in sorted(source_files):
        file_total = sum(1 for row in all_data if row.get('source_file') == source_file)
        file_failed = sum(1 for fail in failed_records if fail.get('source_file') == source_file)
        file_success = file_total - file_failed
        success_rate = (file_success / file_total * 100) if file_total > 0 else 0
        logger.info(f"  {source_file}: {file_success}/{file_total} succeeded ({success_rate:.1f}%)")


def main(config: CPGConfig, limit: Optional[int] = None, num_processes: Optional[int] = None):
    """
    Main processing function

    Args:
        config: Configuration object
        limit: Limit number of samples for testing
        num_processes: Number of worker processes
    """
    logger = setup_logging(config.log_file)

    # Validate Joern installation
    if not validate_joern_installation(config.joern_parse_path):
        logger.error(f"❌ Joern installation not found or invalid at: {config.joern_parse_path}")
        logger.error("Please ensure Joern is properly installed and the path is correct.")
        return

    logger.info(f"✅ Joern installation validated: {config.joern_parse_path}")
    logger.info("📥 Loading JSONL data from directory...")

    # Load all JSONL files
    try:
        jsonl_files = load_jsonl_files(config.input_dir)
        logger.info(f"Found {len(jsonl_files)} JSONL files:")
        for file in jsonl_files:
            logger.info(f"  - {file.name}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Load all data from all JSONL files
    all_data = []
    for jsonl_file in jsonl_files:
        file_data = load_jsonl_data(jsonl_file, logger)
        all_data.extend(file_data)

    if not all_data:
        logger.error("No valid data found in JSONL files. Exiting.")
        return

    # Apply limit if specified
    if limit:
        all_data = all_data[:limit]
        logger.info(f"🔬 Testing mode: Processing {limit} samples")
    else:
        logger.info(f"🏃 Full processing: {len(all_data)} code samples from {len(jsonl_files)} files")

    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), len(all_data)) if len(all_data) > 0 else 1
    else:
        num_processes = min(num_processes, cpu_count(), len(all_data)) if len(all_data) > 0 else 1

    success_count = 0
    failed_records = []

    if num_processes > 1:
        # Multiprocessing mode
        logger.info(f"🚀 Starting multiprocessing with {num_processes} workers")

        with Manager() as manager:
            logger_queue = manager.Queue()
            chunks = split_into_chunks(all_data, num_processes)

            with tqdm(total=len(all_data), desc="Processing code samples") as pbar:
                with Pool(processes=num_processes, initializer=init_logger, initargs=(logger_queue,)) as pool:
                    process_func = partial(process_chunk, config=config, logger_queue=logger_queue)
                    chunk_results = []

                    for chunk_result in pool.imap_unordered(process_func, chunks):
                        chunk_results.extend(chunk_result)

                        # Process log messages from workers
                        while not logger_queue.empty():
                            record = logger_queue.get()
                            logging.getLogger().handle(record)

                        # Update progress
                        pbar.update(len(chunk_result))

                    # Count successes and failures
                    for success, fail in chunk_results:
                        if success:
                            success_count += 1
                        elif fail:
                            failed_records.append(fail)

    else:
        # Single-process mode
        logger.info("🐌 Running in single-process mode")
        for row in tqdm(all_data, desc="Processing code samples"):
            success, fail = process_record(row, config)
            if success:
                success_count += 1
            elif fail:
                failed_records.append(fail)

    # Save failure information
    try:
        with open(config.failed_json, "w", encoding="utf-8") as f:
            json.dump(failed_records, f, ensure_ascii=False, indent=2)
        logger.info(f"❌ Failure details saved to: {config.failed_json}")
    except Exception as e:
        logger.error(f"Failed to save failure log: {e}")

    # Generate summary statistics
    generate_summary_stats(all_data, failed_records, logger)

    # Final summary
    success_rate = (success_count / len(all_data) * 100) if len(all_data) > 0 else 0
    logger.info(f"🎉 Processing complete: {success_count}/{len(all_data)} succeeded ({success_rate:.1f}%)")
    logger.info(f"💾 CPG binaries saved to: {config.output_base_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Joern CPG Binary Generator for C/C++ Code from JSONL Files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--joern-path", type=str, default=DEFAULT_JOERN_PARSE,
                        help="Path to joern-parse executable")
    parser.add_argument("--input-dir", type=str, default="../data/primevul_process",
                        help="Directory containing JSONL files")
    parser.add_argument("--output-dir", type=str, default="../data/primevul_cpg_bin_files",
                        help="Output directory for CPG binary files")
    parser.add_argument("--failed-json", type=str, default="../data/failed_items.json",
                        help="Path to save failed items log")
    parser.add_argument("--log-file", type=str, default="../log/cpg_generation_jsonl.log",
                        help="Log file path")
    parser.add_argument("--memory", type=str, default=DEFAULT_MEMORY,
                        help="Memory limit for Joern (e.g., 20G)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Timeout for CPG generation in seconds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for testing")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of worker processes to use (default: CPU count)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create configuration
    config = CPGConfig(
        joern_parse_path=args.joern_path,
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        failed_json=args.failed_json,
        log_file=args.log_file,
        memory_limit=args.memory,
        timeout=args.timeout
    )

    try:
        main(config, limit=args.limit, num_processes=args.processes)
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    except Exception as e:
        logging.error(f"💥 Critical error: {e}")
        raise