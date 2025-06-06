import os
import json
import subprocess
import logging
import argparse
from tqdm import tqdm
import shutil
import traceback
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Configuration Paths ===
JOERN_EXPORT = "YOUR JOERN-EXPORT PATH"  # Set correct joern-export path
FAILED_JSON_NAME = "../../result/joern_export_failed_export_items.json"
LOG_FILE_NAME = "../../result/cpg_export_json.log"


def ensure_directory_exists(path):
    """Ensure the directory exists for the given path"""
    directory = os.path.dirname(path) if not path.endswith('/') else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def extract_number_from_filename(filename):
    """
    Extract number from filename
    Example: PrimeVul_balanced_test_187732.bin -> 187732
    """
    match = re.search(r'(\d+)\.bin$', filename)
    if match:
        return match.group(1)
    return None


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, LOG_FILE_NAME)
    ensure_directory_exists(log_file)

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def export_to_neo4jgraphson(cpg_info: dict, output_dir: str) -> dict:
    """
    Export single CPG.bin file to Neo4j GraphSON format
    Args:
        cpg_info: Dict containing CPG file info {'path': file_path, 'number': id, 'basename': base_name}
        output_dir: Output directory
    Returns:
        dict: Processing result dictionary
    """
    cpg_path = cpg_info['path']
    file_number = cpg_info['number']
    basename = cpg_info['basename']

    try:
        logging.info(f"[{file_number}] Processing file: {cpg_path}")

        # Check if input file exists
        if not os.path.exists(cpg_path):
            logging.error(f"[{file_number}] Error: CPG bin file not found")
            return {'success': False, 'info': cpg_info, 'reason': 'CPG bin file not found'}

        # Create temp output directory
        temp_output_dir = os.path.join(output_dir, f"temp_{file_number}")
        if os.path.exists(temp_output_dir):
            logging.info(f"[{file_number}] Cleaning existing temp directory")
            shutil.rmtree(temp_output_dir)

        # Build export command
        export_cmd = [
            JOERN_EXPORT,
            "--repr=all",
            "--format=graphson",
            "--out", temp_output_dir,
            cpg_path
        ]
        logging.debug(f"[{file_number}] Executing command: {' '.join(export_cmd)}")

        # Execute export command
        logging.info(f"[{file_number}] Exporting...")
        result = subprocess.run(
            export_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Log command output
        if result.stdout:
            logging.debug(f"[{file_number}] Command output:\n{result.stdout}")
        if result.stderr:
            logging.debug(f"[{file_number}] Command stderr:\n{result.stderr}")

        # Move and rename exported file
        export_json_path = os.path.join(temp_output_dir, "export.json")
        target_json_path = os.path.join(output_dir, f"{file_number}.json")

        if os.path.exists(export_json_path):
            shutil.move(export_json_path, target_json_path)
            logging.info(f"[{file_number}] Export file saved as: {file_number}.json")
        else:
            logging.warning(f"[{file_number}] export.json file not found")
            return {'success': False, 'info': cpg_info, 'reason': 'export.json file not found'}

        # Cleanup temp directory
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            logging.debug(f"[{file_number}] Cleaned temp directory: {temp_output_dir}")

        logging.info(f"[{file_number}] Export completed successfully")
        return {'success': True, 'info': cpg_info, 'reason': 'Export successful'}

    except subprocess.TimeoutExpired as e:
        logging.error(f"[{file_number}] Error: Export timeout - {str(e)}")
        return {'success': False, 'info': cpg_info, 'reason': f'Export timeout: {str(e)}'}
    except subprocess.CalledProcessError as e:
        logging.error(f"[{file_number}] Error: Export failed (exit code {e.returncode})")
        logging.error(f"[{file_number}] Error details:\n{e.stderr}")
        return {'success': False, 'info': cpg_info, 'reason': f'Export failed (exit code {e.returncode}): {e.stderr}'}
    except Exception as e:
        logging.error(f"[{file_number}] Error: Unknown error - {str(e)}")
        logging.error(f"[{file_number}] Stack trace:\n{traceback.format_exc()}")
        return {'success': False, 'info': cpg_info, 'reason': f'Unknown error: {str(e)}'}


def scan_cpg_files(input_dir):
    """
    Scan input directory for all CPG.bin files
    Args:
        input_dir: Input directory path
    Returns:
        list: List of CPG file information
    """
    cpg_files = []

    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return cpg_files

    logging.info(f"Scanning input directory: {input_dir}")

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            # Look for .bin files in subdirectory
            for file in os.listdir(item_path):
                if file.endswith('.bin'):
                    file_path = os.path.join(item_path, file)
                    file_number = extract_number_from_filename(file)

                    if file_number:
                        cpg_info = {
                            'path': file_path,
                            'number': file_number,
                            'basename': os.path.splitext(file)[0],
                            'folder': item
                        }
                        cpg_files.append(cpg_info)
                        logging.debug(f"Found CPG file: {file_path} -> ID: {file_number}")
                    else:
                        logging.warning(f"Cannot extract number from filename: {file}")

    logging.info(f"Found {len(cpg_files)} CPG files total")
    return cpg_files


def main(input_dir, output_dir, limit=None, max_workers=None):
    """Main processing function with multiprocessing support"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        setup_logging(output_dir)

        logging.info(f"Starting CPG export task")
        logging.info(f"Input directory: {input_dir}")
        logging.info(f"Output directory: {output_dir}")

        # Scan CPG files
        cpg_files = scan_cpg_files(input_dir)

        if not cpg_files:
            logging.error("No CPG files found")
            return

        # Sort by number
        cpg_files.sort(key=lambda x: int(x['number']))

        if limit:
            cpg_files = cpg_files[:limit]
            logging.info(f"Test mode: Processing only first {limit} samples")
        else:
            logging.info(f"Full processing mode: Processing {len(cpg_files)} CPG files")

        success_count = 0
        failed_records = []

        logging.info(f"Starting multiprocess export (max workers: {max_workers or os.cpu_count()})...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(export_to_neo4jgraphson, cpg_info, output_dir): cpg_info for cpg_info in
                       cpg_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting CPG files"):
                cpg_info = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        success_count += 1
                    else:
                        failed_records.append({
                            "id": result['info']['number'],
                            "filename": os.path.basename(result['info']['path']),
                            "folder": result['info']['folder'],
                            "reason": result['reason'],
                            "input_file": result['info']['path']
                        })
                except Exception as e:
                    logging.error(f"[{cpg_info['number']}] Multiprocess exception: {str(e)}")
                    failed_records.append({
                        "id": cpg_info['number'],
                        "filename": os.path.basename(cpg_info['path']),
                        "folder": cpg_info['folder'],
                        "reason": f"Multiprocess exception: {str(e)}",
                        "input_file": cpg_info['path']
                    })

        # Save failure information
        failed_json_path = os.path.join(output_dir, FAILED_JSON_NAME)
        with open(failed_json_path, "w", encoding="utf-8") as f:
            json.dump(failed_records, f, ensure_ascii=False, indent=4)

        logging.info(f"Processing completed! Success: {success_count}, Failed: {len(failed_records)}")
        if failed_records:
            logging.info("Failure details:")
            for record in failed_records:
                logging.info(f"- ID {record['id']} ({record['filename']}): {record['reason']}")

    except Exception as e:
        logging.error(f"Critical error in main program: {str(e)}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch export CPG.bin files to Neo4j GraphSON format")
    parser.add_argument("input_dir",
                        help="Input directory path containing CPG.bin file subfolders")
    parser.add_argument("output_dir",
                        help="Output directory path for saving GraphSON format files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes, defaults to CPU count")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found - {args.input_dir}")
        exit(1)

    # Validate joern-export tool
    if not os.path.exists(JOERN_EXPORT):
        print(f"Error: joern-export tool not found - {JOERN_EXPORT}")
        print("Please ensure joern-export path is configured correctly")
        exit(1)

    main(args.input_dir, args.output_dir, limit=args.limit, max_workers=args.workers)