#!/usr/bin/env python3
"""
PrimeVul dataset sampling script

This script processes PrimeVul dataset with 20:1 sampling on non-vulnerable data
while preserving all vulnerable data to address data imbalance.

Author: AI Assistant
Date: 2025-05-24
"""

import json
import random
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrimeVulSampler:
    """PrimeVul dataset sampler"""

    def __init__(self, sampling_ratio: int = 20, random_seed: int = 42):
        """
        Initialize sampler

        Args:
            sampling_ratio (int): Sampling ratio for non-vulnerable data (N:1)
            random_seed (int): Random seed for reproducible results
        """
        self.sampling_ratio = sampling_ratio
        self.random_seed = random_seed
        random.seed(random_seed)

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load JSONL format file

        Args:
            file_path (str): File path

        Returns:
            List[Dict]: Data list
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse error at line {line_num}: {e}")
                            continue
            logger.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return []

    def save_jsonl(self, data: List[Dict], file_path: str) -> bool:
        """
        Save data in JSONL format

        Args:
           data (List[Dict]): Data to save
            file_path (str): Output file path

        Returns:
            bool: Whether save was successful
        """
        try:
            # Create output directory
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Successfully saved {len(data)} records to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False

    def separate_by_target(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate vulnerable and non-vulnerable data by target field

        Args:
           data (List[Dict]): Original data

        Returns:
            Tuple[List[Dict], List[Dict]]: (vulnerable_data, non_vulnerable_data)
        """
        vulnerable_data = []
        non_vulnerable_data = []

        for item in data:
            if item.get('target') == 1:
                vulnerable_data.append(item)
            elif item.get('target') == 0:
                non_vulnerable_data.append(item)
            else:
                logger.warning(f"Found abnormal target value: {item.get('target')}, idx: {item.get('idx')}")

        return vulnerable_data, non_vulnerable_data

    def sample_data(self, data: List[Dict]) -> List[Dict]:
        """
        Sample data

        Args:
           data (List[Dict]): Data to sample

        Returns:
            List[Dict]: Sampled data
        """
        if len(data) == 0:
            return []

        sample_size = max(1, len(data) // self.sampling_ratio)
        sampled_data = random.sample(data, sample_size)

        logger.info(f"Sampled {len(sampled_data)} from {len(data)} non-vulnerable records (ratio: {self.sampling_ratio}:1)")
        return sampled_data

    def process_dataset(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Process single dataset

        Args:
            input_path (str): Input file path
            output_path (str): Output file path

        Returns:
            Dict[str, int]: Processing statistics
        """
        logger.info(f"Processing dataset: {input_path}")

        # Load data
        data = self.load_jsonl(input_path)
        if not data:
            return {"total": 0, "vulnerable": 0, "non_vulnerable": 0, "sampled": 0}

        # Separate data
        vulnerable_data, non_vulnerable_data = self.separate_by_target(data)

        # Sample non-vulnerable data
        sampled_non_vulnerable = self.sample_data(non_vulnerable_data)

        # Merge data
        final_data = vulnerable_data + sampled_non_vulnerable

        # Shuffle data
        random.shuffle(final_data)

        # Save results
        self.save_jsonl(final_data, output_path)

        # Return statistics
        stats = {
            "total": len(data),
            "vulnerable": len(vulnerable_data),
            "non_vulnerable": len(non_vulnerable_data),
            "sampled": len(final_data)
        }

        return stats

    def print_statistics(self, dataset_name: str, stats: Dict[str, int]):
        """
        Print statistics

        Args:
            dataset_name (str): Dataset name
            stats (Dict[str, int]): Statistics
        """
        print(f"\n=== {dataset_name} Statistics ===")
        print(f"Original total: {stats['total']:,}")
        print(f"Vulnerable data: {stats['vulnerable']:,}")
        print(f"Non-vulnerable data: {stats['non_vulnerable']:,}")
        print(f"Sampled total: {stats['sampled']:,}")
        if stats['total'] > 0:
            vulnerable_ratio = stats['vulnerable'] / stats['total'] * 100
            final_vulnerable_ratio = stats['vulnerable'] / stats['sampled'] * 100 if stats['sampled'] > 0 else 0
            print(f"Original vulnerable ratio: {vulnerable_ratio:.2f}%")
            print(f"Sampled vulnerable ratio: {final_vulnerable_ratio:.2f}%")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PrimeVul dataset sampling tool')
    parser.add_argument('--input-dir', type=str, default='../data/PrimeVul',
                        help='Input data directory (default: ../data/PrimeVul)')
    parser.add_argument('--output-dir', type=str, default='../data/PrimeVul_sampled',
                        help='Output data directory (default: ../data/PrimeVul_sampled)')
    parser.add_argument('--sampling-ratio', type=int, default=20,
                        help='Sampling ratio N:1 (default: 20)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Define dataset files
    datasets = {
        'train': 'PrimeVul v0.1 train.jsonl',
        'valid': 'PrimeVul v0.1.jsonl',
        'test': 'PrimeVul v0.1 test.jsonl'
    }

    # Create sampler
    sampler = PrimeVulSampler(
        sampling_ratio=args.sampling_ratio,
        random_seed=args.random_seed
    )

    print(f"PrimeVul Dataset Sampling Tool")
    print(f"Sampling ratio: {args.sampling_ratio}:1")
    print(f"Random seed: {args.random_seed}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    total_stats = {}

    # Process each dataset
    for dataset_name, filename in datasets.items():
        input_path = os.path.join(args.input_dir, filename)
        output_filename = f"{filename.replace('.jsonl', '_sampled.jsonl')}"
        output_path = os.path.join(args.output_dir, output_filename)

        stats = sampler.process_dataset(input_path, output_path)
        total_stats[dataset_name] = stats
        sampler.print_statistics(dataset_name.upper(), stats)

    # Print overall statistics
    print(f"\n{'=' * 50}")
    print("Overall Statistics")
    print(f"{'=' * 50}")

    total_original = sum(stats['total'] for stats in total_stats.values())
    total_vulnerable = sum(stats['vulnerable'] for stats in total_stats.values())
    total_non_vulnerable = sum(stats['non_vulnerable'] for stats in total_stats.values())
    total_sampled = sum(stats['sampled'] for stats in total_stats.values())

    print(f"Original total: {total_original:,}")
    print(f"Total vulnerable: {total_vulnerable:,}")
    print(f"Total non-vulnerable: {total_non_vulnerable:,}")
    print(f"Sampled total: {total_sampled:,}")
    print(f"Data compression ratio: {total_sampled / total_original * 100:.2f}%")

    if total_original > 0:
        original_vulnerable_ratio = total_vulnerable / total_original * 100
        final_vulnerable_ratio = total_vulnerable / total_sampled * 100 if total_sampled > 0 else 0
        print(f"Original vulnerable ratio: {original_vulnerable_ratio:.2f}%")
        print(f"Sampled vulnerable ratio: {final_vulnerable_ratio:.2f}%")

    print(f"\nSampling completed! Output files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()