#!/usr/bin/env python3
"""
Example of well-formatted Python code following best practices.
"""

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Process and analyze data using various statistical methods.
    """

    def __init__(self, data_source: str):
        """
        Initialize the data processor.

        Args:
            data_source: Path to the data source file
        """
        self.data_source = Path(data_source)
        self.processed_data: List[Dict] = []
        self._is_loaded = False

    def load_data(self) -> bool:
        """
        Load data from the configured source.

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            if not self.data_source.exists():
                logger.error(f"Data source not found: {self.data_source}")
                return False

            # Simulated data loading
            self.processed_data = []
            self._is_loaded = True

            logger.info(f"Data loaded successfully from {self.data_source}")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary containing mean, median, and standard deviation
        """
        if not values:
            raise ValueError("Cannot calculate statistics for empty list")

        sorted_values = sorted(values)
        n = len(values)

        # Calculate mean
        mean = sum(values) / n

        # Calculate median
        if n % 2 == 0:
            median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            median = sorted_values[n//2]

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = variance ** 0.5

        return {
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "count": n
        }

def process_file_list(file_paths: List[str], output_dir: str = "./output") -> bool:
    """
    Process a list of files and save results.

    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save processed results

    Returns:
        True if all files processed successfully
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    success_count = 0

    for file_path in file_paths:
        try:
            processor = DataProcessor(file_path)
            if processor.load_data():
                success_count += 1
                logger.info(f"Successfully processed: {file_path}")
            else:
                logger.warning(f"Failed to process: {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    success_rate = success_count / len(file_paths) if file_paths else 0
    logger.info(f"Processing complete: {success_count}/{len(file_paths)} files")

    return success_rate > 0.8

def main():
    """Main function demonstrating usage."""
    logging.basicConfig(level=logging.INFO)

    test_files = ["data1.txt", "data2.txt", "data3.txt"]

    try:
        result = process_file_list(test_files)
        if result:
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing failed")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
