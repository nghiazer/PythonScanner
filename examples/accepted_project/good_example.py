#!/usr/bin/env python3
"""
Example of well-formatted Python code.
"""

import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process data using best practices."""
    
    def __init__(self, data_source: str):
        """Initialize processor."""
        self.data_source = Path(data_source)
        self.processed_data: List[Dict] = []
    
    def load_data(self) -> bool:
        """Load data from source."""
        try:
            if not self.data_source.exists():
                logger.error(f"Data source not found: {self.data_source}")
                return False
            
            # Load data logic here
            self.processed_data = []
            return True
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not values:
            raise ValueError("Cannot calculate statistics for empty list")
        
        mean = sum(values) / len(values)
        
        return {
            "mean": mean,
            "count": len(values)
        }

def main():
    """Main function."""
    processor = DataProcessor("data.txt")
    
    if processor.load_data():
        logger.info("Data loaded successfully")
    else:
        logger.error("Failed to load data")

if __name__ == "__main__":
    main()
