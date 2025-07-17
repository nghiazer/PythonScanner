"""
Memory monitoring utilities for LLM operations
"""

import psutil
import logging
import gc
from typing import Dict

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.cleanup_count = 0
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is high"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100
            
            if usage_percent > self.threshold:
                logger.warning(f"High memory usage: {usage_percent:.1%}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return False
    
    def check_memory_availability(self, required_gb: float) -> bool:
        """Check if enough memory is available"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return available_gb >= required_gb
            
        except Exception as e:
            logger.error(f"Memory availability check failed: {e}")
            return False
    
    def force_cleanup(self):
        """Force garbage collection"""
        gc.collect()
        self.cleanup_count += 1
        logger.info("Memory cleanup performed")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024**2)
        except Exception:
            return 0.0
