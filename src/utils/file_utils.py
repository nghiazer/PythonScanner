"""
File utility functions
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def discover_python_files(project_path: str, ignore_patterns: List[str]) -> List[str]:
    """Discover Python files in project"""
    python_files = []
    project_path = Path(project_path)
    
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return []
    
    for root, dirs, files in os.walk(project_path):
        # Filter ignored directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not any(pattern in file_path for pattern in ignore_patterns):
                    python_files.append(file_path)
    
    logger.info(f"Discovered {len(python_files)} Python files")
    return python_files

def read_file_safely(file_path: str, max_size: int = 1024 * 1024) -> Optional[str]:
    """Safely read file"""
    try:
        file_path = Path(file_path)
        
        if file_path.stat().st_size > max_size:
            logger.warning(f"File too large: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None
