"""
Convention extractor for Python codebases
"""

import logging
import time
import ast
from typing import List, Dict, Optional
from pathlib import Path

from src.core.llm_client import LLMClient
from src.core.rule_database import VectorRuleDatabase
from src.utils.file_utils import read_file_safely

logger = logging.getLogger(__name__)

class ConventionExtractor:
    """Extract conventions from Python code"""
    
    def __init__(self, llm_client: LLMClient, rule_db: VectorRuleDatabase,
                 min_confidence: float = 0.7):
        self.llm_client = llm_client
        self.rule_db = rule_db
        self.min_confidence = min_confidence
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "conventions_extracted": 0,
            "llm_calls": 0,
            "errors": 0
        }
    
    def extract_from_project(self, file_paths: List[str], 
                           project_info: Optional[Dict] = None) -> bool:
        """Extract conventions from project files"""
        logger.info(f"Extracting conventions from {len(file_paths)} files")
        
        all_conventions = []
        
        for file_path in file_paths:
            try:
                logger.debug(f"Processing: {file_path}")
                
                # Read file
                code = read_file_safely(file_path)
                if not code:
                    continue
                
                # Extract conventions
                conventions = self._extract_from_code(code)
                
                if conventions:
                    all_conventions.extend(conventions)
                    self.stats["files_processed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats["errors"] += 1
                continue
        
        # Store conventions
        if all_conventions:
            success = self.rule_db.store_conventions(all_conventions, project_info)
            if success:
                self.stats["conventions_extracted"] = len(all_conventions)
                logger.info(f"Extracted {len(all_conventions)} conventions")
                return True
        
        return False
    
    def _extract_from_code(self, code: str) -> List[Dict]:
        """Extract conventions from code"""
        try:
            # Use LLM to extract conventions
            conventions = self.llm_client.extract_conventions([code])
            self.stats["llm_calls"] += 1
            
            # Filter by confidence
            filtered_conventions = [
                conv for conv in conventions
                if conv.get("confidence", 0) >= self.min_confidence
            ]
            
            return filtered_conventions
            
        except Exception as e:
            logger.error(f"Failed to extract conventions: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.stats.copy()
