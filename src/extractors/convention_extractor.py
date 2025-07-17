"""
Fixed Convention Extractor with better error handling
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
    """Fixed convention extractor with improved error handling"""
    
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
        """Extract conventions from project files with improved error handling"""
        logger.info(f"Starting extraction from {len(file_paths)} files")
        
        all_conventions = []
        
        for i, file_path in enumerate(file_paths):
            try:
                logger.debug(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                # Read file content
                code = read_file_safely(file_path)
                if not code:
                    logger.warning(f"Could not read file: {file_path}")
                    continue
                
                # Skip very small files
                if len(code.strip()) < 50:
                    logger.debug(f"Skipping small file: {file_path}")
                    continue
                
                # Extract conventions from this file
                file_conventions = self._extract_from_code(code, file_path)
                
                if file_conventions:
                    all_conventions.extend(file_conventions)
                    self.stats["files_processed"] += 1
                    logger.debug(f"Extracted {len(file_conventions)} conventions from {file_path}")
                else:
                    logger.debug(f"No conventions extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats["errors"] += 1
                continue
        
        logger.info(f"Total conventions extracted: {len(all_conventions)}")
        
        # Store conventions if any found
        if all_conventions:
            try:
                success = self.rule_db.store_conventions(all_conventions, project_info)
                if success:
                    self.stats["conventions_extracted"] = len(all_conventions)
                    logger.info(f"Successfully stored {len(all_conventions)} conventions")
                    return True
                else:
                    logger.error("Failed to store conventions in database")
                    return False
            except Exception as e:
                logger.error(f"Failed to store conventions: {e}")
                return False
        else:
            logger.warning("No conventions extracted from any file")
            return False
    
    def _extract_from_code(self, code: str, file_path: str = "") -> List[Dict]:
        """Extract conventions from code with improved error handling"""
        try:
            logger.debug(f"Extracting conventions from code ({len(code)} chars)")
            
            # Use LLM to extract conventions
            conventions = self.llm_client.extract_conventions([code])
            self.stats["llm_calls"] += 1
            
            if not conventions:
                logger.debug("No conventions returned from LLM")
                return []
            
            # Filter by confidence
            filtered_conventions = []
            for conv in conventions:
                try:
                    confidence = float(conv.get("confidence", 0.0))
                    if confidence >= self.min_confidence:
                        # Add source file info
                        conv["source_file"] = file_path
                        filtered_conventions.append(conv)
                    else:
                        logger.debug(f"Filtered out low confidence convention: {confidence}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid confidence value: {conv.get('confidence')}")
                    continue
            
            logger.debug(f"Filtered to {len(filtered_conventions)} high-confidence conventions")
            return filtered_conventions
            
        except Exception as e:
            logger.error(f"Failed to extract conventions from code: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.stats.copy()
