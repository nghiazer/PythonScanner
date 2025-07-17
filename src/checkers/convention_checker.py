"""
Convention checker for Python codebases
"""

import logging
import time
from typing import List, Dict, Optional
from pathlib import Path

from src.core.llm_client import LLMClient
from src.core.rule_database import VectorRuleDatabase
from src.utils.file_utils import read_file_safely
from src.models.analysis_models import AnalysisResult, ProjectAnalysisResult, CodeViolation

logger = logging.getLogger(__name__)

class ConventionChecker:
    """Check code against established conventions"""
    
    def __init__(self, llm_client: LLMClient, rule_db: VectorRuleDatabase, config: Dict):
        self.llm_client = llm_client
        self.rule_db = rule_db
        self.config = config
        
        # Statistics
        self.stats = {
            "files_analyzed": 0,
            "violations_found": 0,
            "rules_applied": 0,
            "analysis_time": 0.0
        }
    
    def analyze_project(self, file_paths: List[str], project_path: str,
                       progress_callback: Optional[callable] = None) -> ProjectAnalysisResult:
        """Analyze project against conventions"""
        logger.info(f"Analyzing project: {len(file_paths)} files")
        start_time = time.time()
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Analyze file
                result = self.analyze_file(file_path)
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / len(file_paths)
                    progress_callback(progress, i + 1, len(file_paths))
                    
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                continue
        
        # Create project result
        project_result = ProjectAnalysisResult(project_path=project_path)
        project_result.file_results = results
        project_result.calculate_summary()
        
        # Update statistics
        self.stats["files_analyzed"] = len(results)
        self.stats["analysis_time"] = time.time() - start_time
        
        logger.info(f"Analysis completed in {self.stats['analysis_time']:.2f}s")
        return project_result
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze single file"""
        start_time = time.time()
        
        try:
            # Read file
            code = read_file_safely(file_path)
            if not code:
                return AnalysisResult(file_path=file_path, analysis_time=time.time() - start_time)
            
            # Get relevant rules
            rules = self.rule_db.find_relevant_rules(code[:500], n_results=10)
            
            # Check conventions
            violations = []
            suggestions = []
            
            if rules:
                # Use LLM to check conventions
                result = self.llm_client.check_conventions(code, rules)
                
                # Convert to violation objects
                for violation_data in result.get("violations", []):
                    violation = CodeViolation(
                        rule_type=violation_data.get("rule_type", "unknown"),
                        line=violation_data.get("line", 0),
                        message=violation_data.get("message", ""),
                        severity=violation_data.get("severity", "medium"),
                        current=violation_data.get("current", ""),
                        suggested=violation_data.get("suggested", ""),
                        file_path=file_path
                    )
                    violations.append(violation)
                
                suggestions = result.get("suggestions", [])
            
            # Calculate compliance score
            compliance_score = max(0.0, 1.0 - (len(violations) / 10))
            
            # Create result
            result = AnalysisResult(
                file_path=file_path,
                violations=violations,
                suggestions=suggestions,
                compliance_score=compliance_score,
                analysis_time=time.time() - start_time,
                rules_applied=len(rules)
            )
            
            # Update statistics
            self.stats["violations_found"] += len(violations)
            self.stats["rules_applied"] += len(rules)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return AnalysisResult(file_path=file_path, analysis_time=time.time() - start_time)
    
    def get_analysis_stats(self) -> Dict:
        """Get analysis statistics"""
        return self.stats.copy()
