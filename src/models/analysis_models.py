"""
Analysis data models
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class SeverityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class CodeViolation:
    rule_type: str
    line: int
    message: str
    severity: str
    current: str
    suggested: str
    file_path: str
    
    def to_dict(self) -> Dict:
        return {
            "rule_type": self.rule_type,
            "line": self.line,
            "message": self.message,
            "severity": self.severity,
            "current": self.current,
            "suggested": self.suggested,
            "file_path": self.file_path
        }

@dataclass
class AnalysisResult:
    file_path: str
    violations: List[CodeViolation] = None
    suggestions: List[str] = None
    compliance_score: float = 0.0
    analysis_time: float = 0.0
    rules_applied: int = 0
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class ProjectAnalysisResult:
    project_path: str
    file_results: List[AnalysisResult] = None
    total_files: int = 0
    total_violations: int = 0
    average_compliance_score: float = 0.0
    total_analysis_time: float = 0.0
    
    def __post_init__(self):
        if self.file_results is None:
            self.file_results = []
    
    def calculate_summary(self):
        """Calculate summary statistics"""
        if not self.file_results:
            return
        
        self.total_files = len(self.file_results)
        self.total_violations = sum(len(fr.violations) for fr in self.file_results)
        self.average_compliance_score = sum(fr.compliance_score for fr in self.file_results) / self.total_files
        self.total_analysis_time = sum(fr.analysis_time for fr in self.file_results)
