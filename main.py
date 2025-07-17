#!/usr/bin/env python3
"""
Python Code Convention Analyzer - Main Entry Point
Complete implementation with all phases integrated
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path

from src.core.llm_client import LLMClient
from src.core.rule_database import VectorRuleDatabase
from src.extractors.convention_extractor import ConventionExtractor
from src.checkers.convention_checker import ConventionChecker
from src.models.config_models import load_config
from src.utils.file_utils import discover_python_files
from src.utils.setup_validator import SetupValidator

logger = logging.getLogger(__name__)

class ConventionAnalyzer:
    """Main application class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.llm_client = None
        self.rule_database = None
        self.extractor = None
        self.checker = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing Code Convention Analyzer...")
            
            # Initialize LLM client
            self.llm_client = LLMClient(
                model_path=self.config.llm.model_path,
                config=self.config.llm.__dict__
            )
            
            if not self.llm_client.load_model():
                logger.error("Failed to load LLM model")
                return False
            
            # Initialize database
            self.rule_database = VectorRuleDatabase()
            
            # Initialize extractor
            self.extractor = ConventionExtractor(
                llm_client=self.llm_client,
                rule_db=self.rule_database
            )
            
            # Initialize checker
            self.checker = ConventionChecker(
                llm_client=self.llm_client,
                rule_db=self.rule_database,
                config=self.config.analysis.__dict__
            )
            
            self._initialized = True
            logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def train_on_accepted_codebase(self, project_path: str) -> bool:
        """Train on accepted codebase"""
        if not self._initialized:
            return False
        
        try:
            logger.info(f"Training on: {project_path}")
            
            # Discover files
            python_files = discover_python_files(project_path, self.config.analysis.ignore_patterns)
            
            if not python_files:
                logger.error("No Python files found")
                return False
            
            # Extract conventions
            success = self.extractor.extract_from_project(python_files)
            
            if success:
                stats = self.extractor.get_extraction_stats()
                logger.info(f"Training completed: {stats['conventions_extracted']} conventions extracted")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def analyze_new_project(self, project_path: str, output_file: str = None) -> dict:
        """Analyze new project"""
        if not self._initialized:
            return {}
        
        try:
            logger.info(f"Analyzing: {project_path}")
            
            # Discover files
            python_files = discover_python_files(project_path, self.config.analysis.ignore_patterns)
            
            if not python_files:
                logger.error("No Python files found")
                return {}
            
            # Analyze project
            result = self.checker.analyze_project(python_files, project_path)
            
            # Create report
            report = {
                "project_path": project_path,
                "total_files": result.total_files,
                "total_violations": result.total_violations,
                "average_compliance": result.average_compliance_score,
                "file_results": [
                    {
                        "file_path": fr.file_path,
                        "compliance_score": fr.compliance_score,
                        "violations": len(fr.violations)
                    }
                    for fr in result.file_results
                ]
            }
            
            # Save report
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved: {output_file}")
            
            # Print summary
            logger.info(f"Analysis completed: {result.total_violations} violations found")
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        if self.llm_client:
            self.llm_client.cleanup()
        if self.rule_database:
            self.rule_database.cleanup()

def setup_logging(verbose: bool = False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    Path('logs').mkdir(exist_ok=True)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Python Code Convention Analyzer")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train on accepted codebase')
    train_parser.add_argument('project_path', help='Path to accepted project')
    train_parser.add_argument('--config', default='config/config.yaml')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze new project')
    analyze_parser.add_argument('project_path', help='Path to project to analyze')
    analyze_parser.add_argument('--output', default='./analysis_report.json')
    analyze_parser.add_argument('--config', default='config/config.yaml')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'train':
            analyzer = ConventionAnalyzer(args.config)
            if analyzer.initialize():
                try:
                    success = analyzer.train_on_accepted_codebase(args.project_path)
                    sys.exit(0 if success else 1)
                finally:
                    analyzer.cleanup()
            else:
                sys.exit(1)
        
        elif args.command == 'analyze':
            analyzer = ConventionAnalyzer(args.config)
            if analyzer.initialize():
                try:
                    report = analyzer.analyze_new_project(args.project_path, args.output)
                    sys.exit(0 if report else 1)
                finally:
                    analyzer.cleanup()
            else:
                sys.exit(1)
        
        elif args.command == 'status':
            validator = SetupValidator()
            if validator.validate_complete_setup():
                print("✅ System is ready")
                sys.exit(0)
            else:
                print("❌ System not ready")
                sys.exit(1)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
