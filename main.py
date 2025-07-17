#!/usr/bin/env python3
"""
Fixed main.py with better error handling
"""

import sys
import argparse
import logging
import json
import time
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyzer.log'),
        logging.StreamHandler()
    ]
)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

def safe_import():
    """Safely import modules"""
    try:
        from src.core.llm_client import LLMClient
        from src.core.rule_database import VectorRuleDatabase
        from src.extractors.convention_extractor import ConventionExtractor
        from src.checkers.convention_checker import ConventionChecker
        from src.models.config_models import load_config
        from src.utils.file_utils import discover_python_files
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

class ConventionAnalyzer:
    """Enhanced analyzer with better error handling"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.llm_client = None
        self.rule_database = None
        self.extractor = None
        self.checker = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize with enhanced error handling"""
        try:
            logger.info("Initializing analyzer...")
            
            # Load config
            from src.models.config_models import load_config
            self.config = load_config(self.config_path)
            
            # Initialize components
            from src.core.llm_client import LLMClient
            from src.core.rule_database import VectorRuleDatabase
            from src.extractors.convention_extractor import ConventionExtractor
            from src.checkers.convention_checker import ConventionChecker
            
            self.llm_client = LLMClient(
                model_path=self.config.llm.model_path,
                config=self.config.llm.__dict__
            )
            
            if not self.llm_client.load_model():
                logger.error("Failed to load LLM model")
                return False
                
            self.rule_database = VectorRuleDatabase()
            self.extractor = ConventionExtractor(self.llm_client, self.rule_database)
            self.checker = ConventionChecker(self.llm_client, self.rule_database, self.config.analysis.__dict__)
            
            self._initialized = True
            logger.info("Initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def train_on_accepted_codebase(self, project_path: str) -> bool:
        """Enhanced training with better error handling"""
        if not self._initialized:
            return False
        
        try:
            logger.info(f"Starting training on: {project_path}")
            
            # Discover files
            from src.utils.file_utils import discover_python_files
            python_files = discover_python_files(project_path, self.config.analysis.ignore_patterns)
            
            if not python_files:
                logger.error("No Python files found")
                return False
            
            logger.info(f"Found {len(python_files)} Python files")
            
            # Extract conventions
            logger.info("Extracting conventions...")
            success = self.extractor.extract_from_project(python_files)
            
            if success:
                # Get stats
                stats = self.extractor.get_extraction_stats()
                logger.info(f"Training completed successfully!")
                logger.info(f"  Files processed: {stats.get('files_processed', 0)}")
                logger.info(f"  Conventions extracted: {stats.get('conventions_extracted', 0)}")
                
                # Verify database
                db_stats = self.rule_database.get_statistics()
                logger.info(f"  Database: {db_stats.get('total_rules', 0)} rules")
                
                return True
            else:
                logger.error("Training failed during extraction")
                return False
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Don't cleanup immediately - let user see results
            logger.info("Training phase completed")
    
    def analyze_new_project(self, project_path: str, output_file: str = None) -> dict:
        """Enhanced analysis"""
        if not self._initialized:
            return {}
            
        try:
            logger.info(f"Analyzing: {project_path}")
            
            from src.utils.file_utils import discover_python_files
            python_files = discover_python_files(project_path, self.config.analysis.ignore_patterns)
            
            if not python_files:
                logger.error("No Python files found")
                return {}
            
            # Check rules
            db_stats = self.rule_database.get_statistics()
            if db_stats.get('total_rules', 0) == 0:
                logger.error("No rules found. Run training first.")
                return {}
            
            # Analyze
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
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved: {output_file}")
            
            logger.info("Analysis completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.llm_client:
                self.llm_client.cleanup()
            if self.rule_database:
                self.rule_database.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Python Code Convention Analyzer")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    train_parser = subparsers.add_parser('train', help='Train on accepted codebase')
    train_parser.add_argument('project_path', help='Path to accepted project')
    train_parser.add_argument('--config', default='config/config.yaml')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze new project')
    analyze_parser.add_argument('project_path', help='Path to project to analyze')
    analyze_parser.add_argument('--output', default='./analysis_report.json')
    analyze_parser.add_argument('--config', default='config/config.yaml')
    
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check imports
    if not safe_import():
        logger.error("Failed to import modules")
        sys.exit(1)
    
    try:
        if args.command == 'train':
            analyzer = ConventionAnalyzer(getattr(args, 'config', 'config/config.yaml'))
            
            if analyzer.initialize():
                success = analyzer.train_on_accepted_codebase(args.project_path)
                
                if success:
                    print("\n🎉 Training completed successfully!")
                    print("Next: python main.py analyze <project_path>")
                else:
                    print("\n❌ Training failed!")
                    
                # Cleanup after showing results
                analyzer.cleanup()
                sys.exit(0 if success else 1)
            else:
                print("\n❌ Initialization failed!")
                sys.exit(1)
        
        elif args.command == 'analyze':
            analyzer = ConventionAnalyzer(getattr(args, 'config', 'config/config.yaml'))
            
            if analyzer.initialize():
                report = analyzer.analyze_new_project(args.project_path, args.output)
                
                if report:
                    print("\n🎉 Analysis completed successfully!")
                else:
                    print("\n❌ Analysis failed!")
                    
                analyzer.cleanup()
                sys.exit(0 if report else 1)
            else:
                print("\n❌ Initialization failed!")
                sys.exit(1)
        
        elif args.command == 'status':
            print("System Status:")
            
            # Check database
            db_path = Path("data/databases/rules.db")
            if db_path.exists():
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conventions")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"  Database: {count} conventions")
            else:
                print("  Database: Not found")
            
            # Check model
            from src.models.config_models import load_config
            config = load_config()
            model_path = Path(config.llm.model_path)
            if model_path.exists():
                size_gb = model_path.stat().st_size / (1024**3)
                print(f"  Model: OK ({size_gb:.1f}GB)")
            else:
                print("  Model: Not found")
            
            print("  Status: Ready" if db_path.exists() and model_path.exists() else "  Status: Not ready")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
