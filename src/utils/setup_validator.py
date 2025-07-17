"""
Setup validation utilities
"""
import logging
import sys
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

class SetupValidator:
    """Validate system setup"""
    
    def validate_complete_setup(self) -> bool:
        """Validate complete setup"""
        logger.info("Validating setup...")
        
        checks = [
            self._check_python_version(),
            self._check_dependencies(),
            self._check_directories(),
            self._check_configuration(),
            self._check_model()
        ]
        
        return all(checks)
    
    def _check_python_version(self) -> bool:
        """Check Python version"""
        if sys.version_info < (3, 9):
            logger.error("Python 3.9+ required")
            return False
        logger.info("✅ Python version OK")
        return True
    
    def _check_dependencies(self) -> bool:
        """Check dependencies"""
        try:
            import yaml
            import psutil
            logger.info("✅ Dependencies OK")
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            return False
    
    def _check_directories(self) -> bool:
        """Check directories"""
        required_dirs = ["data", "config", "logs"]
        
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                logger.error(f"❌ Missing directory: {dir_name}")
                return False
        
        logger.info("✅ Directories OK")
        return True
    
    def _check_configuration(self) -> bool:
        """Check configuration"""
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            logger.error("❌ Configuration file missing")
            return False
        
        logger.info("✅ Configuration OK")
        return True
    
    def _check_model(self) -> bool:
        """Check model file"""
        try:
            from src.models.config_models import load_config
            config = load_config()
            
            model_path = Path(config.llm.model_path)
            if not model_path.exists():
                logger.error(f"❌ Model file missing: {model_path}")
                return False
            
            logger.info("✅ Model file OK")
            return True
        
        except Exception as e:
            logger.error(f"❌ Model check failed: {e}")
            return False
