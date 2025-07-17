#!/usr/bin/env python3
"""
Enhanced setup script
"""
import sys
import subprocess
import logging
from pathlib import Path

def setup_environment():
    """Setup environment"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("Python 3.9+ required")
        return False
    
    # Install dependencies
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        logger.info("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        logger.error("❌ Failed to install dependencies")
        return False
    
    # Create directories
    directories = ["data/models", "logs", "reports"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Environment setup complete")
    logger.info("Next steps:")
    logger.info("1. Download model: python scripts/download_model.py")
    logger.info("2. Train: python main.py train examples/accepted_project")
    logger.info("3. Analyze: python main.py analyze examples/test_project")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
