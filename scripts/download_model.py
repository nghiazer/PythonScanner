#!/usr/bin/env python3
"""
Download LLAMA 3 8B model
"""
import sys
import logging
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MODEL_FILENAME = "llama3-8b-q4_K_M.gguf"

def download_model():
    """Download model"""
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    
    if model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        return True
    
    logger.info("Downloading LLAMA 3 8B model (~4.6GB)...")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\rProgress: {percent:3d}%")
                sys.stdout.flush()
        
        urlretrieve(MODEL_URL, model_path, progress_hook)
        print()  # New line
        
        logger.info(f"✅ Model downloaded: {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
