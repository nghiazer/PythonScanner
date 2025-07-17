"""
Configuration data models
"""
from dataclasses import dataclass
from typing import List
import yaml
from pathlib import Path

@dataclass
class LLMConfig:
    model_path: str = "./data/models/llama3-8b-q4_K_M.gguf"
    n_ctx: int = 8000
    n_batch: int = 512
    n_threads: int = 8
    temperature: float = 0.1
    max_tokens: int = 1000

@dataclass
class AnalysisConfig:
    batch_size: int = 10
    max_workers: int = 4
    memory_threshold: float = 0.85
    ignore_patterns: List[str] = None
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = ['__pycache__', '.git', 'venv', 'env']

@dataclass
class AppConfig:
    llm: LLMConfig = None
    analysis: AnalysisConfig = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Load configuration"""
    if not Path(config_path).exists():
        config = AppConfig()
        save_config(config, config_path)
        return config
    
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return AppConfig(
            llm=LLMConfig(**data.get('llm', {})),
            analysis=AnalysisConfig(**data.get('analysis', {}))
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        return AppConfig()

def save_config(config: AppConfig, config_path: str):
    """Save configuration"""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'llm': {
            'model_path': config.llm.model_path,
            'n_ctx': config.llm.n_ctx,
            'n_batch': config.llm.n_batch,
            'n_threads': config.llm.n_threads,
            'temperature': config.llm.temperature,
            'max_tokens': config.llm.max_tokens
        },
        'analysis': {
            'batch_size': config.analysis.batch_size,
            'max_workers': config.analysis.max_workers,
            'memory_threshold': config.analysis.memory_threshold,
            'ignore_patterns': config.analysis.ignore_patterns
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)
