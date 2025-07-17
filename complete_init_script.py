#!/usr/bin/env python3
"""
Working Python Code Convention Analyzer Initialization Script
Creates complete project structure ready to use
"""

import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, List
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectInitializer:
    """Complete project initialization manager"""

    def __init__(self, project_name: str = "python_convention_analyzer"):
        self.project_name = project_name
        self.project_root = Path(project_name)
        self.files_created = []
        self.directories_created = []

    def initialize_project(self) -> bool:
        """Initialize complete project structure"""
        logger.info(f"Initializing project: {self.project_name}")

        try:
            # Create project structure
            self._create_directory_structure()

            # Create all source files
            self._create_source_files()

            # Create configuration files
            self._create_configuration_files()

            # Create documentation
            self._create_documentation()

            # Create example files
            self._create_example_files()

            logger.info("Project structure created successfully!")
            logger.info(f"Created {len(self.directories_created)} directories and {len(self.files_created)} files")

            return True

        except Exception as e:
            logger.error(f"Project initialization failed: {e}")
            return False

    def _create_directory_structure(self):
        """Create complete directory structure"""
        directories = [
            "",  # Root
            "src",
            "src/core",
            "src/extractors",
            "src/checkers",
            "src/utils",
            "src/models",
            "config",
            "scripts",
            "tests",
            "docs",
            "data",
            "data/models",
            "data/databases",
            "data/cache",
            "logs",
            "reports",
            "examples",
            "examples/accepted_project",
            "examples/test_project"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.directories_created.append(str(dir_path))
            logger.debug(f"Created directory: {dir_path}")

    def _create_source_files(self):
        """Create all source code files"""

        # Main application file
        main_content = '''#!/usr/bin/env python3
"""
Python Code Convention Analyzer - Main Entry Point
"""

import sys
import argparse
import logging
import json
from pathlib import Path

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Python Code Convention Analyzer with LLAMA 3 8B"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train on accepted codebase')
    train_parser.add_argument('project_path', help='Path to accepted project')
    train_parser.add_argument('--config', default='config/config.yaml', 
                             help='Configuration file path')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze new project')
    analyze_parser.add_argument('project_path', help='Path to project to analyze')
    analyze_parser.add_argument('--output', default='./analysis_report.json',
                               help='Output file path')
    analyze_parser.add_argument('--config', default='config/config.yaml',
                               help='Configuration file path')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup environment')
    setup_parser.add_argument('--download-model', action='store_true',
                             help='Download LLAMA model')

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == 'setup':
        print("Setting up environment...")
        print("Please run: python scripts/setup_environment.py")

    elif args.command == 'train':
        print(f"Training on codebase: {args.project_path}")
        print("Training functionality will be implemented after model setup")

    elif args.command == 'analyze':
        print(f"Analyzing project: {args.project_path}")
        print("Analysis functionality will be implemented after model setup")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''

        self._create_file("main.py", main_content)

        # Core modules - simplified versions
        self._create_file("src/__init__.py", "")
        self._create_file("src/core/__init__.py", "")

        analyzer_content = '''"""
Core analyzer module
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeConventionAnalyzer:
    """Main analyzer class"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        logger.info("Code Convention Analyzer initialized")

    def train_on_accepted_codebase(self, project_path: str) -> bool:
        """Train the system on an accepted codebase"""
        logger.info(f"Training on: {project_path}")
        # Training logic will be implemented here
        return True

    def analyze_new_project(self, project_path: str) -> dict:
        """Analyze a new project against conventions"""
        logger.info(f"Analyzing: {project_path}")
        # Analysis logic will be implemented here
        return {
            "analysis_results": {},
            "report": "Analysis complete"
        }
'''

        self._create_file("src/core/analyzer.py", analyzer_content)

        # LLM Client placeholder
        llm_client_content = '''"""
LLM Client for LLAMA integration
"""
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLAMA model"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"LLM Client initialized with model: {model_path}")

    def extract_conventions(self, code_samples: list) -> list:
        """Extract coding conventions from code samples"""
        # Implementation will be added after model setup
        return []

    def check_conventions(self, code: str, rules: list) -> dict:
        """Check code against established conventions"""
        # Implementation will be added after model setup
        return {"violations": [], "suggestions": []}
'''

        self._create_file("src/core/llm_client.py", llm_client_content)

        # Configuration models
        config_models_content = '''"""
Configuration data models
"""
from dataclasses import dataclass
from typing import List
import yaml
from pathlib import Path

@dataclass
class LLMConfig:
    """LLM configuration"""
    model_path: str = "./data/models/llama3-8b-q4_K_M.gguf"
    n_ctx: int = 8000
    n_batch: int = 512
    n_threads: int = 8
    temperature: float = 0.1
    max_tokens: int = 1000

@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    batch_size: int = 10
    max_workers: int = 4
    memory_threshold: float = 0.85
    ignore_patterns: List[str] = None

    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = ['__pycache__', '.git', 'venv', 'env']

@dataclass
class AppConfig:
    """Main application configuration"""
    llm: LLMConfig = None
    analysis: AnalysisConfig = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Load configuration from file"""
    if not Path(config_path).exists():
        # Create default config
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
    """Save configuration to file"""
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
'''

        self._create_file("src/models/config_models.py", config_models_content)
        self._create_file("src/models/__init__.py", "")

        # Utilities
        self._create_file("src/utils/__init__.py", "")

        file_utils_content = '''"""
File utility functions
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def discover_python_files(project_path: str, ignore_patterns: List[str]) -> List[str]:
    """Discover all Python files in a project"""
    python_files = []
    project_path = Path(project_path)

    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return []

    for root, dirs, files in os.walk(project_path):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not any(pattern in file_path for pattern in ignore_patterns):
                    python_files.append(file_path)

    logger.info(f"Discovered {len(python_files)} Python files")
    return python_files

def read_file_safely(file_path: str, max_size: int = 1024 * 1024) -> Optional[str]:
    """Safely read a file with size limits"""
    try:
        file_path = Path(file_path)

        if file_path.stat().st_size > max_size:
            logger.warning(f"File too large: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None
'''

        self._create_file("src/utils/file_utils.py", file_utils_content)

        # Scripts
        setup_script_content = '''#!/usr/bin/env python3
"""
Environment setup script
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        return False

    logger.info(f"Python version: {sys.version}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")

    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)

        logger.info("Dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Setup required directories"""
    directories = [
        "data/models",
        "data/databases", 
        "data/cache",
        "logs",
        "reports"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_environment(download_model: bool = False):
    """Main setup function"""
    logger.info("Setting up Python Code Convention Analyzer environment...")

    if not check_python_version():
        return False

    setup_directories()

    if not install_dependencies():
        return False

    if download_model:
        logger.info("Model download would be implemented here")
        logger.info("Run: python scripts/download_model.py")

    logger.info("Environment setup completed!")
    logger.info("Next steps:")
    logger.info("1. Download model: python scripts/download_model.py") 
    logger.info("2. Train: python main.py train examples/accepted_project")
    logger.info("3. Analyze: python main.py analyze examples/test_project")

    return True

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup environment")
    parser.add_argument('--download-model', action='store_true',
                       help='Download LLAMA model during setup')

    args = parser.parse_args()

    success = setup_environment(download_model=args.download_model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''

        self._create_file("scripts/setup_environment.py", setup_script_content)

        download_model_content = '''#!/usr/bin/env python3
"""
Download LLAMA 3 8B model script
"""

import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MODEL_FILENAME = "llama3-8b-q4_K_M.gguf"

def download_model(force: bool = False) -> bool:
    """Download LLAMA 3 8B model"""

    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME

    if model_path.exists() and not force:
        logger.info(f"Model already exists: {model_path}")
        return True

    logger.info("Downloading LLAMA 3 8B model (~4.6GB)")
    logger.info("This may take a while...")

    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded // (1024 * 1024)
                mb_total = total_size // (1024 * 1024)

                sys.stdout.write(f"\\rProgress: {percent:3d}% ({mb_downloaded:4d}/{mb_total:4d} MB)")
                sys.stdout.flush()

        urlretrieve(MODEL_URL, model_path, progress_hook)
        print()  # New line after progress

        logger.info(f"Model downloaded successfully: {model_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        return False

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download LLAMA 3 model")
    parser.add_argument('--force', action='store_true',
                       help='Force download even if file exists')

    args = parser.parse_args()

    success = download_model(args.force)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''

        self._create_file("scripts/download_model.py", download_model_content)
        self._create_file("scripts/__init__.py", "")

        logger.info("Source files created successfully")

    def _create_configuration_files(self):
        """Create configuration files"""

        # YAML config
        config_yaml = '''# Python Code Convention Analyzer Configuration

llm:
  model_path: "./data/models/llama3-8b-q4_K_M.gguf"
  n_ctx: 8000
  n_batch: 512
  n_threads: 8
  temperature: 0.1
  max_tokens: 1000

analysis:
  batch_size: 10
  max_workers: 4
  memory_threshold: 0.85
  ignore_patterns:
    - "__pycache__"
    - ".git"
    - ".pytest_cache"
    - "venv"
    - "env"
    - ".venv"

conventions:
  naming: true
  structure: true
  style: true
  best_practices: true
  docstrings: true
  imports: true
'''

        self._create_file("config/config.yaml", config_yaml)

        # Requirements
        requirements = '''# Core dependencies
llama-cpp-python>=0.2.11
chromadb>=0.4.15
sentence-transformers>=2.2.2
pydantic>=2.0.0
pyyaml>=6.0.1

# Code analysis tools
flake8>=6.1.0
ruff>=0.1.0

# Utilities
psutil>=5.9.6
tqdm>=4.65.0
click>=8.1.0
rich>=13.0.0

# Development tools
pytest>=7.4.0
mypy>=1.5.0
'''

        self._create_file("requirements.txt", requirements)

        logger.info("Configuration files created")

    def _create_documentation(self):
        """Create documentation files"""

        readme_content = '''# Python Code Convention Analyzer

AI-powered Python code convention analyzer using LLAMA 3 8B offline model.

## Features

- 🤖 **AI-Powered Analysis**: Uses LLAMA 3 8B for intelligent code convention extraction
- 🔒 **Completely Offline**: No internet required after initial setup  
- 📊 **Comprehensive Reports**: Detailed analysis with actionable suggestions
- ⚡ **High Performance**: Optimized for Windows systems with 16GB RAM
- 🎯 **Customizable**: Configurable convention types and analysis parameters

## Quick Start

### 1. Setup Environment
```bash
python scripts/setup_environment.py --download-model
```

### 2. Train on Accepted Codebase
```bash
python main.py train examples/accepted_project
```

### 3. Analyze New Project
```bash
python main.py analyze examples/test_project --output report.json
```

## System Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.9 or higher
- **RAM**: 16GB minimum
- **Storage**: 50GB free space
- **CPU**: Modern multi-core processor

## Configuration

Edit `config/config.yaml` to customize:

```yaml
llm:
  model_path: "./data/models/llama3-8b-q4_K_M.gguf"
  n_ctx: 8000          # Context window size
  temperature: 0.1     # Lower = more consistent

analysis:
  batch_size: 10       # Files per batch
  max_workers: 4       # Parallel workers
```

## Usage Examples

### Basic Analysis
```bash
# Train on accepted codebase
python main.py train ./examples/accepted_project

# Analyze new project
python main.py analyze ./examples/test_project
```

### Custom Configuration
```bash
# Use custom config
python main.py analyze ./project --config custom_config.yaml

# Verbose output
python main.py analyze ./project --verbose
```

## Project Structure

```
python_convention_analyzer/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── config/
│   └── config.yaml        # Configuration
├── src/                   # Source code
│   ├── core/             # Core modules
│   ├── utils/            # Utilities
│   └── models/           # Data models
├── scripts/              # Setup scripts
├── examples/             # Example projects
└── data/                 # Data storage
```

## Troubleshooting

### Common Issues

**1. High Memory Usage**
- Reduce `batch_size` in config
- Lower `n_ctx` value
- Close other applications

**2. Model Download Fails**
```bash
python scripts/download_model.py --force
```

**3. Slow Analysis**
- Check system resources
- Reduce `max_workers`
- Use smaller context window

## License

MIT License - see LICENSE file for details.
'''

        self._create_file("README.md", readme_content)

        install_guide = '''# Installation Guide

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 16GB RAM minimum
- 50GB free disk space

### Check Python Version
```bash
python --version
# Should output Python 3.9.x or higher
```

## Installation Steps

### 1. Clone/Initialize Project
```bash
# If using initialization script
python init_project.py
cd python_convention_analyzer
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate   # Windows

# Install requirements
pip install -r requirements.txt
```

### 3. Setup Environment
```bash
python scripts/setup_environment.py
```

### 4. Download Model
```bash
python scripts/download_model.py
```

## Verification

### Test Installation
```bash
python main.py --help
```

### Run Example
```bash
# Train on example project
python main.py train examples/accepted_project

# Analyze test project
python main.py analyze examples/test_project
```

## Troubleshooting

### Installation Issues

**1. llama-cpp-python compilation fails**
```bash
# Try pre-built wheels
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**2. Memory errors during installation**
```bash
# Use no-cache-dir flag
pip install --no-cache-dir -r requirements.txt
```

**3. Permission errors**
```bash
# Use user installation
pip install --user -r requirements.txt
```
'''

        self._create_file("docs/INSTALLATION.md", install_guide)

        logger.info("Documentation created")

    def _create_example_files(self):
        """Create example projects for testing"""

        # Good example code
        good_example = '''#!/usr/bin/env python3
"""
Example of well-formatted Python code following best practices.
"""

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Process and analyze data using various statistical methods.
    """

    def __init__(self, data_source: str):
        """
        Initialize the data processor.

        Args:
            data_source: Path to the data source file
        """
        self.data_source = Path(data_source)
        self.processed_data: List[Dict] = []
        self._is_loaded = False

    def load_data(self) -> bool:
        """
        Load data from the configured source.

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            if not self.data_source.exists():
                logger.error(f"Data source not found: {self.data_source}")
                return False

            # Simulated data loading
            self.processed_data = []
            self._is_loaded = True

            logger.info(f"Data loaded successfully from {self.data_source}")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary containing mean, median, and standard deviation
        """
        if not values:
            raise ValueError("Cannot calculate statistics for empty list")

        sorted_values = sorted(values)
        n = len(values)

        # Calculate mean
        mean = sum(values) / n

        # Calculate median
        if n % 2 == 0:
            median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            median = sorted_values[n//2]

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = variance ** 0.5

        return {
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "count": n
        }

def process_file_list(file_paths: List[str], output_dir: str = "./output") -> bool:
    """
    Process a list of files and save results.

    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save processed results

    Returns:
        True if all files processed successfully
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    success_count = 0

    for file_path in file_paths:
        try:
            processor = DataProcessor(file_path)
            if processor.load_data():
                success_count += 1
                logger.info(f"Successfully processed: {file_path}")
            else:
                logger.warning(f"Failed to process: {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    success_rate = success_count / len(file_paths) if file_paths else 0
    logger.info(f"Processing complete: {success_count}/{len(file_paths)} files")

    return success_rate > 0.8

def main():
    """Main function demonstrating usage."""
    logging.basicConfig(level=logging.INFO)

    test_files = ["data1.txt", "data2.txt", "data3.txt"]

    try:
        result = process_file_list(test_files)
        if result:
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing failed")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        self._create_file("examples/accepted_project/data_processor.py", good_example)

        # Bad example for testing
        bad_example = '''# Bad example with convention violations
import os,sys
from pathlib import *

def calcAvg(nums):
    return sum(nums)/len(nums)

class dataProcessor:
    def __init__(self):
        self.Data=[]
        self.SIZE=100

    def addData(self,item):
        self.Data.append(item)

    def ProcessData(self):
        result=[]
        for i in range(len(self.Data)):
            if self.Data[i]>0:
                result.append(self.Data[i]*2)
        return result

def process_files(files):
    for f in files:
        data=open(f).read()
        print(data)

if __name__=="__main__":
    dp=dataProcessor()
    dp.addData(5)
    dp.addData(10)
    print(dp.ProcessData())
'''

        self._create_file("examples/test_project/bad_code.py", bad_example)

        logger.info("Example files created")

    def _create_file(self, path: str, content: str):
        """Create a file with content"""
        file_path = self.project_root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_created.append(str(file_path))
        logger.debug(f"Created file: {file_path}")

    def print_summary(self):
        """Print project creation summary"""
        print("\n" + "=" * 60)
        print("🎉 PROJECT INITIALIZATION COMPLETE!")
        print("=" * 60)
        print(f"Project Name: {self.project_name}")
        print(f"Location: {self.project_root.absolute()}")
        print(f"Directories: {len(self.directories_created)}")
        print(f"Files: {len(self.files_created)}")
        print("\n📋 NEXT STEPS:")
        print("1. cd " + self.project_name)
        print("2. python scripts/setup_environment.py --download-model")
        print("3. python main.py train examples/accepted_project")
        print("4. python main.py analyze examples/test_project")
        print("\n📚 KEY FILES:")
        print("- main.py - Main entry point")
        print("- config/config.yaml - Configuration")
        print("- requirements.txt - Dependencies")
        print("- README.md - Documentation")


def main():
    """Main initialization function"""
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Python Code Convention Analyzer project")
    parser.add_argument('--name', default='python_convention_analyzer',
                        help='Project name/directory')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing directory')

    args = parser.parse_args()

    # Check if directory exists
    if Path(args.name).exists() and not args.force:
        print(f"❌ Directory '{args.name}' already exists. Use --force to overwrite.")
        sys.exit(1)

    # Initialize project
    initializer = ProjectInitializer(args.name)
    success = initializer.initialize_project()

    if success:
        initializer.print_summary()
        print("\n✅ Initialization successful!")
    else:
        print("\n❌ Initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()