# Installation Guide

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
venv\Scripts\activate   # Windows

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
