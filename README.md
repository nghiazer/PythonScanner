# Python Code Convention Analyzer

AI-powered Python code convention analyzer using LLAMA 3 8B offline model.

## Quick Start

1. **Setup Environment**
```bash
python scripts/setup_environment.py --download-model
```

2. **Train on Accepted Codebase**
```bash
python main.py train examples/accepted_project
```

3. **Analyze New Project**
```bash
python main.py analyze examples/test_project --output report.json
```

## System Requirements

- Python 3.9+
- 16GB RAM minimum
- 50GB storage
- Modern CPU

## Features

- 🤖 AI-powered with LLAMA 3 8B
- 🔒 Completely offline
- 📊 Detailed reports
- ⚡ Optimized performance

## Commands

- `train PROJECT_PATH` - Train on accepted codebase
- `analyze PROJECT_PATH` - Analyze new project
- `status` - Show system status

## Configuration

Edit `config/config.yaml` to customize settings.
