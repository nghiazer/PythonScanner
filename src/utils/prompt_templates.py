"""
Prompt templates for code convention analysis
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PromptTemplates:
    """Prompt templates for LLAMA 3 8B"""
    
    def __init__(self):
        self.max_code_length = 2000
        
    def create_extraction_prompt(self, code: str) -> str:
        """Create convention extraction prompt"""
        if len(code) > self.max_code_length:
            code = code[:self.max_code_length] + "\n# ... (truncated)"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Python code analyzer. Extract specific coding conventions from the code.

Return ONLY valid JSON array with conventions found.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this Python code and extract conventions:

```python
{code}
```

Return JSON array of conventions:
```json
[
  {{
    "convention_type": "naming",
    "pattern": "snake_case_functions",
    "rule_description": "Functions use snake_case naming",
    "example": "def calculate_average():",
    "confidence": 0.95
  }}
]
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```json"""
        
        return prompt
    
    def create_checking_prompt(self, code: str, rules: List[Dict]) -> str:
        """Create convention checking prompt"""
        if len(code) > self.max_code_length:
            code = code[:self.max_code_length] + "\n# ... (truncated)"
        
        # Limit rules
        if len(rules) > 5:
            rules = rules[:5]
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Python code reviewer. Check code against conventions and identify violations.

Return ONLY valid JSON format.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Check this code against conventions:

CODE:
```python
{code}
```

RULES:
{rules}

Return JSON with violations found:
```json
{{
  "violations": [
    {{
      "rule_type": "naming",
      "line": 5,
      "message": "Function should use snake_case",
      "severity": "medium",
      "current": "badName",
      "suggested": "bad_name"
    }}
  ],
  "suggestions": ["Add type hints"],
  "compliance_score": 0.75
}}
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```json"""
        
        return prompt
