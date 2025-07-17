"""
Working prompt templates based on debug results
Simple format without chat templates
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PromptTemplates:
    """Working prompt templates - simple format"""
    
    def __init__(self):
        self.max_code_length = 800  # Further reduced
        
    def create_extraction_prompt(self, code: str) -> str:
        """Create simple extraction prompt that works"""
        if len(code) > self.max_code_length:
            code = code[:self.max_code_length] + "\n# ... (truncated)"
        
        # Simple format without chat template
        prompt = f"""Look at this Python code:

```python
{code}
```

What coding conventions do you see? Return as JSON array:

[
  {{"convention_type": "naming", "pattern": "snake_case", "rule_description": "Functions use snake_case naming", "example": "def my_func():", "confidence": 0.9}},
  {{"convention_type": "docstring", "pattern": "function_docs", "rule_description": "Functions have docstrings", "example": "\"\"\"Function description\"\"\"", "confidence": 0.8}}
]

JSON:"""
        
        return prompt
    
    def create_checking_prompt(self, code: str, rules: List[Dict]) -> str:
        """Create simple checking prompt"""
        if len(code) > self.max_code_length:
            code = code[:self.max_code_length] + "\n# ... (truncated)"
        
        rules_text = "\n".join([f"- {rule['rule_description']}" for rule in rules[:2]])
        
        prompt = f"""Check this Python code against these rules:

{rules_text}

Code:
```python
{code}
```

Find violations and return JSON:
{{"violations": [], "suggestions": [], "compliance_score": 0.8}}

JSON:"""
        
        return prompt
