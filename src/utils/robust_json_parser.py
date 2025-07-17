"""
Robust JSON parser for LLM responses
"""

import json
import re
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

class RobustJSONParser:
    """Robust JSON parser that handles malformed LLM responses"""
    
    def __init__(self):
        self.fix_attempts = 0
        self.max_fix_attempts = 5
    
    def parse_json_response(self, text: str) -> Optional[Any]:
        """Parse JSON with multiple fallback strategies"""
        # Strategy 1: Direct parsing
        json_text = self._extract_json_from_text(text)
        if json_text:
            result = self._try_parse_json(json_text)
            if result is not None:
                return result
        
        # Strategy 2: Fix and parse
        if json_text:
            fixed_json = self._fix_malformed_json(json_text)
            if fixed_json:
                result = self._try_parse_json(fixed_json)
                if result is not None:
                    return result
        
        # Strategy 3: Extract partial valid JSON
        partial_json = self._extract_partial_json(text)
        if partial_json:
            result = self._try_parse_json(partial_json)
            if result is not None:
                return result
        
        logger.error("All JSON parsing strategies failed")
        return None
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text using multiple patterns"""
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # JSON code blocks
            r'```\s*([\s\S]*?)\s*```',      # Generic code blocks
            r'(\[[^\[\]]*(?:\{[^\{\}]*\}[^\[\]]*)*\])',  # Array patterns
            r'(\{[^\{\}]*(?:\[[^\[\]]*\][^\{\}]*)*\})',  # Object patterns
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Return the longest match
                return max(matches, key=len)
        
        # Fallback: look for JSON-like structures
        json_start = text.find('{')
        json_array_start = text.find('[')
        
        if json_start != -1 and (json_array_start == -1 or json_start < json_array_start):
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[json_start:], json_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[json_start:i+1]
        
        elif json_array_start != -1:
            # Find matching closing bracket
            depth = 0
            for i, char in enumerate(text[json_array_start:], json_array_start):
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        return text[json_array_start:i+1]
        
        return None
    
    def _try_parse_json(self, json_text: str) -> Optional[Any]:
        """Try to parse JSON with error handling"""
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None
    
    def _fix_malformed_json(self, json_text: str) -> Optional[str]:
        """Fix common JSON malformation issues"""
        if self.fix_attempts >= self.max_fix_attempts:
            return None
        
        self.fix_attempts += 1
        original_text = json_text
        
        # Remove any text before first { or [
        json_text = re.sub(r'^[^\{\[]*', '', json_text)
        
        # Remove any text after last } or ]
        json_text = re.sub(r'[^\}\]]*$', '', json_text)
        
        # Fix trailing commas
        json_text = re.sub(r',\s*([\}\]])', r'\1', json_text)
        
        # Fix missing quotes around keys
        json_text = re.sub(r'([\{\s,])(\w+):', r'\1"\2":', json_text)
        
        # Fix single quotes to double quotes
        json_text = re.sub(r"'([^']*)'", r'"\1"', json_text)
        
        # Fix unescaped quotes in strings
        json_text = re.sub(r'(["\'"])((?:\\.|[^\\"])*?)(?<!\\)(["\'"])', r'\1\2\\\3', json_text)
        
        # Fix malformed arrays/objects
        json_text = re.sub(r'\}\s*\{', r'},{', json_text)
        json_text = re.sub(r'\]\s*\[', r'],[', json_text)
        
        return json_text if json_text != original_text else None
    
    def _extract_partial_json(self, text: str) -> Optional[str]:
        """Extract partial valid JSON from text"""
        # Look for convention-like patterns and build JSON
        patterns = []
        
        # Find convention patterns
        conv_matches = re.findall(r'(naming|structure|style|best_practice|docstring|import)[^\n]*', text, re.IGNORECASE)
        
        if conv_matches:
            # Build minimal valid JSON
            conventions = []
            for match in conv_matches[:3]:  # Limit to 3
                conventions.append({
                    "convention_type": "naming",
                    "pattern": "extracted_pattern",
                    "rule_description": match,
                    "example": "",
                    "confidence": 0.7
                })
            
            try:
                return json.dumps(conventions)
            except:
                pass
        
        return None

# Global parser instance
json_parser = RobustJSONParser()
