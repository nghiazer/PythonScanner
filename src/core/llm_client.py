"""
Working LLM client with proven parameters
"""

import logging
import json
import re
import time
import gc
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

from src.utils.memory_monitor import MemoryMonitor
from src.utils.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class LLMClient:
    """Working LLM client with proven parameters"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        self.model_path = Path(model_path)
        self.config = self._get_working_config()
        if config:
            self.config.update(config)
        
        self.memory_monitor = MemoryMonitor(threshold=0.85)
        self.prompt_templates = PromptTemplates()
        self.llm = None
        self._model_loaded = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "memory_cleanups": 0
        }
        
        logger.info(f"LLM Client initialized with model: {self.model_path}")
    
    def _get_working_config(self) -> Dict:
        """Get working configuration based on debug results"""
        return {
            "n_ctx": 4096,          # Reduced context
            "n_batch": 256,         # Reduced batch
            "n_threads": 4,         # Reduced threads
            "temperature": 0.3,     # Higher temperature (from debug)
            "top_p": 0.8,          # From debug results
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 500,      # Reduced max tokens
            "mmap": True,
            "mlock": True,
            "verbose": False,
            "seed": -1             # Random seed
        }
    
    def load_model(self) -> bool:
        """Load model with working config"""
        if not LLAMA_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return False
        
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            logger.info("Loading LLAMA model with working config...")
            start_time = time.time()
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config["n_ctx"],
                n_batch=self.config["n_batch"],
                n_threads=self.config["n_threads"],
                verbose=self.config["verbose"],
                mmap=self.config["mmap"],
                mlock=self.config["mlock"],
                seed=self.config["seed"]
            )
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def extract_conventions(self, code_samples: List[str]) -> List[Dict]:
        """Extract conventions with working approach"""
        if not self._model_loaded:
            logger.error("Model not loaded")
            return []
        
        logger.info(f"Extracting conventions from {len(code_samples)} code samples")
        
        all_conventions = []
        
        for i, code_sample in enumerate(code_samples):
            try:
                logger.debug(f"Processing sample {i+1}/{len(code_samples)}")
                
                # Generate prompt
                prompt = self.prompt_templates.create_extraction_prompt(code_sample)
                
                # Get response with working parameters
                response_text = self._generate_response(prompt)
                
                if response_text:
                    # Parse conventions with robust approach
                    conventions = self._parse_extraction_response(response_text)
                    
                    # Add metadata
                    for conv in conventions:
                        conv["sample_index"] = i
                        conv["extraction_timestamp"] = time.time()
                    
                    all_conventions.extend(conventions)
                    logger.debug(f"Extracted {len(conventions)} conventions from sample {i+1}")
                else:
                    logger.warning(f"No response for sample {i+1}")
                    
            except Exception as e:
                logger.error(f"Failed to process sample {i+1}: {e}")
                continue
        
        logger.info(f"Total conventions extracted: {len(all_conventions)}")
        return all_conventions
    
    def check_conventions(self, code: str, rules: List[Dict]) -> Dict:
        """Check conventions with working approach"""
        if not self._model_loaded:
            logger.error("Model not loaded")
            return {"violations": [], "suggestions": [], "compliance_score": 0.0}
        
        try:
            # Generate prompt
            prompt = self.prompt_templates.create_checking_prompt(code, rules)
            
            # Get response
            response_text = self._generate_response(prompt)
            
            if response_text:
                return self._parse_checking_response(response_text)
            else:
                logger.warning("No response from model")
                
        except Exception as e:
            logger.error(f"Failed to check conventions: {e}")
            
        return {"violations": [], "suggestions": [], "compliance_score": 0.0}
    
    def _generate_response(self, prompt: str) -> Optional[str]:
        """Generate response with working parameters"""
        if not self._model_loaded:
            logger.error("Model not loaded")
            return None
        
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Use working config values
            response = self.llm(
                prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"],
                repeat_penalty=self.config["repeat_penalty"],
                echo=False,
                stop=["\n\n\n", "Human:", "User:"]  # Better stop sequences
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self._update_avg_response_time(response_time)
            
            logger.debug(f"Generated response in {response_time:.2f}s")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            self.stats["failed_requests"] += 1
            return None
    
    def _parse_extraction_response(self, response_text: str) -> List[Dict]:
        """Parse extraction response with robust approach"""
        conventions = []
        
        try:
            logger.debug(f"Parsing response: {response_text[:200]}...")
            
            # Extract JSON from response
            json_text = self._extract_json_from_text(response_text)
            
            if not json_text:
                logger.warning("No JSON found in response")
                # Fallback: create basic conventions from text
                return self._create_fallback_conventions(response_text)
            
            # Parse JSON
            try:
                parsed_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                # Try to fix JSON
                fixed_json = self._fix_json(json_text)
                if fixed_json:
                    try:
                        parsed_data = json.loads(fixed_json)
                    except:
                        return self._create_fallback_conventions(response_text)
                else:
                    return self._create_fallback_conventions(response_text)
            
            # Handle different formats
            if isinstance(parsed_data, list):
                conventions_data = parsed_data
            elif isinstance(parsed_data, dict) and "conventions" in parsed_data:
                conventions_data = parsed_data["conventions"]
            else:
                conventions_data = [parsed_data] if isinstance(parsed_data, dict) else []
            
            # Validate conventions
            for conv_data in conventions_data:
                conv = self._validate_convention(conv_data)
                if conv:
                    conventions.append(conv)
            
            logger.debug(f"Successfully parsed {len(conventions)} conventions")
            return conventions
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return self._create_fallback_conventions(response_text)
    
    def _create_fallback_conventions(self, response_text: str) -> List[Dict]:
        """Create fallback conventions from text"""
        conventions = []
        
        # Basic pattern matching
        if "snake_case" in response_text.lower():
            conventions.append({
                "convention_type": "naming",
                "pattern": "snake_case",
                "rule_description": "Uses snake_case naming convention",
                "example": "def my_function():",
                "confidence": 0.7
            })
        
        if "docstring" in response_text.lower():
            conventions.append({
                "convention_type": "docstring",
                "pattern": "function_docstrings",
                "rule_description": "Functions have docstrings",
                "example": "\"\"\"Function description\"\"\"",
                "confidence": 0.7
            })
        
        if "class" in response_text.lower() and "pascal" in response_text.lower():
            conventions.append({
                "convention_type": "naming",
                "pattern": "pascal_case_classes",
                "rule_description": "Classes use PascalCase naming",
                "example": "class MyClass:",
                "confidence": 0.7
            })
        
        return conventions
    
    def _parse_checking_response(self, response_text: str) -> Dict:
        """Parse checking response"""
        try:
            # Extract JSON
            json_text = self._extract_json_from_text(response_text)
            
            if json_text:
                try:
                    parsed_data = json.loads(json_text)
                    return {
                        "violations": parsed_data.get("violations", []),
                        "suggestions": parsed_data.get("suggestions", []),
                        "compliance_score": float(parsed_data.get("compliance_score", 0.8))
                    }
                except:
                    pass
            
            # Fallback
            return {"violations": [], "suggestions": [], "compliance_score": 0.8}
            
        except Exception as e:
            logger.error(f"Failed to parse checking response: {e}")
            return {"violations": [], "suggestions": [], "compliance_score": 0.8}
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text"""
        # Multiple patterns
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'JSON:\s*([\s\S]*?)(?=\n\n|$)',
            r'(\[\s*\{[\s\S]*?\}\s*\])',
            r'(\{[\s\S]*?\})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return max(matches, key=len)
        
        return None
    
    def _fix_json(self, json_text: str) -> Optional[str]:
        """Fix common JSON issues"""
        # Remove trailing commas
        json_text = re.sub(r',\s*([\}\]])', r'\1', json_text)
        
        # Fix quotes
        json_text = re.sub(r"'([^']*)'", r'"\1"', json_text)
        
        # Fix missing quotes around keys
        json_text = re.sub(r'([\{\s,])(\w+):', r'\1"\2":', json_text)
        
        return json_text
    
    def _validate_convention(self, conv_data: Dict) -> Optional[Dict]:
        """Validate convention data"""
        if not isinstance(conv_data, dict):
            return None
        
        # Required fields
        required_fields = ["convention_type", "pattern", "rule_description"]
        for field in required_fields:
            if field not in conv_data:
                return None
        
        # Valid types
        valid_types = ["naming", "structure", "style", "best_practice", "docstring", "import"]
        
        convention_type = str(conv_data["convention_type"]).lower()
        if convention_type not in valid_types:
            convention_type = "naming"  # Default
        
        return {
            "convention_type": convention_type,
            "pattern": str(conv_data["pattern"]).strip(),
            "rule_description": str(conv_data["rule_description"]).strip(),
            "example": str(conv_data.get("example", "")).strip(),
            "confidence": float(conv_data.get("confidence", 0.8))
        }
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.stats["successful_requests"] == 1:
            self.stats["avg_response_time"] = response_time
        else:
            alpha = 0.1
            current_avg = self.stats["avg_response_time"]
            self.stats["avg_response_time"] = alpha * response_time + (1 - alpha) * current_avg
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            "model_loaded": self._model_loaded,
            "model_path": str(self.model_path)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.llm:
            del self.llm
            self.llm = None
        self._model_loaded = False
        gc.collect()
        logger.info("LLM client cleaned up")
