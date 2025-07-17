"""
Core LLM Client for LLAMA 3 8B Integration
Optimized for accuracy and 16GB RAM constraint
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
    logging.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

from src.utils.memory_monitor import MemoryMonitor
from src.utils.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class LLMClient:
    """High-accuracy LLM client for code convention analysis"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        self.model_path = Path(model_path)
        self.config = self._get_default_config()
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
    
    def _get_default_config(self) -> Dict:
        """Get accuracy-optimized default configuration"""
        return {
            "n_ctx": 8000,
            "n_batch": 512,
            "n_threads": 8,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 1000,
            "mmap": True,
            "mlock": True,
            "verbose": False,
            "seed": 42
        }
    
    def load_model(self) -> bool:
        """Load LLAMA model with error handling"""
        if not LLAMA_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return False
        
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            logger.info("Loading LLAMA model...")
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
        """Extract coding conventions from code samples"""
        if not self._model_loaded:
            logger.error("Model not loaded")
            return []
        
        logger.info(f"Extracting conventions from {len(code_samples)} code samples")
        
        all_conventions = []
        
        for i, code_sample in enumerate(code_samples):
            try:
                # Generate prompt
                prompt = self.prompt_templates.create_extraction_prompt(code_sample)
                
                # Get response
                response_text = self._generate_response(prompt)
                
                if response_text:
                    # Parse conventions
                    conventions = self._parse_extraction_response(response_text)
                    
                    # Add metadata
                    for conv in conventions:
                        conv["sample_index"] = i
                        conv["extraction_timestamp"] = time.time()
                    
                    all_conventions.extend(conventions)
                    
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
        
        return all_conventions
    
    def check_conventions(self, code: str, rules: List[Dict]) -> Dict:
        """Check code against established conventions"""
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
                
        except Exception as e:
            logger.error(f"Failed to check conventions: {e}")
            
        return {"violations": [], "suggestions": [], "compliance_score": 0.0}
    
    def _generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from model"""
        if not self._model_loaded:
            return None
        
        self.stats["total_requests"] += 1
        
        try:
            response = self.llm(
                prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"],
                repeat_penalty=self.config["repeat_penalty"],
                echo=False
            )
            
            response_text = response["choices"][0]["text"].strip()
            self.stats["successful_requests"] += 1
            
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            self.stats["failed_requests"] += 1
            return None
    
    def _parse_extraction_response(self, response_text: str) -> List[Dict]:
        """Parse convention extraction response"""
        conventions = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_text = json_match.group(1)
                parsed_data = json.loads(json_text)
                
                if isinstance(parsed_data, list):
                    for conv_data in parsed_data:
                        conv = self._validate_convention(conv_data)
                        if conv:
                            conventions.append(conv)
                            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            
        return conventions
    
    def _parse_checking_response(self, response_text: str) -> Dict:
        """Parse convention checking response"""
        try:
            # Extract JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_text = json_match.group(1)
                parsed_data = json.loads(json_text)
                
                return {
                    "violations": parsed_data.get("violations", []),
                    "suggestions": parsed_data.get("suggestions", []),
                    "compliance_score": float(parsed_data.get("compliance_score", 0.0))
                }
                
        except Exception as e:
            logger.error(f"Failed to parse checking response: {e}")
            
        return {"violations": [], "suggestions": [], "compliance_score": 0.0}
    
    def _validate_convention(self, conv_data: Dict) -> Optional[Dict]:
        """Validate convention data"""
        required_fields = ["convention_type", "pattern", "rule_description"]
        
        for field in required_fields:
            if field not in conv_data:
                return None
        
        return {
            "convention_type": str(conv_data["convention_type"]).lower(),
            "pattern": str(conv_data["pattern"]).strip(),
            "rule_description": str(conv_data["rule_description"]).strip(),
            "example": str(conv_data.get("example", "")).strip(),
            "confidence": float(conv_data.get("confidence", 0.8))
        }
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
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
