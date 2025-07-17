"""
Vector database for storing and retrieving code convention rules
"""

import logging
import json
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorRuleDatabase:
    """Vector database for code convention rules"""
    
    def __init__(self, persist_directory: str = "./data/databases", 
                 collection_name: str = "code_conventions"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.sqlite_db = None
        
        # Initialize database
        self._initialize_database()
        
        # Statistics
        self.stats = {
            "total_rules": 0,
            "successful_searches": 0,
            "failed_searches": 0
        }
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            sqlite_path = self.persist_directory / "rules.db"
            self.sqlite_db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            
            # Create tables
            self.sqlite_db.execute("""
                CREATE TABLE IF NOT EXISTS conventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    convention_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    rule_description TEXT NOT NULL,
                    example TEXT,
                    confidence REAL DEFAULT 0.8,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.sqlite_db.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_conventions(self, conventions: List[Dict], source_info: Optional[Dict] = None) -> bool:
        """Store conventions in database"""
        try:
            for convention in conventions:
                self.sqlite_db.execute("""
                    INSERT INTO conventions 
                    (convention_type, pattern, rule_description, example, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    convention["convention_type"],
                    convention["pattern"],
                    convention["rule_description"],
                    convention.get("example", ""),
                    convention.get("confidence", 0.8)
                ))
            
            self.sqlite_db.commit()
            self.stats["total_rules"] += len(conventions)
            
            logger.info(f"Stored {len(conventions)} conventions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conventions: {e}")
            return False
    
    def find_relevant_rules(self, query_text: str, n_results: int = 5,
                          min_confidence: float = 0.6) -> List[Dict]:
        """Find relevant rules using simple text matching"""
        try:
            cursor = self.sqlite_db.cursor()
            
            # Simple text search
            cursor.execute("""
                SELECT convention_type, pattern, rule_description, example, confidence
                FROM conventions
                WHERE rule_description LIKE ? OR pattern LIKE ?
                AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (f"%{query_text}%", f"%{query_text}%", min_confidence, n_results))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "convention_type": row[0],
                    "pattern": row[1],
                    "rule_description": row[2],
                    "example": row[3],
                    "confidence": row[4],
                    "similarity": 0.8  # Mock similarity
                })
            
            self.stats["successful_searches"] += 1
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.stats["failed_searches"] += 1
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            cursor = self.sqlite_db.cursor()
            cursor.execute("SELECT COUNT(*) FROM conventions")
            total_rules = cursor.fetchone()[0]
            
            return {
                "total_rules": total_rules,
                "successful_searches": self.stats["successful_searches"],
                "failed_searches": self.stats["failed_searches"],
                "status": "healthy" if total_rules > 0 else "empty"
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup database connections"""
        if self.sqlite_db:
            self.sqlite_db.close()
            self.sqlite_db = None
        logger.info("Database cleaned up")
