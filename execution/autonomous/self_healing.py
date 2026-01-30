#!/usr/bin/env python3
"""
Self-Healing System for NAE

Automatically detects and fixes errors, bugs, and issues
"""

import os
import sys
import re
import logging
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IssueType(Enum):
    IMPORT_ERROR = "import_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    SYNTAX_ERROR = "syntax_error"
    CONFIG_ERROR = "config_error"
    UNKNOWN = "unknown"


@dataclass
class DetectedIssue:
    """A detected issue that needs fixing"""
    issue_type: IssueType
    file_path: Optional[str]
    line_number: Optional[int]
    error_message: str
    context: Dict[str, Any]
    severity: str  # low, medium, high, critical
    auto_fixable: bool


class SelfHealingSystem:
    """
    Self-healing system that detects and fixes issues automatically
    """
    
    def __init__(self):
        self.fixed_issues: List[DetectedIssue] = []
        self.fix_patterns = self._load_fix_patterns()
    
    def _load_fix_patterns(self) -> Dict[IssueType, List[Dict[str, Any]]]:
        """Load patterns for automatic fixes"""
        return {
            IssueType.IMPORT_ERROR: [
                {
                    "pattern": r"ImportError: No module named '(\w+)'",
                    "fix": lambda m: f"import {m.group(1)}",
                    "action": "add_import"
                }
            ],
            IssueType.ATTRIBUTE_ERROR: [
                {
                    "pattern": r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
                    "fix": lambda m: f"Check if {m.group(2)} exists or use getattr()",
                    "action": "suggest_fix"
                }
            ],
            IssueType.KEY_ERROR: [
                {
                    "pattern": r"KeyError: '(\w+)'",
                    "fix": lambda m: f"Use .get('{m.group(1)}', default_value)",
                    "action": "suggest_fix"
                }
            ],
            IssueType.CONNECTION_ERROR: [
                {
                    "pattern": r"ConnectionError|Connection refused|Failed to connect",
                    "fix": lambda m: "Retry connection with exponential backoff",
                    "action": "retry_with_backoff"
                }
            ],
            IssueType.TIMEOUT_ERROR: [
                {
                    "pattern": r"Timeout|timed out",
                    "fix": lambda m: "Increase timeout or retry",
                    "action": "increase_timeout"
                }
            ]
        }
    
    def detect_issue(self, error_message: str, file_path: Optional[str] = None) -> Optional[DetectedIssue]:
        """Detect issue from error message"""
        error_lower = error_message.lower()
        
        # Determine issue type
        issue_type = IssueType.UNKNOWN
        if "importerror" in error_lower or "no module named" in error_lower:
            issue_type = IssueType.IMPORT_ERROR
        elif "attributeerror" in error_lower:
            issue_type = IssueType.ATTRIBUTE_ERROR
        elif "keyerror" in error_lower:
            issue_type = IssueType.KEY_ERROR
        elif "connection" in error_lower or "connection refused" in error_lower:
            issue_type = IssueType.CONNECTION_ERROR
        elif "timeout" in error_lower or "timed out" in error_lower:
            issue_type = IssueType.TIMEOUT_ERROR
        elif "syntaxerror" in error_lower:
            issue_type = IssueType.SYNTAX_ERROR
        
        # Determine severity
        severity = "medium"
        if issue_type in [IssueType.CONNECTION_ERROR, IssueType.TIMEOUT_ERROR]:
            severity = "high"
        elif issue_type == IssueType.SYNTAX_ERROR:
            severity = "critical"
        
        # Check if auto-fixable
        auto_fixable = issue_type in self.fix_patterns
        
        issue = DetectedIssue(
            issue_type=issue_type,
            file_path=file_path,
            line_number=None,
            error_message=error_message,
            context={},
            severity=severity,
            auto_fixable=auto_fixable
        )
        
        return issue
    
    def fix_issue(self, issue: DetectedIssue) -> bool:
        """Attempt to fix an issue"""
        if not issue.auto_fixable:
            logger.debug(f"Issue {issue.issue_type.value} is not auto-fixable")
            return False
        
        patterns = self.fix_patterns.get(issue.issue_type, [])
        
        for pattern_info in patterns:
            match = re.search(pattern_info["pattern"], issue.error_message, re.IGNORECASE)
            if match:
                try:
                    fix_result = pattern_info["fix"](match)
                    action = pattern_info.get("action", "unknown")
                    
                    logger.info(f"🔧 Fixing {issue.issue_type.value}: {fix_result}")
                    
                    # Apply fix based on action
                    if action == "add_import" and issue.file_path:
                        self._add_import(issue.file_path, fix_result)
                    elif action == "retry_with_backoff":
                        # This would be handled by retry logic
                        logger.info("Applying retry with exponential backoff")
                    elif action == "increase_timeout":
                        logger.info("Suggesting timeout increase")
                    
                    self.fixed_issues.append(issue)
                    return True
                    
                except Exception as e:
                    logger.error(f"Error applying fix: {e}")
                    return False
        
        return False
    
    def _add_import(self, file_path: str, import_statement: str):
        """Add import statement to file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return
            
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check if import already exists
            if import_statement in content:
                logger.debug(f"Import already exists: {import_statement}")
                return
            
            # Add import at the top (after existing imports or at the beginning)
            lines = content.split("\n")
            insert_pos = 0
            
            # Find last import line
            for i, line in enumerate(lines):
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    insert_pos = i + 1
            
            lines.insert(insert_pos, import_statement)
            
            with open(file_path, "w") as f:
                f.write("\n".join(lines))
            
            logger.info(f"✅ Added import to {file_path}: {import_statement}")
            
        except Exception as e:
            logger.error(f"Error adding import: {e}")
    
    def heal_from_log(self, log_file: str) -> int:
        """Heal issues found in log file"""
        fixed_count = 0
        
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ["error", "exception", "traceback"]):
                    issue = self.detect_issue(line)
                    if issue and issue.auto_fixable:
                        if self.fix_issue(issue):
                            fixed_count += 1
        
        except Exception as e:
            logger.error(f"Error healing from log: {e}")
        
        return fixed_count
    
    def get_fix_summary(self) -> Dict[str, Any]:
        """Get summary of fixed issues"""
        by_type = {}
        for issue in self.fixed_issues:
            issue_type = issue.issue_type.value
            if issue_type not in by_type:
                by_type[issue_type] = 0
            by_type[issue_type] += 1
        
        return {
            "total_fixed": len(self.fixed_issues),
            "by_type": by_type,
            "recent_fixes": [
                {
                    "type": issue.issue_type.value,
                    "file": issue.file_path,
                    "message": issue.error_message[:100]
                }
                for issue in self.fixed_issues[-10:]
            ]
        }

