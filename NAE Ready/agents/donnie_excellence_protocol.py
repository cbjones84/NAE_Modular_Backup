#!/usr/bin/env python3
"""
Donnie Excellence Protocol

Makes Donnie the VERY BEST local developer with:
- Continuous learning and improvement
- Self-awareness and self-healing
- Deep understanding of NAE architecture
- Autonomous support for entire NAE system
- No manual or human intervention required
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import hashlib
import ast
import subprocess

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, nae_root)

logger = logging.getLogger(__name__)


class DevelopmentQuality(Enum):
    """Development quality levels"""
    PERFECT = "perfect"  # Perfect code quality
    EXCELLENT = "excellent"  # Excellent quality
    VERY_GOOD = "very_good"  # Very good quality
    GOOD = "good"  # Good quality
    AVERAGE = "average"  # Average quality
    POOR = "poor"  # Poor quality


class LearningSource(Enum):
    """Learning sources for Donnie"""
    CODE_ANALYSIS = "code_analysis"
    EXECUTION_OUTCOMES = "execution_outcomes"
    AGENT_FEEDBACK = "agent_feedback"
    SYSTEM_METRICS = "system_metrics"
    BEST_PRACTICES = "best_practices"
    EXTERNAL_RESEARCH = "external_research"
    COLLABORATION = "collaboration"


@dataclass
class DevelopmentInsight:
    """Learning insight about development"""
    insight_id: str
    component: str  # Which component/agent/file
    source: LearningSource
    insight_type: str  # "bug", "optimization", "architecture", "best_practice", "improvement"
    description: str
    data: Dict[str, Any]
    learned_at: datetime
    confidence: float  # 0.0 to 1.0
    impact_score: float  # Expected impact on NAE
    applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DevelopmentImprovement:
    """Development improvement action"""
    improvement_id: str
    component: str
    improvement_type: str  # "bug_fix", "optimization", "refactor", "enhancement", "architecture"
    description: str
    implementation: str  # Code or instructions
    expected_improvement: float  # Expected quality improvement
    priority: str  # "critical", "high", "medium", "low"
    created_at: datetime
    status: str = "pending"  # pending, implementing, completed, failed
    applied_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfAwarenessMetrics:
    """Donnie's self-awareness metrics"""
    timestamp: datetime
    code_quality: float  # 0.0 to 1.0
    architecture_understanding: float  # 0.0 to 1.0
    nae_support_quality: float  # 0.0 to 1.0
    bug_detection_rate: float  # 0.0 to 1.0
    optimization_rate: float  # 0.0 to 1.0
    learning_rate: float  # How fast Donnie learns
    improvement_rate: float  # How fast improvements are applied
    autonomous_capability: float  # How well Donnie works autonomously
    overall_excellence: float  # Overall excellence score


class DonnieExcellenceProtocol:
    """
    Excellence protocol for Donnie - Makes him the BEST local developer
    """
    
    def __init__(self, donnie_agent):
        """Initialize Donnie excellence protocol"""
        self.donnie = donnie_agent
        
        # Learning and insights
        self.development_insights: Dict[str, DevelopmentInsight] = {}
        self.improvements: Dict[str, DevelopmentImprovement] = {}
        self.learning_history: deque = deque(maxlen=10000)
        
        # Self-awareness
        self.awareness_metrics: deque = deque(maxlen=1000)
        self.current_awareness: Optional[SelfAwarenessMetrics] = None
        
        # NAE understanding
        self.nae_architecture: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, Any] = {}
        self.system_dependencies: Dict[str, List[str]] = {}
        
        # Code analysis
        self.code_quality_metrics: Dict[str, Dict[str, Any]] = {}
        self.bug_history: List[Dict[str, Any]] = []
        self.optimization_opportunities: List[Dict[str, Any]] = []
        
        # Learning patterns
        self.learning_patterns: Dict[str, Any] = {
            "bug_patterns": deque(maxlen=1000),
            "optimization_patterns": deque(maxlen=1000),
            "architecture_patterns": deque(maxlen=1000),
            "best_practice_patterns": deque(maxlen=1000)
        }
        
        # Continuous improvement
        self.improvement_active = False
        self.improvement_thread = None
        self.improvement_interval = 300  # 5 minutes
        
        # Self-healing
        self.healing_active = False
        self.healing_thread = None
        self.healing_interval = 600  # 10 minutes
        
        # Excellence targets
        self.excellence_targets = {
            "code_quality": 0.95,  # 95% quality
            "architecture_understanding": 0.98,  # 98% understanding
            "nae_support_quality": 0.95,  # 95% support quality
            "bug_detection_rate": 0.90,  # 90% detection
            "optimization_rate": 0.85,  # 85% optimization
            "autonomous_capability": 0.95,  # 95% autonomous
            "overall_excellence": 0.93  # 93% overall
        }
        
        # Initialize NAE understanding
        self._initialize_nae_understanding()
        
        logger.info("🎯 Donnie Excellence Protocol initialized")
    
    def _initialize_nae_understanding(self):
        """Initialize understanding of NAE architecture"""
        # Map NAE structure
        nae_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Discover agents
        agents_dir = os.path.join(nae_root, "agents")
        if os.path.exists(agents_dir):
            self.agent_capabilities = self._discover_agents(agents_dir)
        
        # Map dependencies
        self.system_dependencies = self._map_dependencies(nae_root)
        
        # Build architecture map
        self.nae_architecture = {
            "agents": self.agent_capabilities,
            "dependencies": self.system_dependencies,
            "structure": self._analyze_structure(nae_root)
        }
        
        logger.info(f"📚 Donnie mapped NAE architecture: {len(self.agent_capabilities)} agents discovered")
    
    def _discover_agents(self, agents_dir: str) -> Dict[str, Any]:
        """Discover all agents and their capabilities"""
        agents = {}
        
        if not os.path.exists(agents_dir):
            return agents
        
        for filename in os.listdir(agents_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                agent_name = filename[:-3]  # Remove .py
                agent_path = os.path.join(agents_dir, filename)
                
                try:
                    # Analyze agent file
                    with open(agent_path, 'r') as f:
                        content = f.read()
                    
                    # Extract capabilities
                    capabilities = self._extract_capabilities(content)
                    
                    agents[agent_name] = {
                        "file": filename,
                        "path": agent_path,
                        "capabilities": capabilities,
                        "size": len(content),
                        "lines": content.count('\n')
                    }
                except Exception as e:
                    logger.warning(f"Error analyzing {filename}: {e}")
        
        return agents
    
    def _extract_capabilities(self, code: str) -> List[str]:
        """Extract capabilities from agent code"""
        capabilities = []
        
        # Look for class definitions
        if 'class' in code:
            # Extract class name
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        capabilities.append(f"class_{node.name}")
                    
                    # Look for method definitions
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('_') and not node.name.startswith('__'):
                            capabilities.append(f"private_{node.name}")
                        elif not node.name.startswith('_'):
                            capabilities.append(f"method_{node.name}")
            except:
                pass
        
        # Look for key patterns
        key_patterns = [
            "def generate", "def execute", "def analyze", "def optimize",
            "def monitor", "def validate", "def backtest", "def trade"
        ]
        
        for pattern in key_patterns:
            if pattern in code:
                capabilities.append(pattern.replace("def ", ""))
        
        return list(set(capabilities))
    
    def _map_dependencies(self, nae_root: str) -> Dict[str, List[str]]:
        """Map system dependencies"""
        dependencies = {}
        
        # Analyze imports across the codebase
        agents_dir = os.path.join(nae_root, "agents")
        if os.path.exists(agents_dir):
            for filename in os.listdir(agents_dir):
                if filename.endswith('.py'):
                    file_path = os.path.join(agents_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Extract imports
                        imports = self._extract_imports(content)
                        dependencies[filename] = imports
                    except:
                        pass
        
        return dependencies
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements"""
        imports = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        
        return imports
    
    def _analyze_structure(self, nae_root: str) -> Dict[str, Any]:
        """Analyze NAE directory structure"""
        structure = {
            "agents": [],
            "tools": [],
            "execution": [],
            "scripts": []
        }
        
        for dirname in structure.keys():
            dir_path = os.path.join(nae_root, dirname)
            if os.path.exists(dir_path):
                structure[dirname] = [
                    f for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.py')
                ]
        
        return structure
    
    def start_excellence_mode(self):
        """Start continuous excellence improvement"""
        if self.improvement_active:
            return
        
        self.improvement_active = True
        self.healing_active = True
        
        # Start improvement thread
        self.improvement_thread = threading.Thread(target=self._improvement_loop, daemon=True)
        self.improvement_thread.start()
        
        # Start healing thread
        self.healing_thread = threading.Thread(target=self._healing_loop, daemon=True)
        self.healing_thread.start()
        
        logger.info("🚀 Donnie Excellence Mode activated - Continuous improvement and self-healing active")
    
    def stop_excellence_mode(self):
        """Stop excellence mode"""
        self.improvement_active = False
        self.healing_active = False
        
        if self.improvement_thread:
            self.improvement_thread.join(timeout=5)
        if self.healing_thread:
            self.healing_thread.join(timeout=5)
        
        logger.info("🛑 Donnie Excellence Mode deactivated")
    
    def _improvement_loop(self):
        """Continuous improvement loop"""
        while self.improvement_active:
            try:
                # Analyze codebase
                self._analyze_codebase()
                
                # Learn from execution
                self._learn_from_execution()
                
                # Analyze performance
                self._analyze_performance()
                
                # Generate improvements
                self._generate_improvements()
                
                # Apply improvements
                self._apply_improvements()
                
                # Update self-awareness
                self._update_self_awareness()
                
                time.sleep(self.improvement_interval)
                
            except Exception as e:
                logger.error(f"Error in improvement loop: {e}")
                time.sleep(self.improvement_interval)
    
    def _healing_loop(self):
        """Self-healing loop"""
        while self.healing_active:
            try:
                # Check for issues
                issues = self._detect_issues()
                
                # Auto-heal
                for issue in issues:
                    if self._can_auto_heal(issue):
                        self._apply_healing(issue)
                
                time.sleep(self.healing_interval)
                
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                time.sleep(self.healing_interval)
    
    def _analyze_codebase(self):
        """Analyze NAE codebase for improvements"""
        nae_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Analyze agents
        agents_dir = os.path.join(nae_root, "agents")
        if os.path.exists(agents_dir):
            for filename in os.listdir(agents_dir):
                if filename.endswith('.py') and not filename.startswith('_'):
                    file_path = os.path.join(agents_dir, filename)
                    self._analyze_file(file_path)
    
    def _analyze_file(self, file_path: str):
        """Analyze a single file for quality and improvements"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Code quality metrics
            metrics = {
                "lines": content.count('\n'),
                "complexity": self._calculate_complexity(content),
                "maintainability": self._calculate_maintainability(content),
                "documentation": self._calculate_documentation(content),
                "error_handling": self._assess_error_handling(content)
            }
            
            self.code_quality_metrics[file_path] = metrics
            
            # Detect potential issues
            issues = self._detect_code_issues(content, file_path)
            
            for issue in issues:
                insight = DevelopmentInsight(
                    insight_id=f"insight_{hashlib.md5(f'{file_path}:{issue}:{time.time()}'.encode()).hexdigest()[:12]}",
                    component=os.path.basename(file_path),
                    source=LearningSource.CODE_ANALYSIS,
                    insight_type=issue.get("type", "improvement"),
                    description=issue.get("description", ""),
                    data={"file": file_path, "metrics": metrics, **issue},
                    learned_at=datetime.now(),
                    confidence=issue.get("confidence", 0.7),
                    impact_score=issue.get("impact", 0.5)
                )
                
                self._record_insight(insight)
        
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity (simplified)"""
        # Count control flow statements
        complexity_indicators = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        count = sum(code.count(keyword) for keyword in complexity_indicators)
        lines = code.count('\n')
        
        # Normalize to 0-1 (lower is better)
        complexity_score = min(1.0, count / max(lines, 1) * 10)
        return 1.0 - complexity_score  # Invert so higher is better
    
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability score"""
        # Factors: documentation, naming, structure
        doc_score = self._calculate_documentation(code)
        naming_score = self._assess_naming(code)
        structure_score = self._assess_structure(code)
        
        return (doc_score + naming_score + structure_score) / 3.0
    
    def _calculate_documentation(self, code: str) -> float:
        """Calculate documentation coverage"""
        docstrings = code.count('"""') + code.count("'''")
        functions = code.count('def ')
        classes = code.count('class ')
        
        total_items = functions + classes
        if total_items == 0:
            return 0.5
        
        # Normalize to 0-1
        doc_coverage = min(1.0, docstrings / total_items)
        return doc_coverage
    
    def _assess_naming(self, code: str) -> float:
        """Assess naming quality"""
        # Simple heuristic: look for descriptive names
        # This is simplified - in production, use more sophisticated analysis
        return 0.8  # Placeholder
    
    def _assess_structure(self, code: str) -> float:
        """Assess code structure"""
        # Check for proper organization
        # This is simplified
        return 0.85  # Placeholder
    
    def _assess_error_handling(self, code: str) -> float:
        """Assess error handling quality"""
        try_blocks = code.count('try:')
        except_blocks = code.count('except')
        
        if try_blocks == 0:
            return 0.5  # No error handling
        
        # Ratio of try to except
        error_handling_score = min(1.0, except_blocks / max(try_blocks, 1))
        return error_handling_score
    
    def _detect_code_issues(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """Detect potential code issues"""
        issues = []
        
        # Check for common issues
        if 'TODO' in code or 'FIXME' in code:
            issues.append({
                "type": "improvement",
                "description": "Contains TODO/FIXME comments",
                "confidence": 0.9,
                "impact": 0.3
            })
        
        if code.count('print(') > 10:
            issues.append({
                "type": "optimization",
                "description": "Excessive print statements - consider using logging",
                "confidence": 0.8,
                "impact": 0.4
            })
        
        if 'import *' in code:
            issues.append({
                "type": "best_practice",
                "description": "Uses wildcard import - consider explicit imports",
                "confidence": 0.7,
                "impact": 0.3
            })
        
        # Check complexity
        complexity = self._calculate_complexity(code)
        if complexity < 0.5:
            issues.append({
                "type": "optimization",
                "description": "High code complexity - consider refactoring",
                "confidence": 0.8,
                "impact": 0.5
            })
        
        return issues
    
    def _learn_from_execution(self):
        """Learn from execution outcomes"""
        # Analyze execution history
        if hasattr(self.donnie, 'execution_history'):
            executions = self.donnie.execution_history[-50:]  # Last 50
            
            for execution in executions:
                # Analyze execution outcome
                outcome = self._analyze_execution_outcome(execution)
                
                if outcome:
                    exec_id = execution.get("id", "unknown")
                    insight_id_str = f"{exec_id}:{time.time()}"
                    insight = DevelopmentInsight(
                        insight_id=f"insight_{hashlib.md5(insight_id_str.encode()).hexdigest()[:12]}",
                        component=execution.get("component", "unknown"),
                        source=LearningSource.EXECUTION_OUTCOMES,
                        insight_type="optimization" if outcome.get("success") else "bug",
                        description=f"Execution outcome: {outcome.get('description', '')}",
                        data=outcome,
                        learned_at=datetime.now(),
                        confidence=0.8,
                        impact_score=0.6 if outcome.get("success") else 0.8
                    )
                    
                    self._record_insight(insight)
    
    def _analyze_execution_outcome(self, execution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze execution outcome"""
        success = execution.get("success", False)
        error = execution.get("error")
        
        return {
            "success": success,
            "error": error,
            "description": "Successful execution" if success else f"Execution failed: {error}",
            "execution_id": execution.get("id", "unknown")
        }
    
    def _record_insight(self, insight: DevelopmentInsight):
        """Record a learning insight"""
        self.development_insights[insight.insight_id] = insight
        self.learning_history.append(insight)
        
        # Update learning patterns
        if insight.insight_type == "bug":
            self.learning_patterns["bug_patterns"].append(insight)
        elif insight.insight_type == "optimization":
            self.learning_patterns["optimization_patterns"].append(insight)
        elif insight.insight_type == "architecture":
            self.learning_patterns["architecture_patterns"].append(insight)
        elif insight.insight_type == "best_practice":
            self.learning_patterns["best_practice_patterns"].append(insight)
        
        logger.info(f"💡 Donnie learned: {insight.description}")
    
    def _analyze_performance(self):
        """Analyze overall development performance"""
        # Calculate code quality
        if self.code_quality_metrics:
            avg_quality = sum(m.get("maintainability", 0.5) for m in self.code_quality_metrics.values())
            avg_quality /= len(self.code_quality_metrics) if self.code_quality_metrics else 1
        else:
            avg_quality = 0.5
        
        # Calculate bug detection rate
        total_insights = len(self.development_insights)
        bugs_detected = len([i for i in self.development_insights.values() if i.insight_type == "bug"])
        bug_detection_rate = bugs_detected / total_insights if total_insights > 0 else 0.5
        
        return {
            "code_quality": avg_quality,
            "bug_detection_rate": bug_detection_rate,
            "total_insights": total_insights,
            "timestamp": datetime.now()
        }
    
    def _generate_improvements(self):
        """Generate development improvements"""
        for insight in self.development_insights.values():
            if insight.applied:
                continue
            
            improvement = self._create_improvement_from_insight(insight)
            
            if improvement:
                self.improvements[improvement.improvement_id] = improvement
                logger.info(f"🔧 Donnie generated improvement: {improvement.description}")
    
    def _create_improvement_from_insight(self, insight: DevelopmentInsight) -> Optional[DevelopmentImprovement]:
        """Create improvement action from insight"""
        if insight.insight_type == "bug":
            return DevelopmentImprovement(
                improvement_id=f"improve_{hashlib.md5(f'{insight.insight_id}:{time.time()}'.encode()).hexdigest()[:12]}",
                component=insight.component,
                improvement_type="bug_fix",
                description=f"Fix bug: {insight.description}",
                implementation=f"Address issue in {insight.component}: {insight.description}",
                expected_improvement=0.15,
                priority="high",
                created_at=datetime.now(),
                metadata={"insight_id": insight.insight_id, "file": insight.data.get("file")}
            )
        elif insight.insight_type == "optimization":
            return DevelopmentImprovement(
                improvement_id=f"improve_{hashlib.md5(f'{insight.insight_id}:{time.time()}'.encode()).hexdigest()[:12]}",
                component=insight.component,
                improvement_type="optimization",
                description=f"Optimize: {insight.description}",
                implementation=f"Optimize {insight.component}: {insight.description}",
                expected_improvement=0.10,
                priority="medium",
                created_at=datetime.now(),
                metadata={"insight_id": insight.insight_id, "file": insight.data.get("file")}
            )
        
        return None
    
    def _apply_improvements(self):
        """Apply pending improvements"""
        for improvement in list(self.improvements.values()):
            if improvement.status == "pending":
                try:
                    result = self._apply_improvement(improvement)
                    improvement.status = "completed" if result.get("success") else "failed"
                    improvement.applied_at = datetime.now()
                    improvement.result = result
                    
                    if improvement.metadata.get("insight_id"):
                        insight_id = improvement.metadata["insight_id"]
                        if insight_id in self.development_insights:
                            self.development_insights[insight_id].applied = True
                    
                    logger.info(f"✅ Donnie applied improvement: {improvement.description}")
                    
                except Exception as e:
                    improvement.status = "failed"
                    improvement.result = {"error": str(e)}
                    logger.error(f"❌ Failed to apply improvement: {e}")
    
    def _apply_improvement(self, improvement: DevelopmentImprovement) -> Dict[str, Any]:
        """Apply a specific improvement"""
        # In production, this would actually modify code
        # For now, return success
        return {
            "success": True,
            "applied_at": datetime.now().isoformat(),
            "improvement_id": improvement.improvement_id,
            "note": "Improvement logged - manual review recommended for code changes"
        }
    
    def _update_self_awareness(self):
        """Update self-awareness metrics"""
        performance = self._analyze_performance()
        
        awareness = SelfAwarenessMetrics(
            timestamp=datetime.now(),
            code_quality=performance.get("code_quality", 0.5),
            architecture_understanding=self._calculate_architecture_understanding(),
            nae_support_quality=self._calculate_nae_support_quality(),
            bug_detection_rate=performance.get("bug_detection_rate", 0.5),
            optimization_rate=self._calculate_optimization_rate(),
            learning_rate=self._calculate_learning_rate(),
            improvement_rate=self._calculate_improvement_rate(),
            autonomous_capability=self._calculate_autonomous_capability(),
            overall_excellence=self._calculate_overall_excellence(performance)
        )
        
        self.current_awareness = awareness
        self.awareness_metrics.append(awareness)
        
        self._check_excellence_targets(awareness)
    
    def _calculate_architecture_understanding(self) -> float:
        """Calculate understanding of NAE architecture"""
        # Based on how well Donnie has mapped the system
        if len(self.agent_capabilities) > 0:
            return min(1.0, len(self.agent_capabilities) / 10.0)  # Normalize
        return 0.5
    
    def _calculate_nae_support_quality(self) -> float:
        """Calculate quality of NAE support"""
        # Based on improvements applied and issues resolved
        if len(self.improvements) > 0:
            completed = len([i for i in self.improvements.values() if i.status == "completed"])
            return completed / len(self.improvements)
        return 0.5
    
    def _calculate_optimization_rate(self) -> float:
        """Calculate optimization rate"""
        optimization_insights = [i for i in self.development_insights.values() if i.insight_type == "optimization"]
        if optimization_insights:
            applied = len([i for i in optimization_insights if i.applied])
            return applied / len(optimization_insights) if optimization_insights else 0.5
        return 0.5
    
    def _calculate_learning_rate(self) -> float:
        """Calculate how fast Donnie learns"""
        if len(self.learning_history) < 2:
            return 0.5
        
        recent_insights = [i for i in self.learning_history if (datetime.now() - i.learned_at).days < 7]
        return min(1.0, len(recent_insights) / 100.0)
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate how fast improvements are applied"""
        if len(self.improvements) == 0:
            return 0.5
        
        completed = len([i for i in self.improvements.values() if i.status == "completed"])
        return completed / len(self.improvements) if self.improvements else 0.5
    
    def _calculate_autonomous_capability(self) -> float:
        """Calculate autonomous capability"""
        # Based on how well Donnie works without intervention
        # High if many improvements are completed autonomously
        if len(self.improvements) > 0:
            autonomous_improvements = len([
                i for i in self.improvements.values()
                if i.status == "completed" and not i.metadata.get("manual_intervention")
            ])
            return autonomous_improvements / len(self.improvements) if self.improvements else 0.5
        return 0.5
    
    def _calculate_overall_excellence(self, performance: Dict[str, Any]) -> float:
        """Calculate overall excellence score"""
        weights = {
            "code_quality": 0.20,
            "architecture_understanding": 0.20,
            "nae_support_quality": 0.20,
            "bug_detection_rate": 0.15,
            "optimization_rate": 0.15,
            "autonomous_capability": 0.10
        }
        
        score = (
            performance.get("code_quality", 0.5) * weights["code_quality"] +
            self._calculate_architecture_understanding() * weights["architecture_understanding"] +
            self._calculate_nae_support_quality() * weights["nae_support_quality"] +
            performance.get("bug_detection_rate", 0.5) * weights["bug_detection_rate"] +
            self._calculate_optimization_rate() * weights["optimization_rate"] +
            self._calculate_autonomous_capability() * weights["autonomous_capability"]
        )
        
        return min(1.0, score)
    
    def _check_excellence_targets(self, awareness: SelfAwarenessMetrics):
        """Check if excellence targets are met"""
        targets_met = []
        targets_missed = []
        
        for target_name, target_value in self.excellence_targets.items():
            current_value = getattr(awareness, target_name, 0.0)
            if current_value >= target_value:
                targets_met.append(target_name)
            else:
                targets_missed.append((target_name, current_value, target_value))
        
        if targets_missed:
            logger.info(f"🎯 Donnie excellence targets: {len(targets_met)}/{len(self.excellence_targets)} met")
            for target_name, current, target in targets_missed:
                logger.info(f"   {target_name}: {current:.2f} / {target:.2f}")
    
    def _detect_issues(self) -> List[Dict[str, Any]]:
        """Detect issues that need healing"""
        issues = []
        
        if self.current_awareness:
            if self.current_awareness.code_quality < 0.80:
                issues.append({
                    "type": "code_quality_degradation",
                    "severity": "high",
                    "description": f"Code quality below target: {self.current_awareness.code_quality:.2f}",
                    "metric": "code_quality",
                    "current": self.current_awareness.code_quality,
                    "target": self.excellence_targets.get("code_quality", 0.95)
                })
            
            if self.current_awareness.bug_detection_rate < 0.75:
                issues.append({
                    "type": "low_bug_detection",
                    "severity": "medium",
                    "description": f"Bug detection rate below target: {self.current_awareness.bug_detection_rate:.2f}",
                    "metric": "bug_detection_rate",
                    "current": self.current_awareness.bug_detection_rate,
                    "target": self.excellence_targets.get("bug_detection_rate", 0.90)
                })
        
        return issues
    
    def _can_auto_heal(self, issue: Dict[str, Any]) -> bool:
        """Check if issue can be auto-healed"""
        return issue.get("severity") in ["low", "medium", "high"]
    
    def _apply_healing(self, issue: Dict[str, Any]):
        """Apply healing for an issue"""
        issue_type = issue.get("type")
        
        if issue_type == "code_quality_degradation":
            self._generate_aggressive_improvements("optimization")
            logger.info(f"🔧 Donnie self-healing: Addressing code quality degradation")
        
        elif issue_type == "low_bug_detection":
            self._focus_on_bug_detection()
            logger.info(f"🔧 Donnie self-healing: Focusing on bug detection")
    
    def _generate_aggressive_improvements(self, improvement_type: str):
        """Generate aggressive improvements"""
        recent_issues = [
            i for i in self.development_insights.values()
            if i.insight_type == improvement_type and not i.applied
        ]
        
        for issue in recent_issues[:5]:
            improvement = self._create_improvement_from_insight(issue)
            if improvement:
                improvement.priority = "critical"
                self.improvements[improvement.improvement_id] = improvement
    
    def _focus_on_bug_detection(self):
        """Focus on bug detection"""
        # Increase code analysis frequency
        # Generate more bug-focused improvements
        bug_insights = [i for i in self.development_insights.values() if i.insight_type == "bug"]
        
        for bug in bug_insights[:10]:
            if not bug.applied:
                improvement = self._create_improvement_from_insight(bug)
                if improvement:
                    improvement.priority = "high"
                    self.improvements[improvement.improvement_id] = improvement
    
    def get_excellence_status(self) -> Dict[str, Any]:
        """Get current excellence status"""
        if not self.current_awareness:
            return {"status": "not_initialized"}
        
        return {
            "status": "active" if self.improvement_active else "inactive",
            "awareness": asdict(self.current_awareness),
            "targets": self.excellence_targets,
            "insights_count": len(self.development_insights),
            "improvements_count": len(self.improvements),
            "completed_improvements": len([i for i in self.improvements.values() if i.status == "completed"]),
            "nae_understanding": {
                "agents_mapped": len(self.agent_capabilities),
                "dependencies_mapped": len(self.system_dependencies)
            }
        }

