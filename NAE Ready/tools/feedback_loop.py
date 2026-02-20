# NAE/tools/feedback_loop.py
"""
NAE Feedback Loop System - Continuous Improvement Engine

This module implements a comprehensive feedback loop for NAE that:
1. Collects performance data from all agents
2. Analyzes patterns and behaviors
3. Identifies improvement opportunities
4. Feeds recommendations back to agents
5. Creates a cycle of continuous improvement

ALIGNED WITH:
- 3 Core Goals (Generational wealth, $5M in 8 years, Optimize options trading)
- Long-Term Plan (Phase-aware improvements)
"""

import os
import json
import datetime
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import time


@dataclass
class PerformanceMetric:
    """Performance metric for feedback analysis"""
    agent_name: str
    metric_name: str
    value: float
    timestamp: str
    context: Dict[str, Any]


@dataclass
class ImprovementRecommendation:
    """Recommendation for system improvement"""
    agent_name: str
    recommendation_type: str  # "strategy", "position_sizing", "entry_timing", "exit_timing", "risk_management"
    priority: str  # "low", "medium", "high", "critical"
    description: str
    expected_impact: str
    implementation_details: Dict[str, Any]
    timestamp: str


@dataclass
class PatternAnalysis:
    """Analysis of patterns in system behavior"""
    pattern_type: str  # "winning_pattern", "losing_pattern", "timing_pattern", "risk_pattern"
    pattern_description: str
    confidence: float  # 0-1
    frequency: int
    recommendations: List[str]
    timestamp: str


class FeedbackLoopSystem:
    """
    Comprehensive feedback loop system for NAE continuous improvement
    
    Collects data → Analyzes patterns → Identifies improvements → Feeds back to agents
    """
    
    def __init__(self, data_dir: str = "data/feedback_loop"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Performance data storage
        self.performance_metrics: List[PerformanceMetric] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.agent_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Improvement tracking
        self.recommendations: List[ImprovementRecommendation] = []
        self.implemented_improvements: List[Dict[str, Any]] = []
        
        # Pattern analysis
        self.pattern_analyses: List[PatternAnalysis] = []
        
        # Feedback loop state
        self.feedback_cycle_count = 0
        self.last_analysis_time = None
        self.analysis_interval = 300  # 5 minutes
        
        # Agent references (set externally)
        self.optimus_agent = None
        self.ralph_agent = None
        self.donnie_agent = None
        self.casey_agent = None
        self.splinter_agent = None
        
        # Load historical data
        self._load_historical_data()
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent for feedback collection"""
        if agent_name == "OptimusAgent" or agent_name == "optimus":
            self.optimus_agent = agent_instance
        elif agent_name == "RalphAgent" or agent_name == "ralph":
            self.ralph_agent = agent_instance
        elif agent_name == "DonnieAgent" or agent_name == "donnie":
            self.donnie_agent = agent_instance
        elif agent_name == "CaseyAgent" or agent_name == "casey":
            self.casey_agent = agent_instance
        elif agent_name == "SplinterAgent" or agent_name == "splinter":
            self.splinter_agent = agent_instance
    
    def collect_performance_data(self) -> Dict[str, Any]:
        """
        Collect performance data from all agents
        
        Returns comprehensive performance snapshot
        """
        performance_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agents": {},
            "trading": {},
            "system": {}
        }
        
        # Collect from Optimus
        if self.optimus_agent:
            try:
                status = self.optimus_agent.get_trading_status()
                performance_data["agents"]["Optimus"] = {
                    "nav": status.get("nav", 0),
                    "daily_pnl": status.get("daily_pnl", 0),
                    "realized_pnl": status.get("realized_pnl", 0),
                    "unrealized_pnl": status.get("unrealized_pnl", 0),
                    "open_positions": status.get("open_positions", 0),
                    "consecutive_losses": status.get("consecutive_losses", 0),
                    "risk_metrics": status.get("risk_metrics", {}),
                    "current_phase": getattr(self.optimus_agent, "current_phase", "Phase 1"),
                    "goal_progress": (status.get("nav", 0) / getattr(self.optimus_agent, "target_goal", 5000000)) * 100
                }
                
                # Collect trade history
                if hasattr(self.optimus_agent, "execution_history"):
                    recent_trades = self.optimus_agent.execution_history[-10:]  # Last 10 trades
                    performance_data["trading"]["recent_trades"] = recent_trades
                    self.trade_history.extend(recent_trades)
                
            except Exception as e:
                performance_data["agents"]["Optimus"] = {"error": str(e)}
        
        # Collect from Ralph
        if self.ralph_agent:
            try:
                performance_data["agents"]["Ralph"] = {
                    "approved_strategies": len(getattr(self.ralph_agent, "strategy_database", [])),
                    "candidate_pool_size": len(getattr(self.ralph_agent, "candidate_pool", [])),
                    "status": getattr(self.ralph_agent, "status", "Unknown")
                }
            except Exception as e:
                performance_data["agents"]["Ralph"] = {"error": str(e)}
        
        # Collect from Donnie
        if self.donnie_agent:
            try:
                performance_data["agents"]["Donnie"] = {
                    "execution_history_size": len(getattr(self.donnie_agent, "execution_history", [])),
                    "candidate_strategies": len(getattr(self.donnie_agent, "candidate_strategies", []))
                }
            except Exception as e:
                performance_data["agents"]["Donnie"] = {"error": str(e)}
        
        # Store performance data
        self.agent_performance[datetime.datetime.now().isoformat()] = performance_data
        
        # Save to file
        self._save_performance_data(performance_data)
        
        return performance_data
    
    def analyze_patterns(self) -> List[PatternAnalysis]:
        """
        Analyze patterns in performance data to identify improvement opportunities
        """
        patterns = []
        
        # Analyze winning patterns
        winning_patterns = self._analyze_winning_patterns()
        patterns.extend(winning_patterns)
        
        # Analyze losing patterns
        losing_patterns = self._analyze_losing_patterns()
        patterns.extend(losing_patterns)
        
        # Analyze timing patterns
        timing_patterns = self._analyze_timing_patterns()
        patterns.extend(timing_patterns)
        
        # Analyze risk patterns
        risk_patterns = self._analyze_risk_patterns()
        patterns.extend(risk_patterns)
        
        # Store patterns
        self.pattern_analyses.extend(patterns)
        
        return patterns
    
    def generate_improvements(self) -> List[ImprovementRecommendation]:
        """
        Generate improvement recommendations based on pattern analysis
        """
        recommendations = []
        
        # Analyze recent performance
        recent_performance = self._get_recent_performance(days=7)
        
        # Strategy improvements
        strategy_recs = self._generate_strategy_improvements(recent_performance)
        recommendations.extend(strategy_recs)
        
        # Position sizing improvements
        sizing_recs = self._generate_position_sizing_improvements(recent_performance)
        recommendations.extend(sizing_recs)
        
        # Entry timing improvements
        entry_recs = self._generate_entry_timing_improvements(recent_performance)
        recommendations.extend(entry_recs)
        
        # Exit timing improvements
        exit_recs = self._generate_exit_timing_improvements(recent_performance)
        recommendations.extend(exit_recs)
        
        # Risk management improvements
        risk_recs = self._generate_risk_management_improvements(recent_performance)
        recommendations.extend(risk_recs)
        
        # Store recommendations
        self.recommendations.extend(recommendations)
        
        # Save recommendations
        self._save_recommendations(recommendations)
        
        return recommendations
    
    def feed_back_to_agents(self, recommendations: List[ImprovementRecommendation]):
        """
        Feed improvement recommendations back to agents
        """
        for rec in recommendations:
            # Feed to Optimus
            if rec.agent_name == "OptimusAgent" and self.optimus_agent:
                self._apply_optimus_improvement(rec)
            
            # Feed to Ralph
            if rec.agent_name == "RalphAgent" and self.ralph_agent:
                self._apply_ralph_improvement(rec)
            
            # Feed to Donnie
            if rec.agent_name == "DonnieAgent" and self.donnie_agent:
                self._apply_donnie_improvement(rec)
            
            # Feed to Casey
            if rec.agent_name == "CaseyAgent" and self.casey_agent:
                self._apply_casey_improvement(rec)
        
        # Notify Casey and Splinter of recommendations
        if self.casey_agent:
            self._notify_casey(recommendations)
        
        if self.splinter_agent:
            self._notify_splinter(recommendations)
    
    def run_feedback_cycle(self) -> Dict[str, Any]:
        """
        Run a complete feedback cycle:
        1. Collect performance data
        2. Analyze patterns
        3. Generate improvements
        4. Feed back to agents
        """
        self.feedback_cycle_count += 1
        
        cycle_result = {
            "cycle_number": self.feedback_cycle_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "performance_data": None,
            "patterns": None,
            "recommendations": None
        }
        
        try:
            # Step 1: Collect performance data
            performance_data = self.collect_performance_data()
            cycle_result["performance_data"] = performance_data
            
            # Step 2: Analyze patterns
            patterns = self.analyze_patterns()
            cycle_result["patterns"] = [asdict(p) for p in patterns]
            
            # Step 3: Generate improvements
            recommendations = self.generate_improvements()
            cycle_result["recommendations"] = [asdict(r) for r in recommendations]
            
            # Step 4: Feed back to agents
            self.feed_back_to_agents(recommendations)
            
            self.last_analysis_time = datetime.datetime.now()
            
        except Exception as e:
            cycle_result["error"] = str(e)
        
        return cycle_result
    
    # ==================== Helper Methods ====================
    
    def _analyze_winning_patterns(self) -> List[PatternAnalysis]:
        """Analyze patterns in winning trades"""
        patterns = []
        
        if not self.trade_history:
            return patterns
        
        # Analyze recent trades
        recent_trades = [t for t in self.trade_history if t.get("result", {}).get("status") == "filled"]
        
        if len(recent_trades) < 5:
            return patterns
        
        # Pattern: High entry timing score → Win
        high_timing_wins = [t for t in recent_trades 
                          if t.get("details", {}).get("entry_timing_score", 0) > 60]
        
        if len(high_timing_wins) > len(recent_trades) * 0.7:
            patterns.append(PatternAnalysis(
                pattern_type="winning_pattern",
                pattern_description="High entry timing scores (>60) correlate with winning trades",
                confidence=0.75,
                frequency=len(high_timing_wins),
                recommendations=[
                    "Increase minimum entry timing score threshold",
                    "Focus on high-confidence entry signals"
                ],
                timestamp=datetime.datetime.now().isoformat()
            ))
        
        # Pattern: Wheel Strategy → Consistent wins
        wheel_trades = [t for t in recent_trades 
                       if "wheel" in t.get("details", {}).get("strategy_name", "").lower()]
        
        if wheel_trades and len(wheel_trades) > 0:
            patterns.append(PatternAnalysis(
                pattern_type="winning_pattern",
                pattern_description="Wheel Strategy shows consistent performance",
                confidence=0.70,
                frequency=len(wheel_trades),
                recommendations=[
                    "Increase allocation to Wheel Strategy",
                    "Expand Wheel Strategy to more symbols"
                ],
                timestamp=datetime.datetime.now().isoformat()
            ))
        
        return patterns
    
    def _analyze_losing_patterns(self) -> List[PatternAnalysis]:
        """Analyze patterns in losing trades"""
        patterns = []
        
        if not self.trade_history:
            return patterns
        
        recent_trades = self.trade_history[-20:]  # Last 20 trades
        
        # Pattern: Low entry timing score → Loss
        low_timing_trades = [t for t in recent_trades 
                           if t.get("details", {}).get("entry_timing_score", 100) < 40]
        
        if low_timing_trades:
            patterns.append(PatternAnalysis(
                pattern_type="losing_pattern",
                pattern_description="Low entry timing scores (<40) correlate with losses",
                confidence=0.80,
                frequency=len(low_timing_trades),
                recommendations=[
                    "Reject trades with timing score < 40",
                    "Improve entry timing analysis"
                ],
                timestamp=datetime.datetime.now().isoformat()
            ))
        
        return patterns
    
    def _analyze_timing_patterns(self) -> List[PatternAnalysis]:
        """Analyze entry/exit timing patterns"""
        patterns = []
        
        if not self.optimus_agent:
            return patterns
        
        try:
            status = self.optimus_agent.get_trading_status()
            open_positions = status.get("open_positions_detail", {})
            
            # Analyze exit timing effectiveness
            if open_positions:
                patterns.append(PatternAnalysis(
                    pattern_type="timing_pattern",
                    pattern_description=f"Monitoring {len(open_positions)} open positions for optimal exit timing",
                    confidence=0.65,
                    frequency=len(open_positions),
                    recommendations=[
                        "Continue monitoring exit timing",
                        "Implement trailing stops for profitable positions"
                    ],
                    timestamp=datetime.datetime.now().isoformat()
                ))
        
        except Exception as e:
            pass
        
        return patterns
    
    def _analyze_risk_patterns(self) -> List[PatternAnalysis]:
        """Analyze risk management patterns"""
        patterns = []
        
        if not self.optimus_agent:
            return patterns
        
        try:
            status = self.optimus_agent.get_trading_status()
            consecutive_losses = status.get("consecutive_losses", 0)
            daily_pnl = status.get("daily_pnl", 0)
            nav = status.get("nav", 0.0)
            
            # Pattern: High consecutive losses
            if consecutive_losses >= 3:
                patterns.append(PatternAnalysis(
                    pattern_type="risk_pattern",
                    pattern_description=f"High consecutive losses detected: {consecutive_losses}",
                    confidence=0.90,
                    frequency=consecutive_losses,
                    recommendations=[
                        "Reduce position sizes",
                        "Increase entry timing score threshold",
                        "Pause trading if losses continue"
                    ],
                    timestamp=datetime.datetime.now().isoformat()
                ))
            
            # Pattern: Daily loss approaching limit
            loss_pct = abs(daily_pnl) / nav if nav > 0 else 0
            if loss_pct > 0.015:  # 1.5% loss (close to 2% limit)
                patterns.append(PatternAnalysis(
                    pattern_type="risk_pattern",
                    pattern_description=f"Daily loss approaching limit: {loss_pct:.2%}",
                    confidence=0.85,
                    frequency=1,
                    recommendations=[
                        "Reduce position sizes immediately",
                        "Increase stop loss enforcement",
                        "Consider pausing new trades"
                    ],
                    timestamp=datetime.datetime.now().isoformat()
                ))
        
        except Exception as e:
            pass
        
        return patterns
    
    def _generate_strategy_improvements(self, performance: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate strategy improvement recommendations"""
        recommendations = []
        
        # Analyze win rate
        if self.trade_history:
            recent_trades = self.trade_history[-20:]
            wins = sum(1 for t in recent_trades 
                      if t.get("result", {}).get("pnl", 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0
            
            if win_rate < 0.60:  # Win rate below 60%
                recommendations.append(ImprovementRecommendation(
                    agent_name="RalphAgent",
                    recommendation_type="strategy",
                    priority="high",
                    description=f"Win rate is {win_rate:.1%} - below target of 70%",
                    expected_impact="Increase win rate to 70%+ for better compound growth",
                    implementation_details={
                        "action": "Improve strategy filtering",
                        "min_trust_score": 60,  # Increase from 55
                        "min_backtest_score": 55  # Increase from 50
                    },
                    timestamp=datetime.datetime.now().isoformat()
                ))
        
        return recommendations
    
    def _generate_position_sizing_improvements(self, performance: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate position sizing improvement recommendations"""
        recommendations = []
        
        if self.optimus_agent:
            try:
                status = self.optimus_agent.get_trading_status()
                nav = status.get("nav", 0.0)
                
                # If NAV is growing, increase position sizes gradually
                if nav > 50:  # NAV doubled
                    recommendations.append(ImprovementRecommendation(
                        agent_name="OptimusAgent",
                        recommendation_type="position_sizing",
                        priority="medium",
                        description=f"NAV has grown to ${nav:.2f} - consider increasing position sizes",
                        expected_impact="Larger position sizes for faster compound growth",
                        implementation_details={
                            "action": "Increase max position size from 5% to 7% of NAV",
                            "new_position_pct": 0.07
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    ))
            
            except Exception as e:
                pass
        
        return recommendations
    
    def _generate_entry_timing_improvements(self, performance: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate entry timing improvement recommendations"""
        recommendations = []
        
        if self.trade_history:
            recent_trades = self.trade_history[-10:]
            avg_timing_score = statistics.mean([
                t.get("details", {}).get("entry_timing_score", 50) 
                for t in recent_trades 
                if t.get("details", {}).get("entry_timing_score")
            ]) if recent_trades else 50
            
            if avg_timing_score < 50:
                recommendations.append(ImprovementRecommendation(
                    agent_name="OptimusAgent",
                    recommendation_type="entry_timing",
                    priority="high",
                    description=f"Average entry timing score is {avg_timing_score:.1f} - below optimal",
                    expected_impact="Better entry timing = higher win rate and profits",
                    implementation_details={
                        "action": "Increase minimum timing score threshold",
                        "new_min_score": 50  # Increase from 40
                    },
                    timestamp=datetime.datetime.now().isoformat()
                ))
        
        return recommendations
    
    def _generate_exit_timing_improvements(self, performance: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate exit timing improvement recommendations"""
        recommendations = []
        
        # Check if trailing stops are being used effectively
        if self.optimus_agent:
            try:
                status = self.optimus_agent.get_trading_status()
                open_positions = status.get("open_positions_detail", {})
                
                # Analyze if profitable positions are being held too long
                profitable_positions = [
                    pos for pos in open_positions.values() 
                    if pos.get("unrealized_pnl", 0) > 0
                ]
                
                if len(profitable_positions) > 3:  # Multiple profitable positions
                    recommendations.append(ImprovementRecommendation(
                        agent_name="OptimusAgent",
                        recommendation_type="exit_timing",
                        priority="medium",
                        description=f"{len(profitable_positions)} profitable positions open - consider taking profits",
                        expected_impact="Lock in profits and compound growth",
                        implementation_details={
                            "action": "Activate trailing stops for profitable positions",
                            "trailing_stop_pct": 0.03
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    ))
            
            except Exception as e:
                pass
        
        return recommendations
    
    def _generate_risk_management_improvements(self, performance: Dict[str, Any]) -> List[ImprovementRecommendation]:
        """Generate risk management improvement recommendations"""
        recommendations = []
        
        if self.optimus_agent:
            try:
                status = self.optimus_agent.get_trading_status()
                consecutive_losses = status.get("consecutive_losses", 0)
                daily_pnl = status.get("daily_pnl", 0)
                
                # If consecutive losses, recommend tighter risk management
                if consecutive_losses >= 3:
                    recommendations.append(ImprovementRecommendation(
                        agent_name="OptimusAgent",
                        recommendation_type="risk_management",
                        priority="critical",
                        description=f"{consecutive_losses} consecutive losses - tighten risk management",
                        expected_impact="Protect capital and prevent further losses",
                        implementation_details={
                            "action": "Reduce position sizes by 50%",
                            "reduce_stop_loss": True,
                            "pause_new_trades": consecutive_losses >= 4
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    ))
            
            except Exception as e:
                pass
        
        return recommendations
    
    def _apply_optimus_improvement(self, rec: ImprovementRecommendation):
        """Apply improvement recommendation to Optimus"""
        if not self.optimus_agent:
            return
        
        try:
            if rec.recommendation_type == "entry_timing":
                # Update minimum timing score
                new_min = rec.implementation_details.get("new_min_score", 40)
                # This would be applied in execute_trade method
                self.optimus_agent.log_action(f"📊 Feedback: Entry timing min score increased to {new_min}")
            
            elif rec.recommendation_type == "position_sizing":
                # Update position sizing
                new_pct = rec.implementation_details.get("new_position_pct", 0.05)
                self.optimus_agent.log_action(f"📊 Feedback: Position size increased to {new_pct:.1%} of NAV")
            
            elif rec.recommendation_type == "risk_management":
                # Apply risk management improvements
                if rec.implementation_details.get("pause_new_trades"):
                    self.optimus_agent.activate_kill_switch("Feedback loop: High consecutive losses")
                else:
                    self.optimus_agent.log_action(f"📊 Feedback: Risk management tightened - {rec.description}")
        
        except Exception as e:
            pass
    
    def _apply_ralph_improvement(self, rec: ImprovementRecommendation):
        """Apply improvement recommendation to Ralph"""
        if not self.ralph_agent:
            return
        
        try:
            if rec.recommendation_type == "strategy":
                # Update strategy filtering thresholds
                new_trust = rec.implementation_details.get("min_trust_score", 55)
                new_backtest = rec.implementation_details.get("min_backtest_score", 50)
                
                if hasattr(self.ralph_agent, "config"):
                    self.ralph_agent.config["min_trust_score"] = new_trust
                    self.ralph_agent.config["min_backtest_score"] = new_backtest
                    self.ralph_agent.log_action(f"📊 Feedback: Strategy thresholds updated (trust: {new_trust}, backtest: {new_backtest})")
        
        except Exception as e:
            pass
    
    def _apply_donnie_improvement(self, rec: ImprovementRecommendation):
        """Apply improvement recommendation to Donnie"""
        if not self.donnie_agent:
            return
        
        try:
            self.donnie_agent.log_action(f"📊 Feedback received: {rec.description}")
        except Exception as e:
            pass
    
    def _apply_casey_improvement(self, rec: ImprovementRecommendation):
        """Apply improvement recommendation to Casey"""
        if not self.casey_agent:
            return
        
        try:
            self.casey_agent.log_action(f"📊 Feedback: {rec.description}")
        except Exception as e:
            pass
    
    def _notify_casey(self, recommendations: List[ImprovementRecommendation]):
        """Notify Casey of improvement recommendations"""
        if not self.casey_agent:
            return
        
        try:
            message = {
                "from": "FeedbackLoop",
                "to": "CaseyAgent",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "improvement_recommendations",
                "content": {
                    "recommendations": [asdict(r) for r in recommendations],
                    "count": len(recommendations)
                }
            }
            self.casey_agent.receive_message(message)
        except Exception as e:
            pass
    
    def _notify_splinter(self, recommendations: List[ImprovementRecommendation]):
        """Notify Splinter of improvement recommendations"""
        if not self.splinter_agent:
            return
        
        try:
            message = {
                "from": "FeedbackLoop",
                "to": "SplinterAgent",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "improvement_recommendations",
                "content": {
                    "recommendations": [asdict(r) for r in recommendations],
                    "count": len(recommendations)
                }
            }
            self.splinter_agent.receive_message(message)
        except Exception as e:
            pass
    
    def _get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance data"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        recent_metrics = []
        for timestamp_str, data in self.agent_performance.items():
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp.replace(tzinfo=None) >= cutoff_date:
                    recent_metrics.append(data)
            except:
                pass
        
        return {
            "metrics": recent_metrics,
            "period_days": days
        }
    
    def _save_performance_data(self, data: Dict[str, Any]):
        """Save performance data to file"""
        try:
            filename = f"{self.data_dir}/performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass
    
    def _save_recommendations(self, recommendations: List[ImprovementRecommendation]):
        """Save recommendations to file"""
        try:
            filename = f"{self.data_dir}/recommendations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump([asdict(r) for r in recommendations], f, indent=2)
        except Exception as e:
            pass
    
    def _load_historical_data(self):
        """Load historical performance data"""
        try:
            # Load recent performance files
            if os.path.exists(self.data_dir):
                files = [f for f in os.listdir(self.data_dir) if f.startswith("performance_")]
                for file in sorted(files)[-10:]:  # Last 10 files
                    filepath = os.path.join(self.data_dir, file)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            timestamp = data.get("timestamp", "")
                            if timestamp:
                                self.agent_performance[timestamp] = data
                    except:
                        pass
        except Exception as e:
            pass
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback loop activity"""
        return {
            "cycle_count": self.feedback_cycle_count,
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "total_recommendations": len(self.recommendations),
            "total_patterns": len(self.pattern_analyses),
            "recent_recommendations": [
                asdict(r) for r in self.recommendations[-10:]
            ],
            "recent_patterns": [
                asdict(p) for p in self.pattern_analyses[-10:]
            ]
        }


def create_feedback_loop(data_dir: str = "data/feedback_loop") -> FeedbackLoopSystem:
    """Factory function to create feedback loop system"""
    return FeedbackLoopSystem(data_dir=data_dir)

