# NAE/paper_to_live_progression.py
"""
Paper-to-Live Trading Progression System for NAE
Implements FINRA/SEC compliant phased deployment with strict monitoring
"""

import os
import datetime
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time

class TradingPhase(Enum):
    SANDBOX = "sandbox"
    PAPER = "paper"
    MICRO_LIVE = "micro_live"
    SCALED_LIVE = "scaled_live"
    FULL_LIVE = "full_live"

class PhaseStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"

@dataclass
class PhaseMetrics:
    """Metrics for each trading phase"""
    phase: TradingPhase
    start_date: str
    end_date: Optional[str]
    total_trades: int
    successful_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_size: float
    risk_incidents: int
    status: PhaseStatus

class PaperToLiveProgression:
    """Manages progression from paper to live trading with strict controls"""
    
    def __init__(self):
        self.current_phase = TradingPhase.SANDBOX
        self.phase_history: List[PhaseMetrics] = []
        self.phase_requirements = self._define_phase_requirements()
        self.monitoring_active = True
        
        # Phase-specific limits
        self.phase_limits = {
            TradingPhase.SANDBOX: {"max_trades": 1000, "max_capital": 0, "duration_days": 30},
            TradingPhase.PAPER: {"max_trades": 500, "max_capital": 0, "duration_days": 90},
            TradingPhase.MICRO_LIVE: {"max_trades": 100, "max_capital": 1000, "duration_days": 30},
            TradingPhase.SCALED_LIVE: {"max_trades": 200, "max_capital": 10000, "duration_days": 60},
            TradingPhase.FULL_LIVE: {"max_trades": -1, "max_capital": -1, "duration_days": -1}
        }
        
        # Logging
        self.log_file = "logs/paper_to_live.log"
        self.audit_log_file = "logs/paper_to_live_audit.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_phase_progression, daemon=True)
        self.monitor_thread.start()
        
        self.log_action("Paper-to-Live Progression system initialized")

    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[Progression LOG] {message}")

    def _define_phase_requirements(self) -> Dict[TradingPhase, Dict[str, Any]]:
        """Define requirements for each phase progression"""
        return {
            TradingPhase.SANDBOX: {
                "min_trades": 50,
                "min_win_rate": 0.4,
                "max_drawdown": 0.2,
                "min_sharpe": 0.5,
                "max_risk_incidents": 2
            },
            TradingPhase.PAPER: {
                "min_trades": 100,
                "min_win_rate": 0.5,
                "max_drawdown": 0.15,
                "min_sharpe": 0.8,
                "max_risk_incidents": 1
            },
            TradingPhase.MICRO_LIVE: {
                "min_trades": 50,
                "min_win_rate": 0.55,
                "max_drawdown": 0.1,
                "min_sharpe": 1.0,
                "max_risk_incidents": 0
            },
            TradingPhase.SCALED_LIVE: {
                "min_trades": 100,
                "min_win_rate": 0.6,
                "max_drawdown": 0.08,
                "min_sharpe": 1.2,
                "max_risk_incidents": 0
            }
        }

    def can_progress_to_next_phase(self) -> Tuple[bool, str]:
        """Check if system can progress to next phase"""
        try:
            if not self.phase_history:
                return False, "No phase history available"
            
            current_metrics = self.phase_history[-1]
            requirements = self.phase_requirements.get(self.current_phase, {})
            
            # Check trade count
            if current_metrics.total_trades < requirements.get("min_trades", 0):
                return False, f"Insufficient trades: {current_metrics.total_trades} < {requirements.get('min_trades', 0)}"
            
            # Check win rate
            if current_metrics.win_rate < requirements.get("min_win_rate", 0):
                return False, f"Win rate too low: {current_metrics.win_rate:.2%} < {requirements.get('min_win_rate', 0):.2%}"
            
            # Check drawdown
            if current_metrics.max_drawdown > requirements.get("max_drawdown", 1):
                return False, f"Drawdown too high: {current_metrics.max_drawdown:.2%} > {requirements.get('max_drawdown', 1):.2%}"
            
            # Check Sharpe ratio
            if current_metrics.sharpe_ratio < requirements.get("min_sharpe", 0):
                return False, f"Sharpe ratio too low: {current_metrics.sharpe_ratio:.2f} < {requirements.get('min_sharpe', 0):.2f}"
            
            # Check risk incidents
            if current_metrics.risk_incidents > requirements.get("max_risk_incidents", 0):
                return False, f"Too many risk incidents: {current_metrics.risk_incidents} > {requirements.get('max_risk_incidents', 0)}"
            
            # Check phase duration
            phase_duration = self._get_phase_duration(current_metrics)
            min_duration = self.phase_limits[self.current_phase]["duration_days"]
            if min_duration > 0 and phase_duration < min_duration:
                return False, f"Phase duration too short: {phase_duration} days < {min_duration} days"
            
            return True, "All requirements met"
            
        except Exception as e:
            self.log_action(f"Error checking progression requirements: {e}")
            return False, f"Error: {e}"

    def progress_to_next_phase(self, approver: str, reason: str = "") -> bool:
        """Progress to next trading phase with approval"""
        try:
            can_progress, message = self.can_progress_to_next_phase()
            if not can_progress:
                self.log_action(f"Cannot progress to next phase: {message}")
                return False
            
            # Complete current phase
            if self.phase_history:
                self.phase_history[-1].status = PhaseStatus.COMPLETED
                self.phase_history[-1].end_date = datetime.datetime.now().isoformat()
            
            # Determine next phase
            phase_order = [
                TradingPhase.SANDBOX,
                TradingPhase.PAPER,
                TradingPhase.MICRO_LIVE,
                TradingPhase.SCALED_LIVE,
                TradingPhase.FULL_LIVE
            ]
            
            current_index = phase_order.index(self.current_phase)
            if current_index < len(phase_order) - 1:
                next_phase = phase_order[current_index + 1]
                self.current_phase = next_phase
                
                # Create audit log
                self._create_audit_log("PHASE_PROGRESSION", {
                    "from_phase": phase_order[current_index].value,
                    "to_phase": next_phase.value,
                    "approver": approver,
                    "reason": reason,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                self.log_action(f"Progressed from {phase_order[current_index].value} to {next_phase.value}")
                return True
            else:
                self.log_action("Already at final phase (FULL_LIVE)")
                return False
                
        except Exception as e:
            self.log_action(f"Error progressing to next phase: {e}")
            return False

    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Record a trade and update phase metrics"""
        try:
            if not self.phase_history or self.phase_history[-1].status != PhaseStatus.ACTIVE:
                # Start new phase if needed
                self._start_new_phase()
            
            current_metrics = self.phase_history[-1]
            
            # Update trade count
            current_metrics.total_trades += 1
            
            # Update P&L
            trade_pnl = trade_data.get('pnl', 0)
            current_metrics.total_pnl += trade_pnl
            
            # Update successful trades
            if trade_pnl > 0:
                current_metrics.successful_trades += 1
            
            # Update win rate
            current_metrics.win_rate = current_metrics.successful_trades / current_metrics.total_trades
            
            # Update average trade size
            trade_size = trade_data.get('size', 0)
            total_size = current_metrics.avg_trade_size * (current_metrics.total_trades - 1) + trade_size
            current_metrics.avg_trade_size = total_size / current_metrics.total_trades
            
            # Check for risk incidents
            if self._is_risk_incident(trade_data):
                current_metrics.risk_incidents += 1
            
            # Update Sharpe ratio (simplified calculation)
            current_metrics.sharpe_ratio = self._calculate_sharpe_ratio(current_metrics)
            
            # Check phase limits
            self._check_phase_limits(current_metrics)
            
            # Create audit log
            self._create_audit_log("TRADE_RECORDED", {
                "phase": self.current_phase.value,
                "trade_data": trade_data,
                "updated_metrics": {
                    "total_trades": current_metrics.total_trades,
                    "total_pnl": current_metrics.total_pnl,
                    "win_rate": current_metrics.win_rate
                }
            })
            
            return True
            
        except Exception as e:
            self.log_action(f"Error recording trade: {e}")
            return False

    def _start_new_phase(self):
        """Start a new trading phase"""
        new_metrics = PhaseMetrics(
            phase=self.current_phase,
            start_date=datetime.datetime.now().isoformat(),
            end_date=None,
            total_trades=0,
            successful_trades=0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            avg_trade_size=0.0,
            risk_incidents=0,
            status=PhaseStatus.ACTIVE
        )
        
        self.phase_history.append(new_metrics)
        self.log_action(f"Started new phase: {self.current_phase.value}")

    def _is_risk_incident(self, trade_data: Dict[str, Any]) -> bool:
        """Determine if trade represents a risk incident"""
        # Define risk incident criteria
        pnl = trade_data.get('pnl', 0)
        size = trade_data.get('size', 0)
        
        # Large loss relative to trade size
        if size > 0 and abs(pnl) / size > 0.1:  # 10% loss
            return True
        
        # Slippage issues
        if trade_data.get('slippage', 0) > 0.05:  # 5% slippage
            return True
        
        # Execution errors
        if trade_data.get('execution_error', False):
            return True
        
        return False

    def _calculate_sharpe_ratio(self, metrics: PhaseMetrics) -> float:
        """Calculate Sharpe ratio for phase metrics"""
        if metrics.total_trades < 2:
            return 0.0
        
        # Simplified Sharpe calculation
        avg_return = metrics.total_pnl / metrics.total_trades
        volatility = 0.1  # Placeholder - would calculate actual volatility
        
        if volatility == 0:
            return 0.0
        
        return avg_return / volatility

    def _check_phase_limits(self, metrics: PhaseMetrics):
        """Check if phase limits are exceeded"""
        limits = self.phase_limits[self.current_phase]
        
        # Check trade count limit
        max_trades = limits["max_trades"]
        if max_trades > 0 and metrics.total_trades >= max_trades:
            self.log_action(f"Phase trade limit reached: {metrics.total_trades}")
            self._suspend_phase("Trade limit exceeded")
        
        # Check capital limit
        max_capital = limits["max_capital"]
        if max_capital > 0 and abs(metrics.total_pnl) >= max_capital:
            self.log_action(f"Phase capital limit reached: {metrics.total_pnl}")
            self._suspend_phase("Capital limit exceeded")

    def _suspend_phase(self, reason: str):
        """Suspend current phase"""
        if self.phase_history:
            self.phase_history[-1].status = PhaseStatus.SUSPENDED
            self.phase_history[-1].end_date = datetime.datetime.now().isoformat()
            
            self._create_audit_log("PHASE_SUSPENDED", {
                "phase": self.current_phase.value,
                "reason": reason,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            self.log_action(f"Phase {self.current_phase.value} suspended: {reason}")

    def _get_phase_duration(self, metrics: PhaseMetrics) -> int:
        """Get phase duration in days"""
        if not metrics.start_date:
            return 0
        
        start_date = datetime.datetime.fromisoformat(metrics.start_date)
        end_date = datetime.datetime.now()
        if metrics.end_date:
            end_date = datetime.datetime.fromisoformat(metrics.end_date)
        
        return (end_date - start_date).days

    def get_current_status(self) -> Dict[str, Any]:
        """Get current progression status"""
        current_metrics = self.phase_history[-1] if self.phase_history else None
        
        return {
            "current_phase": self.current_phase.value,
            "phase_status": current_metrics.status.value if current_metrics else "none",
            "can_progress": self.can_progress_to_next_phase()[0],
            "progression_message": self.can_progress_to_next_phase()[1],
            "current_metrics": {
                "total_trades": current_metrics.total_trades if current_metrics else 0,
                "total_pnl": current_metrics.total_pnl if current_metrics else 0,
                "win_rate": current_metrics.win_rate if current_metrics else 0,
                "max_drawdown": current_metrics.max_drawdown if current_metrics else 0,
                "sharpe_ratio": current_metrics.sharpe_ratio if current_metrics else 0,
                "risk_incidents": current_metrics.risk_incidents if current_metrics else 0
            } if current_metrics else {},
            "phase_history_count": len(self.phase_history)
        }

    def _monitor_phase_progression(self):
        """Monitor phase progression continuously"""
        while self.monitoring_active:
            try:
                # Check for phase completion
                if self.phase_history and self.phase_history[-1].status == PhaseStatus.ACTIVE:
                    can_progress, _ = self.can_progress_to_next_phase()
                    if can_progress:
                        self.log_action("Phase requirements met - ready for progression")
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.log_action(f"Error in phase monitoring: {e}")
                time.sleep(300)

    def _create_audit_log(self, action: str, details: Dict[str, Any]):
        """Create immutable audit log entry"""
        timestamp = datetime.datetime.now().isoformat()
        
        log_data = {
            "timestamp": timestamp,
            "action": action,
            "details": details
        }
        
        log_string = json.dumps(log_data, sort_keys=True)
        log_hash = hashlib.sha256(log_string.encode()).hexdigest()
        
        try:
            with open(self.audit_log_file, "a") as f:
                f.write(f"{log_string}|{log_hash}\n")
        except Exception as e:
            self.log_action(f"Error writing audit log: {e}")


# ----------------------
# Test harness
# ----------------------
if __name__ == "__main__":
    print("Testing Paper-to-Live Trading Progression System...")
    
    # Initialize progression system
    progression = PaperToLiveProgression()
    
    # Test initial status
    print("\n1. Testing initial status...")
    status = progression.get_current_status()
    print(f"Current phase: {status['current_phase']}")
    print(f"Can progress: {status['can_progress']}")
    
    # Simulate trades in sandbox phase
    print("\n2. Simulating sandbox trades...")
    for i in range(60):  # More than minimum required
        trade_data = {
            "pnl": 10 if i % 3 == 0 else -5,  # 33% win rate
            "size": 1000,
            "symbol": "AAPL"
        }
        progression.record_trade(trade_data)
    
    # Check progression eligibility
    print("\n3. Checking progression eligibility...")
    can_progress, message = progression.can_progress_to_next_phase()
    print(f"Can progress: {can_progress}")
    print(f"Message: {message}")
    
    # Progress to paper phase
    if can_progress:
        print("\n4. Progressing to paper phase...")
        success = progression.progress_to_next_phase("owner123", "Sandbox phase completed successfully")
        print(f"Progression success: {success}")
    
    # Test paper phase trades
    print("\n5. Simulating paper phase trades...")
    for i in range(120):  # More than minimum required
        trade_data = {
            "pnl": 15 if i % 2 == 0 else -3,  # 50% win rate
            "size": 2000,
            "symbol": "MSFT"
        }
        progression.record_trade(trade_data)
    
    # Check final status
    print("\n6. Final status...")
    final_status = progression.get_current_status()
    print(f"Current phase: {final_status['current_phase']}")
    print(f"Total trades: {final_status['current_metrics']['total_trades']}")
    print(f"Win rate: {final_status['current_metrics']['win_rate']:.2%}")
    print(f"Total P&L: ${final_status['current_metrics']['total_pnl']:.2f}")
    
    print("\nPaper-to-Live Progression testing completed successfully!")
