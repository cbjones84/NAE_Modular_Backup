#!/usr/bin/env python3
"""
Optimus Integration for Self-Healing Engine

Plugs the self-healing diagnostic engine directly into Optimus.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
sys.path.insert(0, nae_root)

from execution.self_healing.tradier_self_healing_engine import (
    TradierSelfHealingEngine,
    DiagnosticIssue,
    AutoFixResult,
    IssueSeverity
)
from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter

logger = logging.getLogger(__name__)


class OptimusSelfHealingIntegration:
    """
    Integration layer between Optimus and Self-Healing Engine
    """
    
    def __init__(self, optimus_agent, tradier_adapter: Optional[TradierBrokerAdapter] = None):
        """
        Initialize integration
        
        Args:
            optimus_agent: OptimusAgent instance
            tradier_adapter: TradierBrokerAdapter instance (optional, will create if not provided)
        """
        self.optimus = optimus_agent
        
        # Get or create Tradier adapter
        if not tradier_adapter:
            try:
                tradier_adapter = TradierBrokerAdapter(
                    client_id=os.getenv("TRADIER_CLIENT_ID"),
                    client_secret=os.getenv("TRADIER_CLIENT_SECRET"),
                    api_key=os.getenv("TRADIER_API_KEY"),
                    account_id=os.getenv("TRADIER_ACCOUNT_ID"),
                    sandbox=os.getenv("TRADIER_SANDBOX", "false").lower() == "true"
                )
            except Exception as e:
                logger.error(f"Failed to create Tradier adapter: {e}")
                tradier_adapter = None
        
        self.tradier_adapter = tradier_adapter
        
        # Initialize self-healing engine with Optimus callbacks
        self.self_healing_engine = None
        if tradier_adapter:
            self.self_healing_engine = TradierSelfHealingEngine(
                tradier_adapter=tradier_adapter,
                check_interval=60,  # Check every minute
                enable_auto_fix=True,
                on_issue_detected=self._on_issue_detected,
                on_issue_resolved=self._on_issue_resolved,
                on_auto_fix_applied=self._on_auto_fix_applied
            )
            
            # Start monitoring
            self.self_healing_engine.start()
            
            logger.info("🔧 Self-healing engine integrated with Optimus")
        else:
            logger.warning("⚠️ Self-healing engine not available (no Tradier adapter)")
    
    def _on_issue_detected(self, issue: DiagnosticIssue):
        """Callback when issue is detected"""
        try:
            # Log to Optimus
            self.optimus.log_action(
                f"🔍 [Self-Healing] Issue detected: {issue.issue_id} - {issue.description}"
            )
            
            # Log severity-based message
            if issue.severity == IssueSeverity.CRITICAL:
                self.optimus.log_action(
                    f"🚨 [Self-Healing] CRITICAL: {issue.description}"
                )
            elif issue.severity == IssueSeverity.ERROR:
                self.optimus.log_action(
                    f"❌ [Self-Healing] ERROR: {issue.description}"
                )
            elif issue.severity == IssueSeverity.WARNING:
                self.optimus.log_action(
                    f"⚠️ [Self-Healing] WARNING: {issue.description}"
                )
            
            # Store in Optimus state (if available)
            if hasattr(self.optimus, 'diagnostic_issues'):
                if not hasattr(self.optimus, 'diagnostic_issues'):
                    self.optimus.diagnostic_issues = []
                self.optimus.diagnostic_issues.append(issue)
            
        except Exception as e:
            logger.error(f"Error in on_issue_detected callback: {e}")
    
    def _on_issue_resolved(self, issue: DiagnosticIssue):
        """Callback when issue is resolved"""
        try:
            # Log to Optimus
            self.optimus.log_action(
                f"✅ [Self-Healing] Issue resolved: {issue.issue_id}"
            )
            
            # Remove from Optimus state
            if hasattr(self.optimus, 'diagnostic_issues'):
                self.optimus.diagnostic_issues = [
                    i for i in self.optimus.diagnostic_issues
                    if i.issue_id != issue.issue_id
                ]
            
        except Exception as e:
            logger.error(f"Error in on_issue_resolved callback: {e}")
    
    def _on_auto_fix_applied(self, issue: DiagnosticIssue, result: AutoFixResult):
        """Callback when auto-fix is applied"""
        try:
            if result == AutoFixResult.FIXED:
                self.optimus.log_action(
                    f"🔧 [Self-Healing] Auto-fixed: {issue.issue_id}"
                )
            elif result == AutoFixResult.PARTIAL:
                self.optimus.log_action(
                    f"⚠️ [Self-Healing] Partially fixed: {issue.issue_id}"
                )
            elif result == AutoFixResult.FAILED:
                self.optimus.log_action(
                    f"❌ [Self-Healing] Auto-fix failed: {issue.issue_id}"
                )
            
        except Exception as e:
            logger.error(f"Error in on_auto_fix_applied callback: {e}")
    
    def diagnose_order_failure(self, order: Dict[str, Any], error: Any) -> DiagnosticIssue:
        """
        Diagnose order failure (called by Optimus when order fails)
        
        Args:
            order: Order that failed
            error: Error from order submission
        
        Returns:
            DiagnosticIssue with diagnosis
        """
        if not self.self_healing_engine:
            return DiagnosticIssue(
                issue_id="order_failure_unknown",
                severity=IssueSeverity.ERROR,
                category="order",
                description=f"Order failed: {str(error)}",
                detected_at=None,
                error_message=str(error),
                auto_fixable=False
            )
        
        return self.self_healing_engine.diagnose_order_failure(order, error)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for Optimus"""
        if not self.self_healing_engine:
            return {
                "status": "operational",
                "health_score": 0.5,
                "message": "Self-healing engine not available - trading allowed"
            }
        
        status = self.self_healing_engine.get_status()
        health_score = self.self_healing_engine.get_health_score()
        
        # Determine status based on critical issues, not just score
        critical_issues = [
            issue for issue in status["issues"]
            if issue["severity"] == "critical"
        ]
        
        if len(critical_issues) > 0:
            health_status = "unhealthy"
        elif health_score > 0.7:
            health_status = "healthy"
        else:
            health_status = "degraded"  # Degraded but still operational
        
        return {
            "status": health_status,
            "health_score": health_score,
            "active_issues": status["active_issues"],
            "issues": status["issues"],
            "current_state": status["current_state"],
            "stats": status["stats"]
        }
    
    def can_trade(self) -> bool:
        """Check if trading is possible
        
        Trading is only blocked for genuinely critical issues (API down, account suspended).
        Non-critical issues (low buying power, options not approved, unknown status) should
        NOT block stock trades - Optimus needs to be able to sell positions to recover.
        """
        if not self.self_healing_engine:
            # If self-healing engine isn't available, allow trading
            # (better to attempt and fail than to silently block)
            return True
        
        status = self.self_healing_engine.get_status()
        
        # Only block for genuinely critical issues that make trading impossible
        critical_issues = [
            issue for issue in status["issues"]
            if issue["severity"] == "critical"
        ]
        
        # Block ONLY if there are critical issues (connection down, account truly suspended)
        return len(critical_issues) == 0
    
    def stop(self):
        """Stop self-healing engine"""
        if self.self_healing_engine:
            self.self_healing_engine.stop()

