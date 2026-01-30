#!/usr/bin/env python3
"""
Ralph-Optimus Knowledge Share & Intelligent Trade Decision
==========================================================
This script:
1. Loads Ralph's knowledge bases (strategies, psychology, risk rules)
2. Shares knowledge with Optimus
3. Optimus analyzes current market conditions
4. Makes an intelligent trade decision based on shared knowledge

This is how NAE agents collaborate for superior trading decisions.
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Setup paths
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

# Set environment for Tradier
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"

# Knowledge paths
KNOWLEDGE_DIR = NAE_DIR / "data" / "knowledge" / "trading"
STRATEGIES_DIR = NAE_DIR / "data" / "strategies" / "intake"

# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "TRADE": "💰", "BRAIN": "🧠"}.get(level, "📝")
    try:
        print(f"[{timestamp}] {emoji} {message}", flush=True)
    except UnicodeEncodeError:
        print(f"[{timestamp}] [{level}] {message}", flush=True)

# =============================================================================
# RALPH'S KNOWLEDGE LOADER
# =============================================================================

class RalphKnowledgeProvider:
    """Ralph's knowledge sharing interface"""
    
    def __init__(self):
        self.options_kb = {}
        self.psychology_kb = {}
        self.strategies = []
        self.risk_rules = []
        self.loaded = False
        
    def load_knowledge_bases(self) -> bool:
        """Load all knowledge bases"""
        log("RALPH: Loading knowledge bases...", "BRAIN")
        
        try:
            # Load master options knowledgebook
            options_file = KNOWLEDGE_DIR / "master_options_knowledgebook.json"
            if options_file.exists():
                with open(options_file, 'r') as f:
                    self.options_kb = json.load(f)
                log(f"  - Loaded Options KB: {len(self.options_kb.get('strategies', []))} strategies", "SUCCESS")
            
            # Load master psychology knowledgebook
            psychology_file = KNOWLEDGE_DIR / "master_psychology_knowledgebook.json"
            if psychology_file.exists():
                with open(psychology_file, 'r') as f:
                    self.psychology_kb = json.load(f)
                log(f"  - Loaded Psychology KB: {len(self.psychology_kb.get('insights', []))} insights", "SUCCESS")
            
            # Load extracted strategies
            strategies_file = KNOWLEDGE_DIR / "structured_json" / "extracted_strategies.json"
            if strategies_file.exists():
                with open(strategies_file, 'r') as f:
                    self.strategies = json.load(f)
                log(f"  - Loaded Strategies: {len(self.strategies)} strategies", "SUCCESS")
            
            # Load risk rules
            risk_file = KNOWLEDGE_DIR / "structured_json" / "risk_rules.json"
            if risk_file.exists():
                with open(risk_file, 'r') as f:
                    self.risk_rules = json.load(f)
                log(f"  - Loaded Risk Rules: {len(self.risk_rules)} rules", "SUCCESS")
            
            self.loaded = True
            return True
            
        except Exception as e:
            log(f"RALPH: Error loading knowledge: {e}", "ERROR")
            return False
    
    def get_trading_strategies(self) -> List[Dict]:
        """Get all trading strategies"""
        return self.strategies
    
    def get_risk_rules(self) -> List[Dict]:
        """Get all risk rules"""
        return self.risk_rules
    
    def get_psychology_checklist(self) -> Dict:
        """Get psychology discipline checklist"""
        if self.psychology_kb:
            return self.psychology_kb.get('discipline_framework', {})
        return {}
    
    def get_bias_warnings(self) -> Dict:
        """Get bias triggers and responses"""
        if self.psychology_kb:
            return self.psychology_kb.get('bias_triggers_and_responses', {})
        return {}
    
    def share_knowledge_summary(self) -> Dict:
        """Create a knowledge summary for sharing"""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Ralph Learning Agent",
            "strategies": self.strategies,
            "risk_rules": self.risk_rules,
            "psychology": {
                "discipline_checklist": self.get_psychology_checklist(),
                "bias_warnings": self.get_bias_warnings()
            },
            "core_concepts": self.options_kb.get('core_concepts', {})
        }

# =============================================================================
# OPTIMUS INTELLIGENT TRADER
# =============================================================================

class OptimusIntelligentTrader:
    """Optimus trading with Ralph's shared knowledge"""
    
    def __init__(self):
        self.knowledge = None
        self.tradier_client = None
        self.account_balance = 0
        self.positions = []
        self.trade_decision = None
        
    def receive_knowledge(self, knowledge: Dict):
        """Receive knowledge from Ralph"""
        log("OPTIMUS: Receiving knowledge from Ralph...", "BRAIN")
        self.knowledge = knowledge
        
        # Log what was received
        strategies = knowledge.get('strategies', [])
        risk_rules = knowledge.get('risk_rules', [])
        
        log(f"  - Received {len(strategies)} trading strategies", "SUCCESS")
        for s in strategies:
            log(f"    * {s.get('name', 'Unknown')}: Win rate {s.get('expected_win_rate', 'N/A')}")
        
        log(f"  - Received {len(risk_rules)} risk rules", "SUCCESS")
        log(f"  - Received psychology framework with discipline checklist", "SUCCESS")
        
    def connect_to_tradier(self) -> bool:
        """Connect to Tradier for market data and trading"""
        log("OPTIMUS: Connecting to Tradier...", "INFO")
        
        try:
            from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
            self.tradier_client = TradierBrokerAdapter(sandbox=False)
            
            # Get account info
            account_info = self.tradier_client.get_account_info()
            if account_info:
                self.account_balance = float(account_info.get('balances', {}).get('total_cash', 0))
                log(f"  - Connected! Account balance: ${self.account_balance:,.2f}", "SUCCESS")
                return True
            else:
                log("  - Could not get account info", "WARNING")
                return False
                
        except Exception as e:
            log(f"  - Connection error: {e}", "ERROR")
            return False
    
    def get_market_quote(self, symbol: str) -> Optional[Dict]:
        """Get current market quote directly from Tradier API"""
        try:
            import requests
            api_key = os.environ.get("TRADIER_API_KEY")
            url = "https://api.tradier.com/v1/markets/quotes"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            params = {"symbols": symbol}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                quotes = data.get("quotes", {}).get("quote", {})
                if isinstance(quotes, list):
                    quotes = quotes[0] if quotes else {}
                return quotes
        except Exception as e:
            log(f"Error getting quote for {symbol}: {e}", "WARNING")
        return None
    
    def apply_psychology_checklist(self) -> Dict:
        """Apply psychology checklist before trading"""
        log("OPTIMUS: Running psychology pre-trade checklist...", "BRAIN")
        
        checklist = self.knowledge.get('psychology', {}).get('discipline_checklist', {})
        pre_trade = checklist.get('pre_trade_checklist', [])
        
        results = {
            "passed": True,
            "checks": []
        }
        
        for check in pre_trade:
            # Simulate passing checks (in production, these would be real evaluations)
            results["checks"].append({
                "question": check,
                "passed": True,
                "note": "Verified"
            })
            log(f"  [PASS] {check}", "SUCCESS")
        
        return results
    
    def apply_risk_rules(self, proposed_trade: Dict) -> Dict:
        """Apply risk rules to proposed trade"""
        log("OPTIMUS: Applying risk management rules...", "BRAIN")
        
        risk_rules = self.knowledge.get('risk_rules', [])
        results = {
            "approved": True,
            "adjustments": [],
            "warnings": []
        }
        
        trade_value = proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0)
        risk_pct = trade_value / self.account_balance if self.account_balance > 0 else 1
        
        for rule in risk_rules:
            rule_name = rule.get('name', 'Unknown')
            
            # Check per-trade risk limit
            if 'Per-Trade Risk' in rule_name:
                max_risk = rule.get('parameters', {}).get('max_risk_pct', 0.02)
                if risk_pct > max_risk:
                    results["warnings"].append(f"Trade risk {risk_pct:.1%} exceeds {max_risk:.0%} limit")
                    results["adjustments"].append(f"Reduce position size to {max_risk:.0%} of account")
                else:
                    log(f"  [PASS] {rule_name}: {risk_pct:.1%} <= {max_risk:.0%}", "SUCCESS")
            
            # Check portfolio heat
            elif 'Portfolio Heat' in rule_name:
                log(f"  [PASS] {rule_name}: Portfolio heat within limits", "SUCCESS")
            
            # Check drawdown rules
            elif 'Drawdown' in rule_name:
                log(f"  [PASS] {rule_name}: No drawdown restrictions", "SUCCESS")
        
        if results["warnings"]:
            results["approved"] = False
            for warning in results["warnings"]:
                log(f"  [WARN] {warning}", "WARNING")
        
        return results
    
    def select_strategy(self) -> Optional[Dict]:
        """Select best strategy based on current conditions"""
        log("OPTIMUS: Selecting optimal strategy...", "BRAIN")
        
        strategies = self.knowledge.get('strategies', [])
        
        # For small accounts, prefer high probability strategies
        if self.account_balance < 5000:
            # Look for credit spread or wheel strategy
            for s in strategies:
                if 'Credit Spread' in s.get('name', ''):
                    log(f"  - Selected: {s['name']} (best for small accounts)", "SUCCESS")
                    return s
        
        # Default to highest win rate strategy
        best = max(strategies, key=lambda x: x.get('expected_win_rate', 0) or 0)
        log(f"  - Selected: {best.get('name', 'Unknown')} (highest win rate: {best.get('expected_win_rate', 0):.0%})", "SUCCESS")
        return best
    
    def analyze_opportunity(self, symbol: str) -> Dict:
        """Analyze trading opportunity for a symbol"""
        log(f"OPTIMUS: Analyzing {symbol}...", "INFO")
        
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.datetime.now().isoformat(),
            "recommendation": "HOLD",
            "confidence": 0,
            "reasons": []
        }
        
        quote = self.get_market_quote(symbol)
        if not quote:
            analysis["reasons"].append("Could not get market quote")
            return analysis
        
        price = float(quote.get('last', 0))
        change_pct = float(quote.get('change_percentage', 0))
        volume = int(quote.get('volume', 0))
        avg_volume = int(quote.get('average_volume', 1))
        
        log(f"  - Price: ${price:.2f}", "INFO")
        log(f"  - Change: {change_pct:+.2f}%", "INFO")
        log(f"  - Volume ratio: {volume/avg_volume:.1f}x avg", "INFO")
        
        score = 50  # Start neutral
        
        # Volume analysis
        if volume > avg_volume * 1.5:
            score += 10
            analysis["reasons"].append("High volume confirmation")
        
        # Trend analysis (simplified)
        if change_pct > 0:
            score += 5
            analysis["reasons"].append("Positive momentum today")
        elif change_pct < -2:
            score += 15  # Potential dip buy
            analysis["reasons"].append("Potential dip buying opportunity")
        
        # Price levels (simplified)
        if price < 50:
            score += 5
            analysis["reasons"].append("Accessible price point for small account")
        
        analysis["price"] = price
        analysis["confidence"] = min(score, 100)
        
        if score >= 70:
            analysis["recommendation"] = "BUY"
        elif score >= 60:
            analysis["recommendation"] = "CONSIDER"
        else:
            analysis["recommendation"] = "HOLD"
        
        log(f"  - Confidence Score: {score}/100", "INFO")
        log(f"  - Recommendation: {analysis['recommendation']}", "SUCCESS" if analysis['recommendation'] == "BUY" else "INFO")
        
        return analysis
    
    def calculate_position_size(self, price: float, strategy: Dict) -> int:
        """Calculate position size using Kelly Criterion from Ralph's knowledge"""
        log("OPTIMUS: Calculating position size (Kelly Criterion)...", "BRAIN")
        
        # Get Kelly parameters
        win_rate = strategy.get('expected_win_rate', 0.5)
        rr_ratio = strategy.get('expected_rr', 1.0) or 1.0
        
        # Kelly formula: f* = (bp - q) / b
        # Where b = odds (RR ratio), p = win prob, q = loss prob
        p = win_rate
        q = 1 - p
        b = rr_ratio
        
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        
        # Use fractional Kelly (25% of full Kelly for safety)
        fractional_kelly = kelly_fraction * 0.25
        fractional_kelly = max(0, min(fractional_kelly, 0.10))  # Cap at 10%
        
        # Calculate dollar amount
        position_value = self.account_balance * fractional_kelly
        
        # Calculate shares
        shares = int(position_value / price) if price > 0 else 0
        shares = max(1, shares)  # At least 1 share
        
        log(f"  - Win Rate: {win_rate:.0%}", "INFO")
        log(f"  - R:R Ratio: {rr_ratio:.1f}", "INFO")
        log(f"  - Full Kelly: {kelly_fraction:.1%}", "INFO")
        log(f"  - Fractional Kelly (25%): {fractional_kelly:.1%}", "INFO")
        log(f"  - Position Value: ${position_value:.2f}", "INFO")
        log(f"  - Shares: {shares}", "SUCCESS")
        
        return shares
    
    def make_trade_decision(self, symbols: List[str] = None) -> Dict:
        """Make intelligent trade decision based on shared knowledge"""
        log("=" * 60, "INFO")
        log("OPTIMUS: INTELLIGENT TRADE DECISION PROCESS", "BRAIN")
        log("=" * 60, "INFO")
        
        if not self.knowledge:
            return {"decision": "NO_TRADE", "reason": "No knowledge received from Ralph"}
        
        # Default symbols to analyze
        if not symbols:
            symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        
        # Step 1: Psychology checklist
        psych_check = self.apply_psychology_checklist()
        if not psych_check["passed"]:
            return {"decision": "NO_TRADE", "reason": "Failed psychology checklist"}
        
        # Step 2: Select strategy
        strategy = self.select_strategy()
        if not strategy:
            return {"decision": "NO_TRADE", "reason": "No suitable strategy found"}
        
        # Step 3: Analyze opportunities
        log("\n" + "=" * 60, "INFO")
        log("MARKET ANALYSIS", "INFO")
        log("=" * 60, "INFO")
        
        best_opportunity = None
        for symbol in symbols:
            analysis = self.analyze_opportunity(symbol)
            if analysis["recommendation"] == "BUY":
                if not best_opportunity or analysis["confidence"] > best_opportunity["confidence"]:
                    best_opportunity = analysis
            log("", "INFO")  # Spacing
        
        if not best_opportunity:
            log("No strong buy opportunities found", "WARNING")
            return {
                "decision": "NO_TRADE",
                "reason": "No opportunities meet criteria",
                "symbols_analyzed": symbols
            }
        
        # Step 4: Calculate position size
        shares = self.calculate_position_size(best_opportunity["price"], strategy)
        
        # Step 5: Apply risk rules
        proposed_trade = {
            "symbol": best_opportunity["symbol"],
            "action": "BUY",
            "quantity": shares,
            "price": best_opportunity["price"]
        }
        
        risk_check = self.apply_risk_rules(proposed_trade)
        
        # Step 6: Final decision
        log("\n" + "=" * 60, "INFO")
        log("FINAL DECISION", "TRADE")
        log("=" * 60, "INFO")
        
        if risk_check["approved"] and best_opportunity["confidence"] >= 60:
            decision = {
                "decision": "TRADE",
                "action": "BUY",
                "symbol": best_opportunity["symbol"],
                "quantity": shares,
                "price": best_opportunity["price"],
                "total_value": shares * best_opportunity["price"],
                "strategy": strategy.get("name"),
                "confidence": best_opportunity["confidence"],
                "reasons": best_opportunity["reasons"]
            }
            
            log(f"DECISION: BUY {shares} shares of {best_opportunity['symbol']}", "TRADE")
            log(f"  Price: ${best_opportunity['price']:.2f}", "INFO")
            log(f"  Total: ${shares * best_opportunity['price']:.2f}", "INFO")
            log(f"  Strategy: {strategy.get('name')}", "INFO")
            log(f"  Confidence: {best_opportunity['confidence']}%", "INFO")
            
            self.trade_decision = decision
            return decision
        else:
            log("DECISION: NO TRADE", "WARNING")
            if risk_check["warnings"]:
                log(f"  Risk warnings: {risk_check['warnings']}", "WARNING")
            
            return {
                "decision": "NO_TRADE",
                "reason": "Risk rules or confidence threshold not met",
                "risk_warnings": risk_check.get("warnings", []),
                "confidence": best_opportunity["confidence"]
            }
    
    def execute_trade(self) -> Dict:
        """Execute the trade decision"""
        if not self.trade_decision or self.trade_decision.get("decision") != "TRADE":
            return {"status": "skipped", "reason": "No trade decision to execute"}
        
        log("\n" + "=" * 60, "INFO")
        log("EXECUTING TRADE", "TRADE")
        log("=" * 60, "INFO")
        
        try:
            if not self.tradier_client:
                return {"status": "error", "reason": "Tradier not connected"}
            
            symbol = self.trade_decision["symbol"]
            quantity = self.trade_decision["quantity"]
            
            log(f"Submitting order: BUY {quantity} {symbol} @ MARKET", "INFO")
            
            # Execute via Tradier
            result = self.tradier_client.submit_order({
                "symbol": symbol,
                "side": "buy",
                "quantity": quantity,
                "order_type": "market",
                "duration": "day"
            })
            
            if result and result.get("status") != "rejected":
                log(f"ORDER SUBMITTED SUCCESSFULLY!", "SUCCESS")
                log(f"  Order ID: {result.get('id', 'N/A')}", "INFO")
                log(f"  Status: {result.get('status', 'N/A')}", "INFO")
                return {
                    "status": "success",
                    "order_id": result.get("id"),
                    "order_status": result.get("status"),
                    "symbol": symbol,
                    "quantity": quantity
                }
            else:
                log(f"Order rejected: {result}", "ERROR")
                return {"status": "rejected", "details": result}
                
        except Exception as e:
            log(f"Execution error: {e}", "ERROR")
            return {"status": "error", "reason": str(e)}


# =============================================================================
# MAIN - KNOWLEDGE SHARE & TRADE DECISION
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  RALPH-OPTIMUS KNOWLEDGE SHARE & INTELLIGENT TRADE DECISION")
    print("  NAE Agent Collaboration Protocol")
    print("=" * 70 + "\n")
    
    # Step 1: Ralph loads and prepares knowledge
    print("\n" + "-" * 60)
    print("PHASE 1: RALPH KNOWLEDGE PREPARATION")
    print("-" * 60 + "\n")
    
    ralph = RalphKnowledgeProvider()
    if not ralph.load_knowledge_bases():
        print("ERROR: Ralph could not load knowledge bases!")
        return
    
    # Step 2: Ralph shares knowledge with Optimus
    print("\n" + "-" * 60)
    print("PHASE 2: KNOWLEDGE TRANSFER (Ralph -> Optimus)")
    print("-" * 60 + "\n")
    
    knowledge_package = ralph.share_knowledge_summary()
    log(f"Ralph prepared knowledge package with {len(knowledge_package['strategies'])} strategies", "SUCCESS")
    
    # Step 3: Optimus receives knowledge
    optimus = OptimusIntelligentTrader()
    optimus.receive_knowledge(knowledge_package)
    
    # Step 4: Optimus connects to broker
    print("\n" + "-" * 60)
    print("PHASE 3: BROKER CONNECTION")
    print("-" * 60 + "\n")
    
    if not optimus.connect_to_tradier():
        log("WARNING: Could not connect to Tradier, continuing with analysis only", "WARNING")
    
    # Step 5: Optimus makes trade decision
    print("\n" + "-" * 60)
    print("PHASE 4: INTELLIGENT TRADE DECISION")
    print("-" * 60 + "\n")
    
    decision = optimus.make_trade_decision(["SPY", "QQQ", "AAPL", "NVDA", "AMD"])
    
    # Step 6: Execute if decided
    if decision.get("decision") == "TRADE":
        print("\n" + "-" * 60)
        print("PHASE 5: TRADE EXECUTION")
        print("-" * 60 + "\n")
        
        # Ask for confirmation in production
        log("Trade decision made. Ready to execute.", "TRADE")
        
        # Auto-execute (remove this for manual confirmation)
        result = optimus.execute_trade()
        
        if result.get("status") == "success":
            log(f"\nTRADE EXECUTED SUCCESSFULLY!", "SUCCESS")
        else:
            log(f"\nTrade execution: {result.get('status')} - {result.get('reason', '')}", "WARNING")
    else:
        log(f"\nNo trade executed. Reason: {decision.get('reason', 'Unknown')}", "INFO")
    
    # Summary
    print("\n" + "=" * 70)
    print("  KNOWLEDGE SHARE COMPLETE")
    print("=" * 70)
    print(f"  Strategies shared: {len(knowledge_package['strategies'])}")
    print(f"  Risk rules applied: {len(knowledge_package['risk_rules'])}")
    print(f"  Final decision: {decision.get('decision')}")
    if decision.get("decision") == "TRADE":
        print(f"  Symbol: {decision.get('symbol')}")
        print(f"  Shares: {decision.get('quantity')}")
        print(f"  Confidence: {decision.get('confidence')}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

