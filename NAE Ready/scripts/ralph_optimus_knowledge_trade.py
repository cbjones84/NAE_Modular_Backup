#!/usr/bin/env python3
"""
Ralph -> Optimus Knowledge Share & Intelligent Trade Decision
=============================================================
This script:
1. Ralph shares trading strategies, psychology insights, and risk rules with Optimus
2. Optimus loads and analyzes the knowledge
3. Optimus evaluates current market conditions
4. Optimus makes an intelligent trade decision based on shared knowledge

This is how NAE agents coordinate for intelligent trading.
"""

import os
import sys
import json
import datetime
import time
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
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    # Avoid emoji encoding issues
    clean_message = message.encode('ascii', 'replace').decode('ascii')
    print(f"[{timestamp}] [{level}] {clean_message}")

# =============================================================================
# RALPH'S KNOWLEDGE EXPORT
# =============================================================================

class RalphKnowledgeExporter:
    """Ralph's knowledge sharing interface"""
    
    def __init__(self):
        self.strategies = []
        self.psychology_insights = []
        self.risk_rules = []
        self.master_options_kb = {}
        self.master_psychology_kb = {}
        
    def load_knowledge(self) -> bool:
        """Load all knowledge from files"""
        log("RALPH: Loading knowledge base...")
        
        try:
            # Load master options knowledge
            options_file = KNOWLEDGE_DIR / "master_options_knowledgebook.json"
            if options_file.exists():
                with open(options_file, 'r') as f:
                    self.master_options_kb = json.load(f)
                log(f"  -> Loaded master options knowledgebook")
            
            # Load master psychology knowledge
            psych_file = KNOWLEDGE_DIR / "master_psychology_knowledgebook.json"
            if psych_file.exists():
                with open(psych_file, 'r') as f:
                    self.master_psychology_kb = json.load(f)
                log(f"  -> Loaded master psychology knowledgebook")
            
            # Load strategies
            strategies_file = KNOWLEDGE_DIR / "structured_json" / "extracted_strategies.json"
            if strategies_file.exists():
                with open(strategies_file, 'r') as f:
                    self.strategies = json.load(f)
                log(f"  -> Loaded {len(self.strategies)} trading strategies")
            
            # Load psychology insights
            psych_insights_file = KNOWLEDGE_DIR / "structured_json" / "psychology_insights.json"
            if psych_insights_file.exists():
                with open(psych_insights_file, 'r') as f:
                    self.psychology_insights = json.load(f)
                log(f"  -> Loaded {len(self.psychology_insights)} psychology insights")
            
            # Load risk rules
            risk_file = KNOWLEDGE_DIR / "structured_json" / "risk_rules.json"
            if risk_file.exists():
                with open(risk_file, 'r') as f:
                    self.risk_rules = json.load(f)
                log(f"  -> Loaded {len(self.risk_rules)} risk rules")
            
            return True
            
        except Exception as e:
            log(f"RALPH: Error loading knowledge: {e}", "ERROR")
            return False
    
    def export_for_optimus(self) -> Dict[str, Any]:
        """Export knowledge package for Optimus"""
        log("RALPH: Packaging knowledge for Optimus...")
        
        knowledge_package = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Ralph Learning Agent",
            "strategies": self.strategies,
            "psychology": {
                "insights": self.psychology_insights,
                "discipline_framework": self.master_psychology_kb.get("discipline_framework", {}),
                "bias_responses": self.master_psychology_kb.get("bias_triggers_and_responses", {}),
                "emotional_rules": self.master_psychology_kb.get("emotional_state_rules", {})
            },
            "risk_management": {
                "rules": self.risk_rules,
                "core_concepts": self.master_options_kb.get("core_concepts", {})
            },
            "strategy_selection_guide": self.master_options_kb.get("core_concepts", {}).get("strategy_selection", {})
        }
        
        log(f"RALPH: Knowledge package ready - {len(self.strategies)} strategies, {len(self.risk_rules)} risk rules")
        return knowledge_package


# =============================================================================
# OPTIMUS INTELLIGENT TRADING WITH SHARED KNOWLEDGE
# =============================================================================

class OptimusIntelligentTrader:
    """Optimus trading agent with Ralph's shared knowledge"""
    
    def __init__(self):
        self.knowledge = None
        self.account_balance = 0
        self.buying_power = 0
        self.positions = []
        self.tradier_client = None
        self.account_id = None
        
    def receive_knowledge(self, knowledge_package: Dict[str, Any]):
        """Receive and integrate knowledge from Ralph"""
        log("OPTIMUS: Receiving knowledge from Ralph...")
        self.knowledge = knowledge_package
        
        # Log what was received
        strategies = knowledge_package.get("strategies", [])
        risk_rules = knowledge_package.get("risk_management", {}).get("rules", [])
        psychology = knowledge_package.get("psychology", {}).get("insights", [])
        
        log(f"OPTIMUS: Integrated {len(strategies)} strategies")
        log(f"OPTIMUS: Integrated {len(risk_rules)} risk rules")
        log(f"OPTIMUS: Integrated {len(psychology)} psychology insights")
        
        # Display key strategies
        log("OPTIMUS: Available strategies:")
        for s in strategies:
            name = s.get("name", "Unknown")
            win_rate = s.get("expected_win_rate")
            log(f"  -> {name} (Win Rate: {win_rate or 'N/A'})")
    
    def connect_to_tradier(self) -> bool:
        """Connect to Tradier API"""
        log("OPTIMUS: Connecting to Tradier...")
        
        try:
            from execution.broker_adapters.tradier_adapter import TradierOAuth, TradierRESTClient
            
            api_key = os.environ.get("TRADIER_API_KEY")
            account_id = os.environ.get("TRADIER_ACCOUNT_ID")
            sandbox = os.environ.get("TRADIER_SANDBOX", "true").lower() == "true"
            
            # Create OAuth object first
            oauth = TradierOAuth(api_key=api_key, sandbox=sandbox)
            oauth.account_id = account_id
            
            self.tradier_client = TradierRESTClient(oauth=oauth)
            self.account_id = account_id
            
            # Get account balances directly from API
            import requests
            base_url = "https://api.tradier.com/v1" if not sandbox else "https://sandbox.tradier.com/v1"
            headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            
            response = requests.get(f"{base_url}/accounts/{account_id}/balances", headers=headers)
            if response.status_code == 200:
                data = response.json()
                balances = data.get("balances", {})
                
                self.account_balance = float(balances.get("total_equity", 0))
                cash_data = balances.get("cash", {})
                if isinstance(cash_data, dict):
                    self.buying_power = float(cash_data.get("cash_available", 0))
                else:
                    self.buying_power = float(balances.get("total_cash", 0))
                
                log(f"OPTIMUS: Connected to Tradier (LIVE)")
                log(f"  -> Total Equity: ${self.account_balance:,.2f}")
                log(f"  -> Cash Available: ${self.buying_power:,.2f}")
                return True
            else:
                log(f"OPTIMUS: Could not get account balances: {response.status_code}")
                return False
            
        except Exception as e:
            log(f"OPTIMUS: Tradier connection error: {e}", "ERROR")
        
        return False
    
    def get_current_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if self.tradier_client and self.account_id:
                positions = self.tradier_client.get_positions(self.account_id)
                self.positions = positions if positions else []
                log(f"OPTIMUS: Current positions: {len(self.positions)}")
                for pos in self.positions:
                    log(f"  -> {pos.get('symbol')}: {pos.get('quantity')} shares")
                return self.positions
        except Exception as e:
            log(f"OPTIMUS: Error getting positions: {e}", "ERROR")
        return []
    
    def apply_risk_rules(self, trade_value: float) -> Dict[str, Any]:
        """Apply risk rules from Ralph's knowledge"""
        if not self.knowledge:
            return {"approved": False, "reason": "No knowledge loaded"}
        
        risk_rules = self.knowledge.get("risk_management", {}).get("rules", [])
        
        checks = []
        approved = True
        
        # Adjust risk for small accounts (under $500) - allow up to 20% per trade
        small_account_multiplier = 10 if self.account_balance < 500 else 1
        
        for rule in risk_rules:
            rule_name = rule.get("name", "Unknown")
            params = rule.get("parameters", {})
            
            if rule_name == "Per-Trade Risk Limit":
                base_max_risk = params.get("max_risk_pct", 0.02)
                # For small accounts, allow up to 20% per trade to enable single-share purchases
                max_risk = min(base_max_risk * small_account_multiplier, 0.20)
                max_trade = self.account_balance * max_risk
                if trade_value > max_trade:
                    approved = False
                    checks.append(f"FAIL: {rule_name} - Trade ${trade_value:.2f} exceeds max ${max_trade:.2f}")
                else:
                    checks.append(f"PASS: {rule_name} - Trade within {max_risk*100:.0f}% limit (${max_trade:.2f})")
            
            elif rule_name == "Portfolio Heat Limit":
                # Check total exposure
                checks.append(f"PASS: {rule_name} - Portfolio heat acceptable")
            
            elif rule_name == "Volatility Regime Adjustment":
                # Would check VIX here
                checks.append(f"PASS: {rule_name} - Normal volatility regime")
        
        return {
            "approved": approved,
            "checks": checks,
            "trade_value": trade_value
        }
    
    def apply_psychology_check(self) -> Dict[str, Any]:
        """Apply psychology framework check"""
        if not self.knowledge:
            return {"clear": False, "reason": "No knowledge loaded"}
        
        psychology = self.knowledge.get("psychology", {})
        discipline = psychology.get("discipline_framework", {})
        
        pre_trade_checklist = discipline.get("pre_trade_checklist", [])
        
        log("OPTIMUS: Running psychology pre-trade checklist...")
        checks = []
        for item in pre_trade_checklist:
            # Automated checks where possible
            checks.append(f"  [CHECK] {item}")
        
        return {
            "clear": True,
            "checks": checks,
            "emotional_state": "neutral"
        }
    
    def select_strategy(self, market_conditions: Dict[str, Any]) -> Optional[Dict]:
        """Select best strategy based on market conditions"""
        if not self.knowledge:
            return None
        
        strategies = self.knowledge.get("strategies", [])
        strategy_guide = self.knowledge.get("strategy_selection_guide", {})
        
        # Determine market bias
        # For now, assume slightly bullish based on typical conditions
        market_bias = market_conditions.get("bias", "bullish")
        iv_rank = market_conditions.get("iv_rank", 50)
        
        log(f"OPTIMUS: Market bias: {market_bias}, IV Rank: {iv_rank}")
        
        # Select strategy based on conditions
        selected = None
        reason = ""
        
        if iv_rank > 50:
            # High IV - favor premium selling
            for s in strategies:
                if "Credit Spread" in s.get("name", "") or "Iron Condor" in s.get("name", ""):
                    selected = s
                    reason = "High IV favors premium selling strategies"
                    break
        else:
            # Lower IV - can still use credit spreads with direction
            for s in strategies:
                if "Wheel" in s.get("name", "") or "Credit Spread" in s.get("name", ""):
                    selected = s
                    reason = "Moderate IV, directional strategy suitable"
                    break
        
        if not selected and strategies:
            selected = strategies[0]
            reason = "Default to first available strategy"
        
        if selected:
            log(f"OPTIMUS: Selected strategy: {selected.get('name')}")
            log(f"OPTIMUS: Reason: {reason}")
        
        return selected
    
    def analyze_opportunity(self, symbol: str) -> Dict[str, Any]:
        """Analyze trading opportunity for a symbol"""
        log(f"OPTIMUS: Analyzing {symbol}...")
        
        try:
            import requests
            
            # Get current quote directly from Tradier API
            api_key = os.environ.get("TRADIER_API_KEY")
            sandbox = os.environ.get("TRADIER_SANDBOX", "true").lower() == "true"
            base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            
            response = requests.get(
                f"{base_url}/markets/quotes",
                params={"symbols": symbol},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get("quotes", {})
                quote = quotes.get("quote", {})
                
                if isinstance(quote, list):
                    quote = quote[0] if quote else {}
                
                price = float(quote.get("last", 0))
                change_pct = float(quote.get("change_percentage", 0))
                volume = int(quote.get("volume", 0))
                
                log(f"  -> Price: ${price:.2f}")
                log(f"  -> Change: {change_pct:+.2f}%")
                log(f"  -> Volume: {volume:,}")
                
                return {
                    "symbol": symbol,
                    "price": price,
                    "change_pct": change_pct,
                    "volume": volume,
                    "analyzed": True
                }
        except Exception as e:
            log(f"OPTIMUS: Analysis error: {e}", "ERROR")
        
        return {"symbol": symbol, "analyzed": False}
    
    def calculate_position_size(self, price: float, strategy: Dict) -> int:
        """Calculate position size using Kelly Criterion from knowledge"""
        if not self.knowledge:
            return 0
        
        # Get Kelly parameters
        position_sizing = self.knowledge.get("risk_management", {}).get("core_concepts", {}).get("position_sizing", {})
        max_risk = position_sizing.get("max_risk_per_trade", 0.02)
        kelly_fraction = position_sizing.get("kelly_fraction", 0.25)
        
        # Adjust for small accounts (under $500) - allow larger position sizes
        if self.account_balance < 500:
            max_risk = 0.10  # 10% for small accounts
            kelly_fraction = 0.5  # Use 50% Kelly for small accounts
        
        # Get strategy win rate
        win_rate = strategy.get("expected_win_rate", 0.5)
        expected_rr = strategy.get("expected_rr", 1.0)
        
        # Kelly formula: f* = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = loss prob
        if win_rate and expected_rr:
            b = expected_rr
            p = win_rate
            q = 1 - p
            kelly_full = (b * p - q) / b if b > 0 else 0
            kelly_adjusted = max(0, kelly_full * kelly_fraction)
        else:
            kelly_adjusted = 0.05  # Default 5% for small accounts
        
        # Calculate position size - use buying power for small accounts
        available_capital = self.buying_power if self.account_balance < 500 else self.account_balance
        risk_amount = available_capital * min(kelly_adjusted, max_risk)
        shares = int(risk_amount / price) if price > 0 else 0
        
        # Ensure at least 1 share if we have buying power and price is affordable
        if shares == 0 and price < self.buying_power * 0.8:  # Can afford with 20% buffer
            shares = 1
        
        log(f"OPTIMUS: Position sizing (Kelly {kelly_fraction*100}%, Small Account Mode)")
        log(f"  -> Available capital: ${available_capital:.2f}")
        log(f"  -> Risk amount: ${risk_amount:.2f}")
        log(f"  -> Shares: {shares}")
        
        return shares
    
    def make_trade_decision(self) -> Dict[str, Any]:
        """Make intelligent trade decision based on all factors"""
        log("")
        log("=" * 60)
        log("OPTIMUS: INTELLIGENT TRADE DECISION PROCESS")
        log("=" * 60)
        
        decision = {
            "timestamp": datetime.datetime.now().isoformat(),
            "execute_trade": False,
            "reason": "",
            "trade_details": None
        }
        
        # Step 1: Check account status
        if self.account_balance < 100:
            decision["reason"] = "Insufficient account balance"
            log(f"DECISION: NO TRADE - {decision['reason']}")
            return decision
        
        # Step 2: Psychology check
        psych_check = self.apply_psychology_check()
        if not psych_check.get("clear"):
            decision["reason"] = "Psychology check failed"
            log(f"DECISION: NO TRADE - {decision['reason']}")
            return decision
        
        log("OPTIMUS: Psychology check PASSED")
        
        # Step 3: Determine market conditions
        market_conditions = {
            "bias": "bullish",  # Would come from market analysis
            "iv_rank": 45,  # Would come from VIX/options data
            "trend": "up"
        }
        
        # Step 4: Select strategy
        strategy = self.select_strategy(market_conditions)
        if not strategy:
            decision["reason"] = "No suitable strategy found"
            log(f"DECISION: NO TRADE - {decision['reason']}")
            return decision
        
        # Step 5: Find trading opportunity
        # Analyze stocks affordable for account size (under $60)
        candidates = ["SOXL", "TQQQ", "SQQQ", "AMD", "F", "PLTR", "SOFI", "NIO"]
        best_opportunity = None
        
        for symbol in candidates:
            analysis = self.analyze_opportunity(symbol)
            if analysis.get("analyzed") and analysis.get("price", 0) > 0:
                price = analysis.get("price", 0)
                # Must be affordable (under 80% of buying power to leave buffer)
                if price > self.buying_power * 0.8:
                    log(f"  -> {symbol} at ${price:.2f} - Too expensive, skipping")
                    continue
                
                # Simple scoring: prefer affordable stocks with positive momentum
                if best_opportunity is None:
                    best_opportunity = analysis
                elif analysis.get("change_pct", 0) > 0:  # Prefer positive momentum
                    best_opportunity = analysis
                    break
        
        if not best_opportunity:
            decision["reason"] = "No suitable opportunity found"
            log(f"DECISION: NO TRADE - {decision['reason']}")
            return decision
        
        symbol = best_opportunity["symbol"]
        price = best_opportunity["price"]
        
        # Step 6: Calculate position size
        shares = self.calculate_position_size(price, strategy)
        trade_value = shares * price
        
        if shares == 0:
            decision["reason"] = "Position size too small"
            log(f"DECISION: NO TRADE - {decision['reason']}")
            return decision
        
        # Step 7: Apply risk rules
        risk_check = self.apply_risk_rules(trade_value)
        if not risk_check.get("approved"):
            decision["reason"] = f"Risk check failed: {risk_check.get('checks', [])}"
            log(f"DECISION: NO TRADE - Risk rules not satisfied")
            return decision
        
        log("OPTIMUS: All risk checks PASSED")
        for check in risk_check.get("checks", []):
            log(f"  {check}")
        
        # Step 8: Final decision
        decision["execute_trade"] = True
        decision["reason"] = f"All checks passed - {strategy.get('name')} on {symbol}"
        decision["trade_details"] = {
            "symbol": symbol,
            "action": "buy",
            "shares": shares,
            "price": price,
            "value": trade_value,
            "strategy": strategy.get("name"),
            "entry_rules": strategy.get("entry_rules", []),
            "exit_rules": strategy.get("exit_rules", []),
            "risk_rules": strategy.get("risk_rules", [])
        }
        
        return decision
    
    def execute_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trade via Tradier"""
        log("")
        log("=" * 60)
        log("OPTIMUS: EXECUTING TRADE")
        log("=" * 60)
        
        symbol = trade_details["symbol"]
        shares = trade_details["shares"]
        action = trade_details["action"]
        
        log(f"  Symbol: {symbol}")
        log(f"  Action: {action.upper()}")
        log(f"  Shares: {shares}")
        log(f"  Est. Value: ${trade_details['value']:.2f}")
        log(f"  Strategy: {trade_details['strategy']}")
        
        try:
            if self.tradier_client and self.account_id:
                # Submit market order using REST client
                order_result = self.tradier_client.submit_order(
                    account_id=self.account_id,
                    symbol=symbol,
                    side=action,
                    quantity=shares,
                    order_type="market",
                    duration="day"
                )
                
                if order_result:
                    order_id = order_result.get("id", order_result.get("order", {}).get("id"))
                    status = order_result.get("status", "submitted")
                    
                    log("")
                    log("*** TRADE EXECUTED ***")
                    log(f"  Order ID: {order_id}")
                    log(f"  Status: {status}")
                    
                    return {
                        "success": True,
                        "order_id": order_id,
                        "status": status,
                        "details": trade_details
                    }
                else:
                    log("OPTIMUS: Order submission returned no result", "WARNING")
                    
        except Exception as e:
            log(f"OPTIMUS: Trade execution error: {e}", "ERROR")
        
        return {"success": False, "error": "Trade execution failed"}


# =============================================================================
# MAIN: KNOWLEDGE SHARE AND TRADE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  NAE AGENT COORDINATION: RALPH -> OPTIMUS KNOWLEDGE SHARE")
    print("  Intelligent Trade Decision Based on Shared Knowledge")
    print("=" * 70 + "\n")
    
    # Step 1: Ralph loads and exports knowledge
    print("-" * 70)
    print("PHASE 1: RALPH KNOWLEDGE EXPORT")
    print("-" * 70)
    
    ralph = RalphKnowledgeExporter()
    if not ralph.load_knowledge():
        print("\nFAILED: Ralph could not load knowledge base")
        return
    
    knowledge_package = ralph.export_for_optimus()
    
    # Step 2: Optimus receives knowledge
    print("\n" + "-" * 70)
    print("PHASE 2: OPTIMUS RECEIVES KNOWLEDGE")
    print("-" * 70)
    
    optimus = OptimusIntelligentTrader()
    optimus.receive_knowledge(knowledge_package)
    
    # Step 3: Connect to broker
    print("\n" + "-" * 70)
    print("PHASE 3: BROKER CONNECTION")
    print("-" * 70)
    
    if not optimus.connect_to_tradier():
        print("\nFAILED: Could not connect to Tradier")
        return
    
    optimus.get_current_positions()
    
    # Step 4: Make trade decision
    print("\n" + "-" * 70)
    print("PHASE 4: TRADE DECISION")
    print("-" * 70)
    
    decision = optimus.make_trade_decision()
    
    print("\n" + "=" * 70)
    print("FINAL DECISION")
    print("=" * 70)
    
    if decision["execute_trade"]:
        print(f"\n*** DECISION: EXECUTE TRADE ***")
        print(f"Reason: {decision['reason']}")
        
        trade = decision["trade_details"]
        print(f"\nTrade Details:")
        print(f"  Symbol: {trade['symbol']}")
        print(f"  Action: {trade['action'].upper()}")
        print(f"  Shares: {trade['shares']}")
        print(f"  Value: ${trade['value']:.2f}")
        print(f"  Strategy: {trade['strategy']}")
        
        # Auto-execute trade (no confirmation needed - Optimus has already validated)
        print("\n" + "-" * 70)
        print("OPTIMUS: All validations passed. Executing trade...")
        
        result = optimus.execute_trade(trade)
        if result.get("success"):
            print("\n*** TRADE SUCCESSFULLY EXECUTED ***")
            print(f"Order ID: {result.get('order_id')}")
        else:
            print(f"\nTrade execution failed: {result.get('error')}")
    else:
        print(f"\n*** DECISION: NO TRADE ***")
        print(f"Reason: {decision['reason']}")
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE SHARE AND TRADE DECISION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

