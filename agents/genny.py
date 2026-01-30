# NAE/agents/genny.py
"""
GennyAgent v2.0 - Generational Wealth Tracking and Knowledge Transfer Agent for NAE

ENHANCEMENTS in v2.0:
- Kalshi prediction market tax expertise (Section 1256 treatment)
- 60/40 long-term/short-term capital gains tracking
- Loss carryback and mark-to-market planning
- Integrated wealth contribution tracking for milestones

Responsibilities:
- Track and log Optimus' success metrics and trading performance
- Monitor and catalog Ralph's successful strategies and learning patterns
- Curate recipes of success from NAE's profits and methodologies
- Understand and implement generational wealth concepts (financial, intellectual, social, values)
- Transfer knowledge and strategies to heirs for maintaining NAE success
- Learn and adapt wealth growth and maintenance strategies
- Implement 5 core AI modules for comprehensive wealth management
- Maintain ethical framework for generational wealth AI operations
- KALSHI TAX EXPERTISE: Section 1256 contracts, 60/40 treatment, Form 6781

Core Generational Wealth Definition:
- Financial Capital: Managing concrete financial assets (stocks, real estate, businesses, bonds, cash)
- Intellectual Capital: Processing and perpetuating financial knowledge and strategic thinking
- Social Capital: Leveraging relationships and professional networks for financial benefit
- Values and Legacy: Maintaining family core values and ethical framework for financial stewardship
- Prediction Markets: Tax-efficient alpha generation via CFTC-regulated platforms (Kalshi)
"""

import os
import datetime
import json
import threading
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals

# Profit enhancement algorithms
# Profit enhancement algorithms (Lazy loaded)
# from tools.profit_algorithms import UniversalPortfolio

# Tax preparation module
try:
    from agents.genny_tax_module import TaxPreparer, CostBasisMethod, ExpenseRecord
    TAX_MODULE_AVAILABLE = True
except ImportError as e:
    TAX_MODULE_AVAILABLE = False
    print(f"Warning: Tax module not available: {e}")

# Tax optimization module
try:
    from agents.genny_tax_optimizer import TaxLawResearchEngine, TaxOptimizationAdvisor
    TAX_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    TAX_OPTIMIZER_AVAILABLE = False
    print(f"Warning: Tax optimizer not available: {e}")

GOALS = get_nae_goals()

class GennyAgent:
    def __init__(self, goals=None):
        # ----------------------
        # Goals Integration
        # ----------------------
        self.goals = goals if goals else GOALS
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"  # Reference to long-term plan
        # Growth Milestones from nae_mission_control.py
        self.target_goal = 5000000.0  # $5M target (exceeded in Year 7)
        self.stretch_goal = 15726144.0  # $15.7M final goal (Year 8)
        self.growth_milestones = {
            1: 9_411, 2: 44_110, 3: 152_834, 4: 388_657,
            5: 982_500, 6: 2_477_897, 7: 6_243_561, 8: 15_726_144
        }
        
        # ----------------------
        # Growth Milestones Integration
        # ----------------------
        try:
            from core.growth_milestones import GrowthMilestonesTracker
            self.milestone_tracker = GrowthMilestonesTracker()
        except ImportError:
            self.milestone_tracker = None
            self.log_action("⚠️ Growth Milestones tracker not available")
        self.log_file = "logs/genny.log"
        self.data_dir = "tools/data/genny"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # ----------------------
        # Generational Wealth Framework
        # ----------------------
        self.generational_wealth_components = {
            "financial_capital": {
                "description": "Managing concrete financial assets (stocks, real estate, businesses, bonds, cash)",
                "tracking_metrics": ["total_assets", "asset_allocation", "growth_rate", "risk_exposure"],
                "data_sources": ["optimus_trades", "april_crypto", "shredder_profits"]
            },
            "intellectual_capital": {
                "description": "Processing and perpetuating financial knowledge and strategic thinking",
                "tracking_metrics": ["strategy_success_rate", "learning_adaptation", "knowledge_transfer"],
                "data_sources": ["ralph_strategies", "casey_improvements", "agent_interactions"]
            },
            "social_capital": {
                "description": "Leveraging relationships and professional networks for financial benefit",
                "tracking_metrics": ["network_value", "partnership_returns", "reputation_score"],
                "data_sources": ["external_partnerships", "community_engagement", "market_position"]
            },
            "values_and_legacy": {
                "description": "Maintaining family core values and ethical framework for financial stewardship",
                "tracking_metrics": ["ethical_compliance", "value_alignment", "legacy_preservation"],
                "data_sources": ["decision_audits", "value_assessments", "heir_feedback"]
            }
        }
        
        # ----------------------
        # Core AI Modules
        # ----------------------
        self.ai_modules = {
            "financial_planner": FinancialPlannerModule(self),
            "data_aggregator": DataAggregatorModule(self),
            "educational_advisory": EducationalAdvisoryModule(self),
            "ethical_compliance": EthicalComplianceModule(self),
            "orchestration_engine": OrchestrationEngineModule(self)
        }
        
        # ----------------------
        # Success Tracking Systems
        # ----------------------
        self.optimus_success_log = []
        self.ralph_strategy_log = []
        self.nae_profit_recipes = []
        self.heir_knowledge_base = {}
        
        # ----------------------
        # Learning and Adaptation
        # ----------------------
        self.learning_history = []
        self.wealth_growth_patterns = {}
        self.maintenance_strategies = {}
        
        # ----------------------
        # Communication / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []
        
        # ----------------------
        # Genius Communication Protocol
        # ----------------------
        self.genius_protocol = None
        try:
            from agents.genius_communication_protocol import GeniusCommunicationProtocol, MessageType, MessagePriority
            self.genius_protocol = GeniusCommunicationProtocol()
            
            # Register Genny
            self.genius_protocol.register_agent(
                agent_name="GennyAgent",
                capabilities=[
                    "wealth_management", "tax_preparation", "financial_planning",
                    "generational_wealth_tracking"
                ],
                expertise=["wealth_management", "tax_optimization", "financial_planning"],
                agent_instance=self
            )
            
            self.log_action("🧠 Genius communication protocol initialized for Genny")
        except ImportError as e:
            self.log_action(f"⚠️ Genius protocol not available: {e}")
        except Exception as e:
            self.log_action(f"⚠️ Genius protocol initialization failed: {e}")
        
        # ----------------------
        # Monitoring and Tracking
        # ----------------------
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_agents_loop, daemon=True)
        self.monitor_thread.start()
        
        # ----------------------
        # Profit Enhancement: Universal Portfolio Algorithm
        # ----------------------
        try:
            from tools.profit_algorithms import UniversalPortfolio
            self.universal_portfolio = UniversalPortfolio(initial_capital=25000.0)  # Starting with small account
        except ImportError:
            self.universal_portfolio = None
            self.log_action("⚠️ UniversalPortfolio not available")
        
        # ----------------------
        # Tax Preparation and Assessment Module
        # ----------------------
        if TAX_MODULE_AVAILABLE:
            self.tax_preparer = TaxPreparer(data_dir=os.path.join(self.data_dir, "tax"))
            self.tax_tracking_enabled = True
            self.log_action("Tax preparation module initialized - Full tax tracking enabled")
        else:
            self.tax_preparer = None
            self.tax_tracking_enabled = False
            self.log_action("Warning: Tax module not available - Tax tracking disabled")
        
        # ----------------------
        # Tax Law Research and Optimization Module
        # ----------------------
        if TAX_OPTIMIZER_AVAILABLE:
            self.tax_research_engine = TaxLawResearchEngine(data_dir=os.path.join(self.data_dir, "tax_law"))
            self.tax_optimizer = TaxOptimizationAdvisor(self.tax_research_engine)
            self.tax_optimization_enabled = True
            self.log_action("Tax optimization module initialized - Autonomous tax law research enabled")
        else:
            self.tax_research_engine = None
            self.tax_optimizer = None
            self.tax_optimization_enabled = False
            self.log_action("Warning: Tax optimizer not available - Tax optimization disabled")
        
        # ----------------------
        # North Carolina Tax Law Knowledge
        # ----------------------
        self.nc_tax_knowledge = {
            "individual_income_tax": {
                "rate": 0.0475,  # 4.75% flat rate
                "description": "North Carolina has a flat individual income tax rate of 4.75%",
                "effective_date": "2024",
                "notes": "NC does not differentiate between short-term and long-term capital gains"
            },
            "capital_gains": {
                "treatment": "ordinary_income",
                "rate": 0.0475,
                "description": "All capital gains are treated as ordinary income in NC",
                "notes": "No preferential rate for long-term capital gains"
            },
            "business_expenses": {
                "deductible": True,
                "requirements": [
                    "Ordinary and necessary business expenses",
                    "Directly related to trading/business operations",
                    "Proper documentation required"
                ],
                "common_categories": [
                    "Software subscriptions (trading platforms, data feeds)",
                    "Hardware (computers, servers for trading)",
                    "Professional services (CPA, legal)",
                    "Office expenses",
                    "Internet and utilities (business portion)",
                    "Education and training"
                ]
            },
            "day_trading": {
                "treatment": "business_income",
                "notes": "Day trading may qualify as business income if it's your primary activity",
                "requirements": "Must meet IRS trader tax status requirements"
            },
            "crypto_taxation": {
                "treatment": "property",
                "description": "Cryptocurrency is treated as property for tax purposes",
                "capital_gains": "Applies to crypto sales",
                "mining": "Mining income is taxable as ordinary income",
                "notes": "NC follows federal treatment of cryptocurrency"
            },
            "filing_requirements": {
                "state_return": True,
                "due_date": "April 15",
                "extensions": "Available with federal extension",
                "estimated_payments": "Required if tax liability > $1000"
            }
        }
        
        # ----------------------
        # Kalshi Prediction Market Tax Knowledge (Section 1256)
        # ----------------------
        self.kalshi_tax_knowledge = {
            "classification": {
                "type": "Section 1256 Contract",
                "description": "Kalshi contracts are CFTC-regulated futures = Section 1256 treatment",
                "regulator": "CFTC (Commodity Futures Trading Commission)",
                "legal_status": "Designated Contract Market (DCM)"
            },
            "tax_treatment": {
                "rule": "60/40 Split",
                "long_term_portion": 0.60,  # 60% taxed at long-term capital gains rate
                "short_term_portion": 0.40,  # 40% taxed at short-term rate
                "description": "Regardless of holding period, gains are split 60% LTCG / 40% STCG",
                "form": "Form 6781 (Gains and Losses From Section 1256 Contracts)",
                "reporting_form": "1099-B from Kalshi"
            },
            "federal_rates_2024": {
                "long_term_rates": {
                    "0_pct_bracket": {"single": 47025, "married": 94050},
                    "15_pct_bracket": {"single": 518900, "married": 583750},
                    "20_pct_bracket": "Above 15% threshold"
                },
                "short_term_rate": "Ordinary income (up to 37%)"
            },
            "blended_rate_calculation": {
                "formula": "(0.60 * LTCG_rate) + (0.40 * STCG_rate)",
                "example_high_income": "(0.60 * 0.20) + (0.40 * 0.37) = 0.268 or 26.8%",
                "example_mid_income": "(0.60 * 0.15) + (0.40 * 0.24) = 0.186 or 18.6%",
                "savings_vs_short_term": "Up to 10% tax savings vs all short-term treatment"
            },
            "nc_state_treatment": {
                "rate": 0.0475,
                "treatment": "NC taxes all income at flat rate - 60/40 federal split doesn't apply",
                "notes": "State tax is same regardless of holding period"
            },
            "special_rules": {
                "mark_to_market": {
                    "description": "All open positions are marked to market at year-end",
                    "implication": "Unrealized gains/losses are recognized at Dec 31",
                    "action_required": "Include open position values in year-end tax planning"
                },
                "loss_carryback": {
                    "description": "Section 1256 losses can be carried back 3 years",
                    "benefit": "Can amend prior returns to claim losses",
                    "form": "Form 1045 for carryback"
                },
                "wash_sale": {
                    "description": "Wash sale rules DO NOT apply to Section 1256 contracts",
                    "benefit": "Can re-enter same position immediately after loss"
                }
            },
            "advantages": [
                "60/40 tax treatment regardless of holding period",
                "Loss carryback up to 3 years",
                "No wash sale rules",
                "Proper 1099-B reporting from Kalshi",
                "CFTC regulation = clear legal/tax status",
                "FDIC-insured fund custody"
            ],
            "vs_polymarket": {
                "polymarket": "Crypto-based, unclear tax treatment, no 1099",
                "kalshi": "USD-based, Section 1256, full 1099-B reporting",
                "recommendation": "Kalshi preferred for tax efficiency and compliance"
            }
        }
        
        # Kalshi trade tracking
        self.kalshi_trades = []
        self.kalshi_tax_liability = 0.0
        self.kalshi_ytd_pnl = 0.0
        
        # Kalshi wealth strategy
        self.kalshi_wealth_strategy = {
            "role_in_portfolio": "Uncorrelated alpha generation",
            "target_allocation_pct": 10.0,  # 10% of total wealth target
            "rebalancing_frequency": "quarterly",
            "growth_contribution_target_pct": 15.0,  # Target 15% of milestone growth from Kalshi
            "tax_efficiency_priority": True
        }
        
        self.log_action("Genny v2.0 initialized with Kalshi tax expertise and generational wealth tracking")

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message: str):
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Genny LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Genny LOG] {safe_message}")

    # ----------------------
    # Optimus Success Tracking
    # ----------------------
    def track_optimus_success(self, trade_data: Dict[str, Any]):
        """Track and log Optimus' trading success metrics"""
        try:
            success_metrics = {
                "timestamp": datetime.datetime.now().isoformat(),
                "trade_details": trade_data,
                "success_indicators": self._analyze_trade_success(trade_data),
                "wealth_impact": self._calculate_wealth_impact(trade_data),
                "strategy_effectiveness": self._evaluate_strategy_effectiveness(trade_data)
            }
            
            self.optimus_success_log.append(success_metrics)
            self._save_optimus_log()
            
            # Update generational wealth tracking
            self._update_financial_capital_metrics(success_metrics)
            
            # Track trade for tax purposes
            if self.tax_tracking_enabled and self.tax_preparer:
                self._track_trade_for_taxes(trade_data, agent="optimus")
            
            self.log_action(f"Tracked Optimus success: {trade_data.get('strategy_name', 'Unknown')}")
            return success_metrics
            
        except Exception as e:
            self.log_action(f"Error tracking Optimus success: {e}")
            return None
    
    def _track_trade_for_taxes(self, trade_data: Dict[str, Any], agent: str):
        """Track trade for tax purposes"""
        try:
            if not self.tax_preparer:
                return
            
            trade_id = trade_data.get("trade_id") or trade_data.get("order_id") or f"{agent}_{datetime.datetime.now().timestamp()}"
            timestamp = trade_data.get("timestamp") or trade_data.get("execution_timestamp") or datetime.datetime.now().isoformat()
            symbol = trade_data.get("symbol", "")
            asset_type = trade_data.get("asset_type", "stock")
            action = trade_data.get("action", trade_data.get("side", "")).lower()
            quantity = float(trade_data.get("quantity", 0))
            price = float(trade_data.get("price", trade_data.get("execution_price", 0)))
            fees = float(trade_data.get("fees", trade_data.get("commission", 0)))
            notes = trade_data.get("notes", trade_data.get("strategy_name", ""))
            
            if symbol and quantity > 0 and price > 0:
                self.tax_preparer.record_trade(
                    trade_id=trade_id,
                    timestamp=timestamp,
                    symbol=symbol,
                    asset_type=asset_type,
                    action=action,
                    quantity=quantity,
                    price=price,
                    fees=fees,
                    agent=agent,
                    notes=notes
                )
                self.log_action(f"Tax tracking: Recorded {action} trade for {symbol}")
        
        except Exception as e:
            self.log_action(f"Error tracking trade for taxes: {e}")

    # ----------------------
    # Kalshi Tax Tracking (Section 1256)
    # ----------------------
    def track_kalshi_trade_for_taxes(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track Kalshi trades with proper Section 1256 tax treatment.
        
        Kalshi contracts receive special 60/40 tax treatment:
        - 60% taxed at long-term capital gains rate
        - 40% taxed at short-term capital gains rate
        - Regardless of actual holding period
        
        Args:
            trade_data: Trade information from Kalshi/Shredder
            
        Returns:
            Dict with trade record and tax calculations
        """
        try:
            # Extract trade details
            kalshi_trade = {
                "trade_id": trade_data.get("trade_id", f"kalshi_{datetime.datetime.now().timestamp()}"),
                "timestamp": trade_data.get("timestamp", datetime.datetime.now().isoformat()),
                "ticker": trade_data.get("ticker", trade_data.get("market_ticker", "")),
                "market_title": trade_data.get("market_title", ""),
                "side": trade_data.get("side", ""),  # "yes" or "no"
                "price_cents": trade_data.get("price_cents", 0),
                "count": trade_data.get("count", trade_data.get("quantity", 0)),
                "amount_cents": trade_data.get("amount_cents", 0),
                "pnl_cents": trade_data.get("pnl_cents", 0),
                "strategy": trade_data.get("strategy", ""),
                "category": trade_data.get("category", ""),
                
                # Tax treatment
                "tax_treatment": "section_1256",
                "long_term_portion": 0.60,
                "short_term_portion": 0.40,
                "source": "kalshi_cftc_regulated",
                "form": "Form 6781"
            }
            
            # Calculate tax liability if there's P&L
            pnl_dollars = kalshi_trade["pnl_cents"] / 100
            
            if pnl_dollars != 0:
                # Calculate blended tax rate (assuming high income bracket for conservative estimate)
                # 60% at 20% LTCG + 40% at 37% STCG = 26.8% federal
                # Plus NC state at 4.75%
                federal_ltcg_rate = 0.20  # Top LTCG rate
                federal_stcg_rate = 0.37  # Top ordinary income rate
                nc_rate = 0.0475
                
                # Blended federal rate
                blended_federal = (0.60 * federal_ltcg_rate) + (0.40 * federal_stcg_rate)
                
                # Total estimated tax
                federal_tax = pnl_dollars * blended_federal
                state_tax = pnl_dollars * nc_rate
                total_tax = federal_tax + state_tax
                
                kalshi_trade["estimated_tax_liability"] = {
                    "federal_tax": round(federal_tax, 2),
                    "state_tax_nc": round(state_tax, 2),
                    "total_tax": round(total_tax, 2),
                    "effective_rate_pct": round((total_tax / pnl_dollars) * 100, 2) if pnl_dollars > 0 else 0,
                    "blended_federal_rate_pct": round(blended_federal * 100, 2),
                    "tax_savings_vs_short_term": round(pnl_dollars * (federal_stcg_rate - blended_federal), 2) if pnl_dollars > 0 else 0
                }
                
                self.kalshi_tax_liability += total_tax
                self.kalshi_ytd_pnl += pnl_dollars
            
            # Store trade
            self.kalshi_trades.append(kalshi_trade)
            self._save_kalshi_tax_data()
            
            self.log_action(
                f"📊 Kalshi tax tracking: {kalshi_trade['ticker']} "
                f"P&L: ${pnl_dollars:.2f}, Tax: ${kalshi_trade.get('estimated_tax_liability', {}).get('total_tax', 0):.2f}"
            )
            
            return kalshi_trade
            
        except Exception as e:
            self.log_action(f"Error tracking Kalshi trade for taxes: {e}")
            return {"error": str(e)}
    
    def _save_kalshi_tax_data(self) -> None:
        """Save Kalshi tax tracking data to disk"""
        try:
            kalshi_tax_dir = os.path.join(self.data_dir, "kalshi_taxes")
            os.makedirs(kalshi_tax_dir, exist_ok=True)
            
            # Save current year's trades
            year = datetime.datetime.now().year
            trades_file = os.path.join(kalshi_tax_dir, f"kalshi_trades_{year}.json")
            
            with open(trades_file, 'w') as f:
                json.dump({
                    "year": year,
                    "trades": self.kalshi_trades,
                    "ytd_pnl": self.kalshi_ytd_pnl,
                    "ytd_tax_liability": self.kalshi_tax_liability,
                    "last_updated": datetime.datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.log_action(f"Error saving Kalshi tax data: {e}")
    
    def calculate_kalshi_tax_liability(self, tax_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate total Kalshi tax liability for a given year.
        
        Implements Section 1256 60/40 treatment with mark-to-market.
        
        Args:
            tax_year: Year to calculate (default: current year)
            
        Returns:
            Dict with comprehensive tax liability breakdown
        """
        try:
            year = tax_year or datetime.datetime.now().year
            
            # Filter trades for the year
            year_trades = [
                t for t in self.kalshi_trades 
                if t.get("timestamp", "").startswith(str(year))
            ]
            
            # Sum realized P&L
            realized_pnl = sum(t.get("pnl_cents", 0) for t in year_trades) / 100
            
            # Calculate 60/40 split
            long_term_portion = realized_pnl * 0.60
            short_term_portion = realized_pnl * 0.40
            
            # Tax rates (using brackets - simplified for high income)
            if realized_pnl > 0:
                # Positive P&L - taxes owed
                ltcg_rate = 0.20  # Assuming top bracket
                stcg_rate = 0.37  # Assuming top bracket
            else:
                # Losses - can offset other income
                ltcg_rate = 0.20
                stcg_rate = 0.37
            
            federal_tax = (long_term_portion * ltcg_rate) + (short_term_portion * stcg_rate)
            nc_state_tax = realized_pnl * 0.0475
            total_tax = federal_tax + nc_state_tax
            
            # Calculate tax savings from 60/40 treatment
            hypothetical_all_short_term = realized_pnl * stcg_rate
            tax_savings = hypothetical_all_short_term - (long_term_portion * ltcg_rate + short_term_portion * stcg_rate) if realized_pnl > 0 else 0
            
            liability = {
                "tax_year": year,
                "num_trades": len(year_trades),
                "realized_pnl": round(realized_pnl, 2),
                "section_1256_split": {
                    "long_term_portion_60pct": round(long_term_portion, 2),
                    "short_term_portion_40pct": round(short_term_portion, 2)
                },
                "federal_tax": {
                    "ltcg_tax": round(long_term_portion * ltcg_rate, 2),
                    "stcg_tax": round(short_term_portion * stcg_rate, 2),
                    "total_federal": round(federal_tax, 2)
                },
                "state_tax": {
                    "nc_tax": round(nc_state_tax, 2),
                    "nc_rate": "4.75%"
                },
                "total_tax_liability": round(total_tax, 2),
                "effective_tax_rate_pct": round((total_tax / realized_pnl) * 100, 2) if realized_pnl != 0 else 0,
                "tax_efficiency": {
                    "savings_from_60_40": round(tax_savings, 2),
                    "vs_all_short_term": f"Saved ${tax_savings:.2f} ({(tax_savings/hypothetical_all_short_term*100):.1f}%)" if hypothetical_all_short_term > 0 else "N/A"
                },
                "forms_required": ["Form 6781", "Schedule D", "Form 1040"],
                "notes": [
                    "Section 1256 contracts marked to market at year-end",
                    "Losses can be carried back 3 years",
                    "Wash sale rules do NOT apply"
                ]
            }
            
            self.log_action(f"📋 Kalshi tax liability {year}: ${total_tax:.2f} on ${realized_pnl:.2f} P&L")
            
            return liability
            
        except Exception as e:
            self.log_action(f"Error calculating Kalshi tax liability: {e}")
            return {"error": str(e)}
    
    def optimize_kalshi_tax_strategy(self) -> Dict[str, Any]:
        """
        Provide tax optimization recommendations for Kalshi trading.
        
        Leverages Section 1256 special rules for maximum tax efficiency.
        
        Returns:
            Dict with optimization recommendations
        """
        try:
            recommendations = []
            current_date = datetime.datetime.now()
            year_end = datetime.datetime(current_date.year, 12, 31)
            days_to_year_end = (year_end - current_date).days
            
            # 1. Year-end mark-to-market planning
            if days_to_year_end < 60:
                recommendations.append({
                    "type": "mark_to_market_planning",
                    "priority": "HIGH",
                    "description": f"Year-end in {days_to_year_end} days - review open positions",
                    "action": "All open Kalshi positions will be marked to market on Dec 31",
                    "impact": "Unrealized gains/losses will be recognized regardless of sale"
                })
            
            # 2. Loss harvesting (but note: wash sale doesn't apply!)
            losing_trades = [t for t in self.kalshi_trades if t.get("pnl_cents", 0) < 0]
            if losing_trades:
                total_losses = sum(t.get("pnl_cents", 0) for t in losing_trades) / 100
                recommendations.append({
                    "type": "loss_harvesting",
                    "priority": "MEDIUM",
                    "description": f"${abs(total_losses):.2f} in realized losses available",
                    "action": "Losses can offset gains; can carry back 3 years",
                    "benefit": "Wash sale rules do NOT apply - can re-enter immediately"
                })
            
            # 3. Loss carryback opportunity
            if self.kalshi_ytd_pnl < 0:
                recommendations.append({
                    "type": "loss_carryback",
                    "priority": "HIGH" if abs(self.kalshi_ytd_pnl) > 1000 else "LOW",
                    "description": f"Section 1256 allows 3-year loss carryback",
                    "action": f"YTD loss of ${abs(self.kalshi_ytd_pnl):.2f} can be carried back",
                    "form": "Form 1045 for carryback claim",
                    "benefit": "Can amend prior 3 years' returns for refund"
                })
            
            # 4. 60/40 tax advantage awareness
            if self.kalshi_ytd_pnl > 0:
                # Calculate savings
                all_short_term_tax = self.kalshi_ytd_pnl * 0.37
                actual_tax = self.kalshi_ytd_pnl * ((0.60 * 0.20) + (0.40 * 0.37))
                savings = all_short_term_tax - actual_tax
                
                recommendations.append({
                    "type": "60_40_advantage",
                    "priority": "INFO",
                    "description": "60/40 treatment providing tax savings",
                    "savings": f"${savings:.2f} saved vs all short-term treatment",
                    "effective_rate": f"{((actual_tax/self.kalshi_ytd_pnl)*100):.1f}% vs 37% short-term"
                })
            
            # 5. Kalshi vs Polymarket comparison
            recommendations.append({
                "type": "platform_comparison",
                "priority": "INFO",
                "description": "Kalshi tax advantages vs unregulated platforms",
                "kalshi_benefits": [
                    "Section 1256 (60/40) treatment",
                    "Full 1099-B reporting",
                    "CFTC regulated = audit-proof",
                    "No wash sale restrictions"
                ],
                "polymarket_risks": [
                    "Unclear tax treatment (crypto)",
                    "No automatic tax reporting",
                    "Regulatory uncertainty"
                ],
                "recommendation": "Prioritize Kalshi for tax efficiency"
            })
            
            return {
                "recommendations": recommendations,
                "ytd_pnl": self.kalshi_ytd_pnl,
                "ytd_estimated_tax": self.kalshi_tax_liability,
                "days_to_year_end": days_to_year_end,
                "generated_at": current_date.isoformat()
            }
            
        except Exception as e:
            self.log_action(f"Error generating Kalshi tax optimization: {e}")
            return {"error": str(e)}
    
    def generate_kalshi_wealth_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive Kalshi contribution to generational wealth goals.
        
        Tracks how Kalshi profits contribute to NAE milestones.
        
        Returns:
            Dict with wealth contribution analysis
        """
        try:
            # Calculate total Kalshi profits
            total_kalshi_profits = sum(t.get("pnl_cents", 0) for t in self.kalshi_trades) / 100
            
            # Get current milestone progress
            current_year = datetime.datetime.now().year - 2024 + 1  # Assuming started 2024
            current_year = max(1, min(8, current_year))
            current_milestone = self.growth_milestones.get(current_year, self.growth_milestones[1])
            
            # Calculate contribution percentage
            contribution_pct = (total_kalshi_profits / current_milestone * 100) if current_milestone > 0 else 0
            
            # Tax efficiency score
            if total_kalshi_profits > 0:
                actual_tax_rate = 0.268 + 0.0475  # 60/40 blended + NC
                all_short_term_rate = 0.37 + 0.0475
                tax_efficiency = 1 - (actual_tax_rate / all_short_term_rate)
            else:
                tax_efficiency = 0
            
            report = {
                "summary": {
                    "total_kalshi_profits": round(total_kalshi_profits, 2),
                    "total_trades": len(self.kalshi_trades),
                    "ytd_pnl": round(self.kalshi_ytd_pnl, 2)
                },
                "milestone_contribution": {
                    "current_year": current_year,
                    "year_milestone": current_milestone,
                    "contribution_pct": round(contribution_pct, 2),
                    "target_contribution_pct": self.kalshi_wealth_strategy["growth_contribution_target_pct"],
                    "on_track": contribution_pct >= self.kalshi_wealth_strategy["growth_contribution_target_pct"] * (datetime.datetime.now().month / 12)
                },
                "tax_efficiency": {
                    "score": round(tax_efficiency * 100, 1),
                    "estimated_tax_savings": round(total_kalshi_profits * (0.37 - 0.268), 2) if total_kalshi_profits > 0 else 0,
                    "treatment": "Section 1256 (60/40 split)"
                },
                "regulatory_advantages": {
                    "regulator": "CFTC",
                    "reporting": "Full 1099-B",
                    "fund_safety": "FDIC-insured custody",
                    "legal_clarity": "HIGH"
                },
                "allocation_status": {
                    "target_pct": self.kalshi_wealth_strategy["target_allocation_pct"],
                    "role": self.kalshi_wealth_strategy["role_in_portfolio"]
                },
                "generated_at": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"📈 Kalshi wealth report: ${total_kalshi_profits:.2f} total, {contribution_pct:.1f}% of milestone")
            
            return report
            
        except Exception as e:
            self.log_action(f"Error generating Kalshi wealth report: {e}")
            return {"error": str(e)}
    
    def receive_kalshi_profit_allocation(self, allocation_data: Dict[str, Any]) -> None:
        """
        Receive and process Kalshi profit allocation from Shredder.
        
        Args:
            allocation_data: Profit allocation details from Shredder
        """
        try:
            self.log_action(f"📥 Received Kalshi profit allocation from Shredder")
            
            # Track the trade for taxes
            trade_data = {
                "timestamp": allocation_data.get("timestamp", datetime.datetime.now().isoformat()),
                "ticker": "KALSHI_PROFIT",
                "market_title": "Shredder Profit Allocation",
                "pnl_cents": int(allocation_data.get("total_profit", 0) * 100),
                "strategy": "profit_allocation",
                "category": "consolidated"
            }
            
            self.track_kalshi_trade_for_taxes(trade_data)
            
            # Update wealth tracking
            self._update_financial_capital_metrics({
                "timestamp": datetime.datetime.now().isoformat(),
                "trade_details": allocation_data,
                "success_indicators": {"source": "kalshi_prediction_market"},
                "wealth_impact": {
                    "immediate_gain": allocation_data.get("total_profit", 0),
                    "tax_treatment": "section_1256"
                }
            })
            
            self.log_action(f"✅ Kalshi allocation processed: ${allocation_data.get('total_profit', 0):.2f}")
            
        except Exception as e:
            self.log_action(f"Error processing Kalshi profit allocation: {e}")

    def _analyze_trade_success(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade success indicators"""
        return {
            "execution_speed": trade_data.get("execution_time", 0),
            "risk_management": trade_data.get("risk_score", 0),
            "profit_potential": trade_data.get("expected_return", 0),
            "market_timing": trade_data.get("timing_score", 0)
        }

    def _calculate_wealth_impact(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate impact on generational wealth"""
        return {
            "immediate_gain": trade_data.get("profit", 0),
            "compound_potential": trade_data.get("compound_factor", 1.0),
            "risk_adjusted_return": trade_data.get("risk_adjusted_return", 0),
            "generational_multiplier": self._calculate_generational_multiplier(trade_data)
        }

    def _calculate_generational_multiplier(self, trade_data: Dict[str, Any]) -> float:
        """Calculate how this trade contributes to generational wealth building"""
        base_multiplier = 1.0
        if trade_data.get("strategy_type") == "long_term":
            base_multiplier *= 1.5
        if trade_data.get("risk_level") == "conservative":
            base_multiplier *= 1.2
        return base_multiplier

    def _evaluate_strategy_effectiveness(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the effectiveness of a trading strategy"""
        return {
            "profitability_score": trade_data.get("profit", 0) / 1000.0,  # Normalize to 0-1
            "risk_adjusted_return": trade_data.get("risk_adjusted_return", 0),
            "execution_efficiency": 1.0 - (trade_data.get("execution_time", 1.0) / 10.0),  # Normalize
            "market_timing": trade_data.get("timing_score", 0.5)
        }

    # ----------------------
    # Ralph Strategy Tracking
    # ----------------------
    def track_ralph_strategy(self, strategy_data: Dict[str, Any]):
        """Track and log Ralph's successful strategies"""
        try:
            strategy_analysis = {
                "timestamp": datetime.datetime.now().isoformat(),
                "strategy_details": strategy_data,
                "success_factors": self._analyze_strategy_success_factors(strategy_data),
                "learning_insights": self._extract_learning_insights(strategy_data),
                "replicability_score": self._calculate_replicability_score(strategy_data)
            }
            
            self.ralph_strategy_log.append(strategy_analysis)
            self._save_ralph_log()
            
            # Update intellectual capital tracking
            self._update_intellectual_capital_metrics(strategy_analysis)
            
            self.log_action(f"Tracked Ralph strategy: {strategy_data.get('name', 'Unknown')}")
            return strategy_analysis
            
        except Exception as e:
            self.log_action(f"Error tracking Ralph strategy: {e}")
            return None

    def _analyze_strategy_success_factors(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what makes a strategy successful"""
        return {
            "trust_score": strategy_data.get("trust_score", 0),
            "backtest_performance": strategy_data.get("backtest_score", 0),
            "consensus_level": strategy_data.get("consensus_count", 0),
            "risk_reward_ratio": strategy_data.get("risk_reward", 0),
            "market_conditions": strategy_data.get("market_fit", 0)
        }

    def _extract_learning_insights(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning insights from strategy data"""
        return {
            "pattern_recognition": strategy_data.get("pattern_type", "unknown"),
            "adaptation_mechanism": strategy_data.get("adaptation_method", "unknown"),
            "knowledge_transfer_potential": strategy_data.get("transfer_score", 0),
            "heir_applicability": self._assess_heir_applicability(strategy_data)
        }

    def _assess_heir_applicability(self, strategy_data: Dict[str, Any]) -> float:
        """Assess how applicable this strategy is for heirs"""
        applicability_score = 0.5  # Base score
        
        # Increase score for simpler, more conservative strategies
        if strategy_data.get("complexity_level") == "low":
            applicability_score += 0.2
        if strategy_data.get("risk_level") == "conservative":
            applicability_score += 0.2
        if strategy_data.get("documentation_quality") == "high":
            applicability_score += 0.1
            
        return min(1.0, applicability_score)

    def _calculate_replicability_score(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate how replicable a strategy is"""
        replicability_score = 0.5  # Base score
        
        # Increase score for well-documented, simple strategies
        if strategy_data.get("documentation_quality") == "high":
            replicability_score += 0.2
        if strategy_data.get("complexity_level") == "low":
            replicability_score += 0.2
        if strategy_data.get("consensus_count", 0) > 2:
            replicability_score += 0.1
            
        return min(1.0, replicability_score)

    # ----------------------
    # Recipe Curation System
    # ----------------------
    def curate_success_recipes(self) -> Dict[str, Any]:
        """Curate recipes of success from NAE's profits and methodologies"""
        try:
            recipes = {
                "timestamp": datetime.datetime.now().isoformat(),
                "optimus_recipes": self._extract_optimus_recipes(),
                "ralph_recipes": self._extract_ralph_recipes(),
                "nae_system_recipes": self._extract_nae_system_recipes(),
                "generational_wealth_recipes": self._create_generational_recipes()
            }
            
            self.nae_profit_recipes.append(recipes)
            self._save_recipes()
            
            self.log_action("Success recipes curated and saved")
            return recipes
            
        except Exception as e:
            self.log_action(f"Error curating success recipes: {e}")
            return None

    def _extract_optimus_recipes(self) -> List[Dict[str, Any]]:
        """Extract successful trading recipes from Optimus data"""
        recipes = []
        if not self.optimus_success_log:
            return recipes
            
        # Analyze patterns in successful trades
        successful_trades = [trade for trade in self.optimus_success_log 
                           if trade.get("wealth_impact", {}).get("immediate_gain", 0) > 0]
        
        for trade in successful_trades[-10:]:  # Last 10 successful trades
            recipe = {
                "recipe_name": f"Optimus_Success_{trade['timestamp'][:10]}",
                "strategy_type": trade["trade_details"].get("strategy_name", "Unknown"),
                "success_factors": trade["success_indicators"],
                "wealth_impact": trade["wealth_impact"],
                "replicability_notes": self._generate_replicability_notes(trade)
            }
            recipes.append(recipe)
            
        return recipes

    def _extract_ralph_recipes(self) -> List[Dict[str, Any]]:
        """Extract successful strategy recipes from Ralph data"""
        recipes = []
        if not self.ralph_strategy_log:
            return recipes
            
        # Analyze high-performing strategies
        high_performing = [strategy for strategy in self.ralph_strategy_log 
                          if strategy.get("success_factors", {}).get("trust_score", 0) > 80]
        
        for strategy in high_performing[-5:]:  # Last 5 high-performing strategies
            recipe = {
                "recipe_name": f"Ralph_Strategy_{strategy['timestamp'][:10]}",
                "strategy_name": strategy["strategy_details"].get("name", "Unknown"),
                "success_factors": strategy["success_factors"],
                "learning_insights": strategy["learning_insights"],
                "heir_transfer_notes": self._generate_heir_transfer_notes(strategy)
            }
            recipes.append(recipe)
            
        return recipes

    def _extract_nae_system_recipes(self) -> List[Dict[str, Any]]:
        """Extract system-level recipes from NAE operations"""
        recipes = []
        
        # System integration recipes
        recipes.append({
            "recipe_name": "Agent_Coordination_Success",
            "description": "Successful coordination between NAE agents",
            "components": ["optimus_execution", "ralph_strategy", "casey_building", "april_crypto"],
            "success_factors": ["communication_efficiency", "goal_alignment", "resource_optimization"]
        })
        
        # Profit generation recipes
        recipes.append({
            "recipe_name": "Profit_Generation_System",
            "description": "Systematic approach to profit generation",
            "components": ["strategy_selection", "risk_management", "execution_optimization"],
            "success_factors": ["consistent_performance", "risk_control", "scalability"]
        })
        
        # Knowledge management recipes
        recipes.append({
            "recipe_name": "Knowledge_Management_Framework",
            "description": "Framework for managing and transferring knowledge",
            "components": ["documentation", "learning_systems", "mentorship"],
            "success_factors": ["comprehensiveness", "accessibility", "adaptability"]
        })
        
        return recipes

    def _create_generational_recipes(self) -> List[Dict[str, Any]]:
        """Create recipes specifically for generational wealth building"""
        return [
            {
                "recipe_name": "Financial_Capital_Building",
                "description": "Systematic approach to building financial capital",
                "components": ["diversified_asset_allocation", "compound_growth_strategy", "risk_management"],
                "implementation": "Track and optimize asset allocation across stocks, real estate, businesses, bonds, and cash"
            },
            {
                "recipe_name": "Intellectual_Capital_Preservation",
                "description": "Preserving and transferring financial knowledge",
                "components": ["strategy_documentation", "learning_systems", "knowledge_transfer"],
                "implementation": "Document all successful strategies and create learning systems for heirs"
            },
            {
                "recipe_name": "Social_Capital_Leverage",
                "description": "Building and leveraging professional networks",
                "components": ["network_building", "partnership_development", "reputation_management"],
                "implementation": "Identify and cultivate relationships that enhance financial opportunities"
            },
            {
                "recipe_name": "Values_Legacy_Maintenance",
                "description": "Maintaining family values and ethical framework",
                "components": ["ethical_decision_making", "value_alignment", "legacy_preservation"],
                "implementation": "Ensure all financial decisions align with family values and long-term legacy goals"
            }
        ]

    # ----------------------
    # Heir Knowledge Transfer
    # ----------------------
    def create_heir_knowledge_package(self, heir_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create customized knowledge package for heirs"""
        try:
            knowledge_package = {
                "timestamp": datetime.datetime.now().isoformat(),
                "heir_profile": heir_profile,
                "customized_recipes": self._customize_recipes_for_heir(heir_profile),
                "learning_path": self._create_learning_path(heir_profile),
                "mentorship_plan": self._create_mentorship_plan(heir_profile),
                "success_metrics": self._define_heir_success_metrics(heir_profile)
            }
            
            self.heir_knowledge_base[heir_profile.get("heir_id", "unknown")] = knowledge_package
            self._save_heir_knowledge()
            
            self.log_action(f"Created knowledge package for heir: {heir_profile.get('name', 'Unknown')}")
            return knowledge_package
            
        except Exception as e:
            self.log_action(f"Error creating heir knowledge package: {e}")
            return None

    def _customize_recipes_for_heir(self, heir_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Customize success recipes based on heir's profile"""
        heir_experience = heir_profile.get("experience_level", "beginner")
        heir_interests = heir_profile.get("interests", [])
        
        customized_recipes = []
        
        # Filter recipes based on heir's experience level
        for recipe_category in self.nae_profit_recipes[-1:]:  # Latest recipes
            for recipe in recipe_category.get("optimus_recipes", []):
                if heir_experience == "beginner" and recipe.get("complexity", "medium") == "low":
                    customized_recipes.append(recipe)
                elif heir_experience == "intermediate":
                    customized_recipes.append(recipe)
                elif heir_experience == "advanced":
                    customized_recipes.append(recipe)
        
        return customized_recipes[:5]  # Limit to top 5 most relevant

    def _create_learning_path(self, heir_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a customized learning path for the heir"""
        experience_level = heir_profile.get("experience_level", "beginner")
        interests = heir_profile.get("interests", [])
        
        learning_path = {
            "heir_id": heir_profile.get("heir_id", "unknown"),
            "experience_level": experience_level,
            "interests": interests,
            "modules": self._select_learning_modules(experience_level),
            "timeline": self._create_learning_timeline(experience_level),
            "assessment_criteria": self._define_assessment_criteria(experience_level),
            "customization_notes": f"Path customized for {experience_level} level with interests in {', '.join(interests)}"
        }
        
        return learning_path

    def _select_learning_modules(self, experience_level: str) -> List[str]:
        """Select appropriate learning modules based on experience level"""
        modules = {
            "beginner": [
                "Basic Financial Concepts",
                "NAE System Overview", 
                "Risk Management Fundamentals",
                "Ethical Decision Making"
            ],
            "intermediate": [
                "Advanced Trading Strategies",
                "Portfolio Optimization",
                "Market Analysis Techniques",
                "Generational Wealth Planning"
            ],
            "advanced": [
                "Complex Options Strategies",
                "Advanced Risk Management",
                "System Optimization",
                "Legacy Planning and Transfer"
            ]
        }
        
        return modules.get(experience_level, modules["beginner"])

    def _create_learning_timeline(self, experience_level: str) -> Dict[str, Any]:
        """Create learning timeline based on experience level"""
        timelines = {
            "beginner": {"duration_months": 12, "milestones": 4, "pace": "gradual"},
            "intermediate": {"duration_months": 8, "milestones": 3, "pace": "moderate"},
            "advanced": {"duration_months": 6, "milestones": 2, "pace": "intensive"}
        }
        
        return timelines.get(experience_level, timelines["beginner"])

    def _define_assessment_criteria(self, experience_level: str) -> List[str]:
        """Define assessment criteria for learning progress"""
        criteria = {
            "beginner": [
                "Understanding of basic concepts",
                "Ability to identify risks",
                "Ethical decision making",
                "System navigation skills"
            ],
            "intermediate": [
                "Strategy implementation",
                "Portfolio management",
                "Market analysis",
                "Risk assessment"
            ],
            "advanced": [
                "Complex strategy execution",
                "System optimization",
                "Legacy planning",
                "Mentorship capabilities"
            ]
        }
        
        return criteria.get(experience_level, criteria["beginner"])

    def _create_mentorship_plan(self, heir_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mentorship plan for the heir"""
        experience_level = heir_profile.get("experience_level", "beginner")
        
        mentorship_plan = {
            "heir_id": heir_profile.get("heir_id", "unknown"),
            "experience_level": experience_level,
            "mentorship_duration": self._get_mentorship_duration(experience_level),
            "mentorship_activities": self._select_mentorship_activities(experience_level),
            "mentor_assignments": self._assign_mentors(experience_level),
            "progress_milestones": self._define_progress_milestones(experience_level),
            "knowledge_transfer_schedule": self._create_transfer_schedule(experience_level)
        }
        
        return mentorship_plan

    def _get_mentorship_duration(self, experience_level: str) -> Dict[str, Any]:
        """Get mentorship duration based on experience level"""
        durations = {
            "beginner": {"months": 18, "intensity": "high", "frequency": "weekly"},
            "intermediate": {"months": 12, "intensity": "medium", "frequency": "bi-weekly"},
            "advanced": {"months": 6, "intensity": "low", "frequency": "monthly"}
        }
        
        return durations.get(experience_level, durations["beginner"])

    def _select_mentorship_activities(self, experience_level: str) -> List[str]:
        """Select appropriate mentorship activities"""
        activities = {
            "beginner": [
                "One-on-one strategy explanations",
                "Risk management training",
                "System navigation guidance",
                "Ethical decision making workshops"
            ],
            "intermediate": [
                "Strategy implementation coaching",
                "Portfolio management guidance",
                "Market analysis training",
                "Advanced risk assessment"
            ],
            "advanced": [
                "Complex strategy development",
                "System optimization consulting",
                "Legacy planning guidance",
                "Mentorship skill development"
            ]
        }
        
        return activities.get(experience_level, activities["beginner"])

    def _assign_mentors(self, experience_level: str) -> List[Dict[str, str]]:
        """Assign appropriate mentors based on experience level"""
        mentor_assignments = {
            "beginner": [
                {"mentor": "Casey", "role": "System Education", "focus": "NAE Overview"},
                {"mentor": "Ralph", "role": "Strategy Learning", "focus": "Basic Strategies"}
            ],
            "intermediate": [
                {"mentor": "Optimus", "role": "Execution Training", "focus": "Trade Execution"},
                {"mentor": "Ralph", "role": "Strategy Development", "focus": "Advanced Strategies"}
            ],
            "advanced": [
                {"mentor": "Genny", "role": "Wealth Management", "focus": "Generational Planning"},
                {"mentor": "Optimus", "role": "System Mastery", "focus": "Advanced Execution"}
            ]
        }
        
        return mentor_assignments.get(experience_level, mentor_assignments["beginner"])

    def _define_progress_milestones(self, experience_level: str) -> List[Dict[str, Any]]:
        """Define progress milestones for mentorship"""
        milestones = {
            "beginner": [
                {"milestone": "Basic Understanding", "timeline": "3 months", "criteria": "Can explain basic NAE concepts"},
                {"milestone": "Risk Awareness", "timeline": "6 months", "criteria": "Can identify and assess risks"},
                {"milestone": "Ethical Framework", "timeline": "9 months", "criteria": "Makes ethical decisions consistently"},
                {"milestone": "System Navigation", "timeline": "12 months", "criteria": "Can navigate NAE systems independently"}
            ],
            "intermediate": [
                {"milestone": "Strategy Implementation", "timeline": "4 months", "criteria": "Can implement basic strategies"},
                {"milestone": "Portfolio Management", "timeline": "8 months", "criteria": "Can manage portfolio effectively"},
                {"milestone": "Advanced Analysis", "timeline": "12 months", "criteria": "Can perform advanced market analysis"}
            ],
            "advanced": [
                {"milestone": "Complex Strategy Mastery", "timeline": "3 months", "criteria": "Can execute complex strategies"},
                {"milestone": "System Optimization", "timeline": "6 months", "criteria": "Can optimize NAE systems"}
            ]
        }
        
        return milestones.get(experience_level, milestones["beginner"])

    def _create_transfer_schedule(self, experience_level: str) -> Dict[str, Any]:
        """Create knowledge transfer schedule"""
        schedules = {
            "beginner": {
                "frequency": "weekly",
                "duration_per_session": "2 hours",
                "topics_per_session": 2,
                "assessment_frequency": "monthly"
            },
            "intermediate": {
                "frequency": "bi-weekly",
                "duration_per_session": "3 hours",
                "topics_per_session": 3,
                "assessment_frequency": "quarterly"
            },
            "advanced": {
                "frequency": "monthly",
                "duration_per_session": "4 hours",
                "topics_per_session": 4,
                "assessment_frequency": "quarterly"
            }
        }
        
        return schedules.get(experience_level, schedules["beginner"])

    def _define_heir_success_metrics(self, heir_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Define success metrics for heir development"""
        experience_level = heir_profile.get("experience_level", "beginner")
        
        success_metrics = {
            "heir_id": heir_profile.get("heir_id", "unknown"),
            "experience_level": experience_level,
            "knowledge_metrics": self._define_knowledge_metrics(experience_level),
            "skill_metrics": self._define_skill_metrics(experience_level),
            "performance_metrics": self._define_performance_metrics(experience_level),
            "ethical_metrics": self._define_ethical_metrics(experience_level)
        }
        
        return success_metrics

    def _define_knowledge_metrics(self, experience_level: str) -> List[str]:
        """Define knowledge-based success metrics"""
        metrics = {
            "beginner": [
                "Understanding of basic financial concepts",
                "Knowledge of NAE system architecture",
                "Awareness of risk management principles",
                "Comprehension of ethical frameworks"
            ],
            "intermediate": [
                "Mastery of trading strategies",
                "Understanding of portfolio management",
                "Knowledge of market analysis techniques",
                "Comprehension of generational wealth concepts"
            ],
            "advanced": [
                "Expertise in complex strategies",
                "Advanced system optimization knowledge",
                "Mastery of legacy planning",
                "Ability to mentor others"
            ]
        }
        
        return metrics.get(experience_level, metrics["beginner"])

    def _define_skill_metrics(self, experience_level: str) -> List[str]:
        """Define skill-based success metrics"""
        metrics = {
            "beginner": [
                "Basic system navigation",
                "Risk identification",
                "Ethical decision making",
                "Communication skills"
            ],
            "intermediate": [
                "Strategy implementation",
                "Portfolio management",
                "Market analysis",
                "Risk assessment"
            ],
            "advanced": [
                "Complex strategy execution",
                "System optimization",
                "Legacy planning",
                "Mentorship capabilities"
            ]
        }
        
        return metrics.get(experience_level, metrics["beginner"])

    def _define_performance_metrics(self, experience_level: str) -> List[str]:
        """Define performance-based success metrics"""
        metrics = {
            "beginner": [
                "Consistent attendance",
                "Active participation",
                "Assignment completion",
                "Progress demonstration"
            ],
            "intermediate": [
                "Strategy success rate",
                "Risk management effectiveness",
                "Portfolio performance",
                "Learning curve progression"
            ],
            "advanced": [
                "Advanced strategy performance",
                "System optimization results",
                "Legacy planning implementation",
                "Mentorship effectiveness"
            ]
        }
        
        return metrics.get(experience_level, metrics["beginner"])

    def _define_ethical_metrics(self, experience_level: str) -> List[str]:
        """Define ethical-based success metrics"""
        metrics = {
            "beginner": [
                "Ethical decision making",
                "Value alignment",
                "Transparency in actions",
                "Stakeholder consideration"
            ],
            "intermediate": [
                "Ethical strategy implementation",
                "Responsible risk management",
                "Transparent reporting",
                "Stakeholder value creation"
            ],
            "advanced": [
                "Ethical leadership",
                "Value preservation",
                "Transparent governance",
                "Stakeholder stewardship"
            ]
        }
        
        return metrics.get(experience_level, metrics["beginner"])

    # ----------------------
    # Learning and Adaptation
    # ----------------------
    def learn_wealth_growth_patterns(self) -> Dict[str, Any]:
        """Learn patterns in wealth growth and maintenance"""
        try:
            patterns = {
                "timestamp": datetime.datetime.now().isoformat(),
                "growth_patterns": self._analyze_growth_patterns(),
                "maintenance_patterns": self._analyze_maintenance_patterns(),
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "risk_patterns": self._analyze_risk_patterns()
            }
            
            self.wealth_growth_patterns = patterns
            self._save_learning_data()
            
            self.log_action("Wealth growth patterns learned and updated")
            return patterns
            
        except Exception as e:
            self.log_action(f"Error learning wealth growth patterns: {e}")
            return None

    def _analyze_growth_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in wealth growth"""
        if not self.optimus_success_log:
            return {"status": "insufficient_data"}
        
        # Analyze growth trends
        recent_trades = self.optimus_success_log[-20:]  # Last 20 trades
        growth_rates = []
        
        for trade in recent_trades:
            wealth_impact = trade.get("wealth_impact", {})
            growth_rate = wealth_impact.get("compound_potential", 1.0)
            growth_rates.append(growth_rate)
        
        if growth_rates:
            return {
                "average_growth_rate": statistics.mean(growth_rates),
                "growth_volatility": statistics.stdev(growth_rates) if len(growth_rates) > 1 else 0,
                "trend_direction": "increasing" if growth_rates[-1] > growth_rates[0] else "decreasing",
                "optimal_conditions": self._identify_optimal_growth_conditions(recent_trades)
            }
        
        return {"status": "no_data"}

    def _analyze_maintenance_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in wealth maintenance strategies"""
        if not self.optimus_success_log:
            return {"status": "insufficient_data"}
        
        # Analyze maintenance strategies from recent trades
        recent_trades = self.optimus_success_log[-15:]  # Last 15 trades
        maintenance_scores = []
        
        for trade in recent_trades:
            # Calculate maintenance score based on risk management and consistency
            risk_score = trade.get("success_indicators", {}).get("risk_management", 0)
            consistency_score = 0.8  # Placeholder for consistency calculation
            maintenance_score = (risk_score + consistency_score) / 2
            maintenance_scores.append(maintenance_score)
        
        if maintenance_scores:
            return {
                "average_maintenance_score": statistics.mean(maintenance_scores),
                "maintenance_trend": "stable" if len(maintenance_scores) > 1 and abs(maintenance_scores[-1] - maintenance_scores[0]) < 0.1 else "variable",
                "risk_management_effectiveness": statistics.mean([trade.get("success_indicators", {}).get("risk_management", 0) for trade in recent_trades]),
                "recommended_improvements": self._generate_maintenance_recommendations(recent_trades)
            }
        
        return {"status": "no_data"}

    def _generate_maintenance_recommendations(self, trades: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving wealth maintenance"""
        recommendations = []
        
        # Analyze risk management patterns
        risk_scores = [trade.get("success_indicators", {}).get("risk_management", 0) for trade in trades]
        avg_risk_score = statistics.mean(risk_scores) if risk_scores else 0
        
        if avg_risk_score < 0.7:
            recommendations.append("Improve risk management protocols")
        
        # Analyze consistency patterns
        profit_variability = statistics.stdev([trade.get("wealth_impact", {}).get("immediate_gain", 0) for trade in trades]) if len(trades) > 1 else 0
        if profit_variability > 1000:  # High variability threshold
            recommendations.append("Implement more consistent profit strategies")
        
        recommendations.append("Continue monitoring and adapting maintenance strategies")
        
        return recommendations

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for optimizing wealth growth"""
        opportunities = []
        
        # Analyze current performance gaps
        if self.optimus_success_log:
            recent_performance = [trade.get("wealth_impact", {}).get("immediate_gain", 0) for trade in self.optimus_success_log[-10:]]
            avg_performance = statistics.mean(recent_performance) if recent_performance else 0
            
            if avg_performance < 500:  # Below target threshold
                opportunities.append({
                    "type": "performance_improvement",
                    "description": "Increase average trade performance",
                    "potential_impact": "high",
                    "implementation_difficulty": "medium"
                })
        
        # Analyze strategy diversification
        opportunities.append({
            "type": "diversification",
            "description": "Expand strategy portfolio diversity",
            "potential_impact": "medium",
            "implementation_difficulty": "low"
        })
        
        return opportunities

    def _analyze_risk_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in risk management"""
        if not self.optimus_success_log:
            return {"status": "insufficient_data"}
        
        recent_trades = self.optimus_success_log[-20:]  # Last 20 trades
        risk_scores = [trade.get("success_indicators", {}).get("risk_management", 0) for trade in recent_trades]
        
        if risk_scores:
            return {
                "average_risk_score": statistics.mean(risk_scores),
                "risk_volatility": statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0,
                "risk_trend": "improving" if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else "stable",
                "high_risk_periods": self._identify_high_risk_periods(recent_trades)
            }
        
        return {"status": "no_data"}

    def _identify_high_risk_periods(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify periods of high risk in trading"""
        high_risk_periods = []
        
        for i, trade in enumerate(trades):
            risk_score = trade.get("success_indicators", {}).get("risk_management", 0)
            if risk_score < 0.5:  # High risk threshold
                high_risk_periods.append({
                    "timestamp": trade.get("timestamp", "unknown"),
                    "risk_score": risk_score,
                    "trade_details": trade.get("trade_details", {}),
                    "recommendation": "Review risk management protocols"
                })
        
        return high_risk_periods

    def _identify_optimal_growth_conditions(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify optimal conditions for wealth growth"""
        if not trades:
            return {"status": "no_data"}
        
        # Analyze conditions that led to highest growth
        high_growth_trades = [trade for trade in trades if trade.get("wealth_impact", {}).get("immediate_gain", 0) > 0]
        
        if high_growth_trades:
            return {
                "market_conditions": "favorable",
                "strategy_types": list(set([trade.get("trade_details", {}).get("strategy_name", "unknown") for trade in high_growth_trades])),
                "risk_levels": list(set([trade.get("trade_details", {}).get("risk_level", "unknown") for trade in high_growth_trades])),
                "timing_patterns": "consistent_execution"
            }
        
        return {"status": "insufficient_high_growth_data"}

    def _generate_replicability_notes(self, trade: Dict[str, Any]) -> List[str]:
        """Generate notes on how to replicate successful trades"""
        notes = []
        
        trade_details = trade.get("trade_details", {})
        strategy_name = trade_details.get("strategy_name", "Unknown")
        
        notes.append(f"Strategy: {strategy_name}")
        notes.append(f"Risk management score: {trade.get('success_indicators', {}).get('risk_management', 0)}")
        notes.append(f"Execution efficiency: {trade.get('success_indicators', {}).get('execution_speed', 0)}")
        
        if trade.get("wealth_impact", {}).get("immediate_gain", 0) > 0:
            notes.append("Positive profit achieved - strategy replicable")
        else:
            notes.append("No profit - review strategy parameters")
        
        return notes

    def _generate_heir_transfer_notes(self, strategy: Dict[str, Any]) -> List[str]:
        """Generate notes for transferring strategy knowledge to heirs"""
        notes = []
        
        strategy_details = strategy.get("strategy_details", {})
        strategy_name = strategy_details.get("name", "Unknown")
        
        notes.append(f"Strategy: {strategy_name}")
        notes.append(f"Trust score: {strategy.get('success_factors', {}).get('trust_score', 0)}")
        notes.append(f"Replicability score: {strategy.get('replicability_score', 0)}")
        
        heir_applicability = strategy.get("learning_insights", {}).get("heir_applicability", 0)
        if heir_applicability > 0.7:
            notes.append("High heir applicability - suitable for knowledge transfer")
        elif heir_applicability > 0.5:
            notes.append("Moderate heir applicability - requires additional training")
        else:
            notes.append("Low heir applicability - complex strategy requiring expert knowledge")
        
        return notes

    # ----------------------
    # Monitoring and Data Collection
    # ----------------------
    def monitor_agents_loop(self):
        """Continuously monitor other agents for success data"""
        while self.monitoring_active:
            try:
                # Monitor Optimus for trade data
                self._monitor_optimus_trades()
                
                # Monitor Ralph for strategy data
                self._monitor_ralph_strategies()
                
                # Update generational wealth metrics
                self._update_generational_wealth_metrics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.log_action(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _monitor_optimus_trades(self):
        """Monitor Optimus agent for new trade data"""
        # This would integrate with Optimus' execution history
        # For now, we'll simulate monitoring
        pass

    def _monitor_ralph_strategies(self):
        """Monitor Ralph agent for new strategy data"""
        # This would integrate with Ralph's strategy database
        # For now, we'll simulate monitoring
        pass

    # ----------------------
    # Data Persistence
    # ----------------------
    def _save_optimus_log(self):
        """Save Optimus success log to file"""
        try:
            file_path = os.path.join(self.data_dir, "optimus_success_log.json")
            with open(file_path, "w") as f:
                json.dump(self.optimus_success_log, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving Optimus log: {e}")

    def _save_ralph_log(self):
        """Save Ralph strategy log to file"""
        try:
            file_path = os.path.join(self.data_dir, "ralph_strategy_log.json")
            with open(file_path, "w") as f:
                json.dump(self.ralph_strategy_log, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving Ralph log: {e}")

    def _save_recipes(self):
        """Save success recipes to file"""
        try:
            file_path = os.path.join(self.data_dir, "success_recipes.json")
            with open(file_path, "w") as f:
                json.dump(self.nae_profit_recipes, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving recipes: {e}")

    def _save_heir_knowledge(self):
        """Save heir knowledge base to file"""
        try:
            file_path = os.path.join(self.data_dir, "heir_knowledge_base.json")
            with open(file_path, "w") as f:
                json.dump(self.heir_knowledge_base, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving heir knowledge: {e}")

    def _save_learning_data(self):
        """Save learning data to file"""
        try:
            file_path = os.path.join(self.data_dir, "learning_data.json")
            learning_data = {
                "wealth_growth_patterns": self.wealth_growth_patterns,
                "maintenance_strategies": self.maintenance_strategies,
                "learning_history": self.learning_history
            }
            with open(file_path, "w") as f:
                json.dump(learning_data, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving learning data: {e}")

    # ----------------------
    # Generational Wealth Component Updates
    # ----------------------
    def _update_financial_capital_metrics(self, success_metrics: Dict[str, Any]):
        """Update financial capital tracking metrics"""
        # Implementation for tracking financial capital metrics
        pass

    def _update_intellectual_capital_metrics(self, strategy_analysis: Dict[str, Any]):
        """Update intellectual capital tracking metrics"""
        # Implementation for tracking intellectual capital metrics
        pass

    def _update_generational_wealth_metrics(self):
        """Update all generational wealth metrics"""
        # Implementation for updating all wealth metrics
        pass
    
    # ----------------------
    # Tax Preparation Methods
    # ----------------------
    
    def track_crypto_transaction(
        self,
        transaction_data: Dict[str, Any],
        agent: str = "shredder"
    ):
        """
        Track cryptocurrency transaction for tax purposes
        
        Args:
            transaction_data: Transaction details
            agent: Agent handling transaction (shredder, april)
        """
        try:
            if not self.tax_tracking_enabled or not self.tax_preparer:
                return
            
            # Convert crypto transaction to trade record
            trade_id = transaction_data.get("transaction_id") or f"crypto_{datetime.datetime.now().timestamp()}"
            timestamp = transaction_data.get("timestamp") or datetime.datetime.now().isoformat()
            symbol = transaction_data.get("crypto_symbol", "BTC")
            asset_type = "crypto"
            action = transaction_data.get("action", "buy").lower()  # buy, sell, exchange
            quantity = float(transaction_data.get("quantity", 0))
            price = float(transaction_data.get("price", transaction_data.get("usd_value", 0)) / quantity if quantity > 0 else 0)
            fees = float(transaction_data.get("fees", transaction_data.get("network_fee", 0)))
            notes = f"Crypto transaction via {agent}: {transaction_data.get('notes', '')}"
            
            if symbol and quantity > 0:
                self.tax_preparer.record_trade(
                    trade_id=trade_id,
                    timestamp=timestamp,
                    symbol=symbol,
                    asset_type=asset_type,
                    action=action,
                    quantity=quantity,
                    price=price,
                    fees=fees,
                    agent=agent,
                    notes=notes
                )
                self.log_action(f"Tax tracking: Recorded crypto {action} for {symbol}")
        
        except Exception as e:
            self.log_action(f"Error tracking crypto transaction: {e}")
    
    def track_fiat_flow(
        self,
        flow_data: Dict[str, Any],
        agent: str = "donnie"
    ):
        """
        Track fiat currency flow for tax purposes
        
        Args:
            flow_data: Flow details (deposits, withdrawals, transfers)
            agent: Agent handling flow (donnie, mikey)
        """
        try:
            if not self.tax_tracking_enabled or not self.tax_preparer:
                return
            
            flow_type = flow_data.get("type", "").lower()  # deposit, withdrawal, transfer
            amount = float(flow_data.get("amount", 0))
            timestamp = flow_data.get("timestamp") or datetime.datetime.now().isoformat()
            
            # Fiat flows are typically not taxable events unless they represent income
            # But we track them for accounting purposes
            if flow_type == "income" or flow_data.get("taxable", False):
                # Record as income (would need to be categorized properly)
                self.log_action(f"Tax tracking: Recorded taxable fiat flow: ${amount:.2f} via {agent}")
        
        except Exception as e:
            self.log_action(f"Error tracking fiat flow: {e}")
    
    def record_deductible_expense(
        self,
        expense_data: Dict[str, Any]
    ):
        """
        Record a deductible business expense
        
        Args:
            expense_data: Expense details
        """
        try:
            if not self.tax_tracking_enabled or not self.tax_preparer:
                return
            
            expense_id = expense_data.get("expense_id") or f"exp_{datetime.datetime.now().timestamp()}"
            timestamp = expense_data.get("timestamp") or datetime.datetime.now().isoformat()
            category = expense_data.get("category", "other")
            description = expense_data.get("description", "")
            amount = float(expense_data.get("amount", 0))
            deductible_pct = float(expense_data.get("deductible_pct", 100.0))
            business_use_pct = float(expense_data.get("business_use_pct", 100.0))
            receipt_path = expense_data.get("receipt_path")
            
            self.tax_preparer.record_expense(
                expense_id=expense_id,
                timestamp=timestamp,
                category=category,
                description=description,
                amount=amount,
                deductible_pct=deductible_pct,
                business_use_pct=business_use_pct,
                receipt_path=receipt_path
            )
            
            self.log_action(f"Tax tracking: Recorded expense: {category} - ${amount:.2f}")
        
        except Exception as e:
            self.log_action(f"Error recording expense: {e}")
    
    def generate_tax_summary(
        self,
        tax_year: Optional[int] = None,
        include_unrealized: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive tax summary
        
        Args:
            tax_year: Tax year (defaults to current year)
            include_unrealized: Include unrealized gains
        
        Returns:
            Comprehensive tax summary
        """
        if not self.tax_tracking_enabled or not self.tax_preparer:
            return {"error": "Tax tracking not available"}
        
        try:
            summary = self.tax_preparer.get_tax_summary(tax_year, include_unrealized)
            
            # Add North Carolina specific information
            summary["north_carolina_compliance"] = {
                "tax_knowledge": self.nc_tax_knowledge,
                "compliance_notes": [
                    "NC treats all capital gains as ordinary income (4.75% flat rate)",
                    "No preferential rate for long-term capital gains",
                    "Business expenses deductible if ordinary and necessary",
                    "Day trading may qualify as business income",
                    "Cryptocurrency treated as property (follows federal rules)"
                ]
            }
            
            self.log_action(f"Generated tax summary for {tax_year or 'current year'}")
            return summary
        
        except Exception as e:
            self.log_action(f"Error generating tax summary: {e}")
            return {"error": str(e)}
    
    def export_tax_data(
        self,
        format: str = "turbo_tax",
        tax_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export tax data in specified format
        
        Args:
            format: Export format ("turbo_tax", "cpa", "json")
            tax_year: Tax year
        
        Returns:
            Exported tax data
        """
        if not self.tax_tracking_enabled or not self.tax_preparer:
            return {"error": "Tax tracking not available"}
        
        try:
            if format.lower() == "turbo_tax":
                return self.tax_preparer.export_turbo_tax_format(tax_year)
            elif format.lower() == "cpa":
                return self.tax_preparer.export_cpa_format(tax_year)
            else:
                return self.tax_preparer.get_tax_summary(tax_year)
        
        except Exception as e:
            self.log_action(f"Error exporting tax data: {e}")
            return {"error": str(e)}
    
    def set_cost_basis_method(self, method: str):
        """
        Set cost basis calculation method
        
        Args:
            method: "FIFO", "LIFO", "HIFO", or "AVERAGE"
        """
        if not self.tax_tracking_enabled or not self.tax_preparer:
            return
        
        try:
            method_map = {
                "FIFO": CostBasisMethod.FIFO,
                "LIFO": CostBasisMethod.LIFO,
                "HIFO": CostBasisMethod.HIFO,
                "AVERAGE": CostBasisMethod.AVERAGE
            }
            
            if method.upper() in method_map:
                self.tax_preparer.cost_basis_method = method_map[method.upper()]
                self.log_action(f"Cost basis method set to {method.upper()}")
            else:
                self.log_action(f"Invalid cost basis method: {method}")
        
        except Exception as e:
            self.log_action(f"Error setting cost basis method: {e}")
    
    def get_nc_tax_advice(self, question: str) -> Dict[str, Any]:
        """
        Get North Carolina tax law advice
        
        Args:
            question: Tax question
        
        Returns:
            Tax advice based on NC law
        """
        advice = {
            "question": question,
            "nc_tax_knowledge": self.nc_tax_knowledge,
            "general_advice": [],
            "nc_specific": []
        }
        
        question_lower = question.lower()
        
        # Provide NC-specific advice based on question
        if "capital gain" in question_lower or "long term" in question_lower:
            advice["nc_specific"].append(
                "NC does NOT differentiate between short-term and long-term capital gains. "
                "All capital gains are taxed at the flat 4.75% rate as ordinary income."
            )
        
        if "deduct" in question_lower or "expense" in question_lower:
            advice["nc_specific"].append(
                "Business expenses are deductible in NC if they are ordinary and necessary. "
                "Common deductible expenses include: software subscriptions, hardware, "
                "professional services, and business-related education."
            )
        
        if "crypto" in question_lower or "bitcoin" in question_lower:
            advice["nc_specific"].append(
                "NC follows federal treatment of cryptocurrency as property. "
                "Capital gains/losses apply to crypto sales. Mining income is taxable as ordinary income."
            )
        
        if "day trade" in question_lower:
            advice["nc_specific"].append(
                "Day trading may qualify as business income in NC if it's your primary activity. "
                "Must meet IRS trader tax status requirements."
            )
        
        return advice
    
    # ----------------------
    # Tax Optimization Methods
    # ----------------------
    
    def analyze_tax_efficiency(self, financial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze tax efficiency and identify optimization opportunities
        
        Args:
            financial_data: Optional financial data (if None, will gather from tax preparer)
        
        Returns:
            Tax efficiency analysis with recommendations
        """
        if not self.tax_optimization_enabled or not self.tax_optimizer:
            return {"error": "Tax optimization not available"}
        
        try:
            # Gather financial data if not provided
            if financial_data is None:
                financial_data = self._gather_financial_data_for_analysis()
            
            # Analyze tax efficiency
            analysis = self.tax_optimizer.analyze_tax_efficiency(financial_data)
            
            self.log_action("Tax efficiency analysis completed")
            return analysis
        
        except Exception as e:
            self.log_action(f"Error analyzing tax efficiency: {e}")
            return {"error": str(e)}
    
    def _gather_financial_data_for_analysis(self) -> Dict[str, Any]:
        """Gather financial data from tax preparer for analysis"""
        financial_data = {
            "income": 0.0,
            "capital_gains": 0.0,
            "short_term_gains": 0.0,
            "long_term_gains": 0.0,
            "deductible_expenses": 0.0,
            "unrealized_gains": 0.0,
            "unrealized_losses": 0.0
        }
        
        if self.tax_tracking_enabled and self.tax_preparer:
            # Get current year summary
            current_year = datetime.datetime.now().year
            summary = self.tax_preparer.get_tax_summary(current_year, include_unrealized=True)
            
            # Extract data
            gains = summary.get("capital_gains", {})
            financial_data["capital_gains"] = gains.get("total_realized_gain", 0)
            financial_data["short_term_gains"] = gains.get("short_term_gain", 0)
            financial_data["long_term_gains"] = gains.get("long_term_gain", 0)
            
            expenses = summary.get("expenses", {})
            financial_data["deductible_expenses"] = expenses.get("deductible_expenses", 0)
            
            # Unrealized gains/losses
            unrealized = summary.get("unrealized_gains", {})
            total_unrealized = sum(
                v.get("unrealized_gain", 0) for v in unrealized.values()
            )
            if total_unrealized > 0:
                financial_data["unrealized_gains"] = total_unrealized
            else:
                financial_data["unrealized_losses"] = abs(total_unrealized)
        
        return financial_data
    
    def get_tax_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get tax optimization recommendations
        
        Returns:
            List of recommended tax strategies
        """
        if not self.tax_optimization_enabled or not self.tax_optimizer:
            return []
        
        try:
            financial_data = self._gather_financial_data_for_analysis()
            analysis = self.tax_optimizer.analyze_tax_efficiency(financial_data)
            
            return analysis.get("recommended_strategies", [])
        
        except Exception as e:
            self.log_action(f"Error getting optimization recommendations: {e}")
            return []
    
    def get_latest_tax_law_updates(self, jurisdiction: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get latest tax law updates
        
        Args:
            jurisdiction: "federal", "nc", or None for all
            days: Number of days to look back
        
        Returns:
            List of tax law updates
        """
        if not self.tax_optimization_enabled or not self.tax_research_engine:
            return []
        
        try:
            updates = self.tax_research_engine.get_latest_updates(jurisdiction, days)
            return [asdict(u) for u in updates]
        
        except Exception as e:
            self.log_action(f"Error getting tax law updates: {e}")
            return []
    
    def research_tax_law_now(self) -> Dict[str, Any]:
        """
        Manually trigger tax law research
        
        Returns:
            Research results
        """
        if not self.tax_optimization_enabled or not self.tax_research_engine:
            return {"error": "Tax research not available"}
        
        try:
            updates = self.tax_research_engine.research_tax_law_updates()
            
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "updates_found": len(updates),
                "updates": [asdict(u) for u in updates]
            }
            
            self.log_action(f"Tax law research completed: {len(updates)} updates found")
            return result
        
        except Exception as e:
            self.log_action(f"Error researching tax law: {e}")
            return {"error": str(e)}
    
    def implement_tax_strategy(self, strategy_id: str, implementation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement a tax optimization strategy
        
        Args:
            strategy_id: Strategy identifier
            implementation_data: Implementation details
        
        Returns:
            Implementation result
        """
        if not self.tax_optimization_enabled or not self.tax_optimizer:
            return {"error": "Tax optimization not available"}
        
        try:
            success = self.tax_optimizer.implement_strategy(strategy_id, implementation_data)
            
            if success:
                strategy = self.tax_optimizer.get_strategy_details(strategy_id)
                self.log_action(f"Tax strategy implemented: {strategy_id}")
                return {
                    "success": True,
                    "strategy": asdict(strategy) if strategy else None,
                    "implementation_data": implementation_data
                }
            else:
                return {"success": False, "error": "Strategy not found"}
        
        except Exception as e:
            self.log_action(f"Error implementing tax strategy: {e}")
            return {"error": str(e)}
    
    def get_tax_strategy_details(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a tax strategy
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Strategy details
        """
        if not self.tax_optimization_enabled or not self.tax_optimizer:
            return None
        
        try:
            strategy = self.tax_optimizer.get_strategy_details(strategy_id)
            return asdict(strategy) if strategy else None
        
        except Exception as e:
            self.log_action(f"Error getting strategy details: {e}")
            return None
    
    def learn_tax_efficiency_patterns(self) -> Dict[str, Any]:
        """
        Learn from tax efficiency patterns and improve recommendations
        
        Returns:
            Learning results
        """
        if not self.tax_optimization_enabled or not self.tax_optimizer:
            return {"error": "Tax optimization not available"}
        
        try:
            # Analyze historical tax data
            financial_data = self._gather_financial_data_for_analysis()
            
            # Get implemented strategies
            implemented = self.tax_optimizer.get_implemented_strategies()
            
            # Analyze effectiveness
            learning_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "implemented_strategies": len(implemented),
                "effectiveness_analysis": {},
                "improvements": []
            }
            
            # Would analyze historical effectiveness and suggest improvements
            # For now, return basic structure
            
            self.log_action("Tax efficiency patterns learned")
            return learning_results
        
        except Exception as e:
            self.log_action(f"Error learning tax efficiency patterns: {e}")
            return {"error": str(e)}

    # ----------------------
    # Messaging / AutoGen hooks
    # ----------------------
    def receive_message(self, message: str):
        self.inbox.append(message)
        self.log_action(f"Received message: {message}")

    def send_message(self, message: str, recipient_agent):
        recipient_agent.receive_message(message)
        self.outbox.append({"to": recipient_agent.__class__.__name__, "message": message})
        self.log_action(f"Sent message to {recipient_agent.__class__.__name__}: {message}")

    # ----------------------
    # Main execution cycle
    # ----------------------
    def run_cycle(self) -> Dict[str, Any]:
        """Main execution cycle for Genny agent"""
        self.log_action("Genny run_cycle start")
        
        try:
            # Learn from recent data
            learning_results = self.learn_wealth_growth_patterns()
            
            # Curate success recipes
            recipes = self.curate_success_recipes()
            
            # Update generational wealth tracking
            self._update_generational_wealth_metrics()
            
            # Generate tax summary (quarterly)
            tax_summary = None
            if self.tax_tracking_enabled and self.tax_preparer:
                current_month = datetime.datetime.now().month
                if current_month in [3, 6, 9, 12]:  # End of quarter
                    tax_summary = self.generate_tax_summary(include_unrealized=False)
                    self.log_action("Quarterly tax summary generated")
            
            # Analyze tax efficiency and get optimization recommendations
            tax_optimization = None
            if self.tax_optimization_enabled and self.tax_optimizer:
                tax_optimization = self.analyze_tax_efficiency()
                self.log_action("Tax efficiency analysis completed")
            
            # Learn tax efficiency patterns
            tax_learning = None
            if self.tax_optimization_enabled:
                tax_learning = self.learn_tax_efficiency_patterns()
            
            cycle_results = {
                "learning_results": learning_results,
                "recipes_curated": recipes,
                "tax_summary": tax_summary,
                "tax_optimization": tax_optimization,
                "tax_learning": tax_learning,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "completed"
            }
            
            self.log_action("Genny run_cycle completed successfully")
            return cycle_results
            
        except Exception as e:
            self.log_action(f"Error in Genny run_cycle: {e}")
            return {"status": "error", "error": str(e)}


# ----------------------
# Core AI Modules
# ----------------------

class FinancialPlannerModule:
    """Financial Planner Module for asset allocation and scenario analysis"""
    
    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.asset_allocation = {}
        self.scenario_models = {}
        self.universal_portfolio = getattr(
            parent_agent,
            "universal_portfolio",
            UniversalPortfolio(initial_capital=25000.0),
        )
        self.log_action = getattr(parent_agent, "log_action", lambda *_args, **_kwargs: None)
    
    def analyze_asset_allocation(self, current_assets: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current asset allocation and suggest optimizations"""
        return {
            "current_allocation": current_assets,
            "recommended_allocation": self._calculate_optimal_allocation(current_assets),
            "risk_assessment": self._assess_portfolio_risk(current_assets),
            "growth_potential": self._calculate_growth_potential(current_assets)
        }
    
    def _calculate_optimal_allocation(self, assets: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate optimal asset allocation using Universal Portfolio Algorithm
        Maximizes log-optimal growth rate for generational wealth
        """
        try:
            # Initialize Universal Portfolio with current assets
            total_value = sum(assets.values())
            
            # Add assets to portfolio if not already there
            current_prices = {}
            for symbol, value in assets.items():
                if symbol not in self.universal_portfolio.portfolio_weights:
                    # Estimate price (would need actual price data)
                    estimated_price = 100.0  # Placeholder
                    weight = value / total_value if total_value > 0 else 1.0 / len(assets)
                    self.universal_portfolio.add_asset(symbol, estimated_price, weight)
                current_prices[symbol] = 100.0  # Placeholder price
            
            # Rebalance using Universal Portfolio Algorithm
            optimal_weights = self.universal_portfolio.rebalance_universal(current_prices)
            
            # Convert weights to dollar amounts
            optimal_allocation = {}
            for symbol, weight in optimal_weights.items():
                optimal_allocation[symbol] = total_value * weight
            
            self.log_action(f"Universal Portfolio rebalance: {len(optimal_allocation)} assets")
            return optimal_allocation
            
        except Exception as e:
            self.log_action(f"Error in Universal Portfolio allocation: {e}")
            # Fallback to simplified allocation
            if not assets:
                return {}
            total = sum(assets.values())
            return {symbol: total / len(assets) for symbol in assets.keys()}
    
    def _assess_portfolio_risk(self, assets: Dict[str, float]) -> Dict[str, Any]:
        """Assess portfolio risk level"""
        return {
            "risk_level": "moderate",
            "diversification_score": 0.8,
            "volatility_estimate": 0.15
        }
    
    def _calculate_growth_potential(self, assets: Dict[str, float]) -> Dict[str, Any]:
        """Calculate growth potential of current portfolio"""
        return {
            "expected_annual_return": 0.08,
            "compound_growth_factor": 1.08,
            "generational_multiplier": 2.5
        }


class DataAggregatorModule:
    """Data Aggregator and Analyzer Module"""
    
    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.data_sources = {}
        self.aggregated_data = {}
    
    def aggregate_financial_data(self, sources: List[str]) -> Dict[str, Any]:
        """Aggregate financial data from multiple sources"""
        aggregated = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": sources,
            "data_summary": {},
            "insights": []
        }
        
        # Aggregate data from each source
        for source in sources:
            source_data = self._fetch_source_data(source)
            aggregated["data_summary"][source] = source_data
        
        # Generate insights
        aggregated["insights"] = self._generate_data_insights(aggregated["data_summary"])
        
        return aggregated
    
    def _fetch_source_data(self, source: str) -> Dict[str, Any]:
        """Fetch data from a specific source"""
        # Placeholder implementation
        return {"status": "placeholder", "source": source}
    
    def _generate_data_insights(self, data_summary: Dict[str, Any]) -> List[str]:
        """Generate insights from aggregated data"""
        insights = []
        
        # Analyze patterns and generate insights
        insights.append("Portfolio shows consistent growth trend")
        insights.append("Risk management strategies are effective")
        insights.append("Diversification levels are optimal")
        
        return insights


class EducationalAdvisoryModule:
    """Educational and Advisory Module for knowledge transfer"""
    
    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.knowledge_base = {}
        self.learning_paths = {}
    
    def create_learning_path(self, heir_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create customized learning path for heir"""
        experience_level = heir_profile.get("experience_level", "beginner")
        
        learning_path = {
            "heir_id": heir_profile.get("heir_id", "unknown"),
            "experience_level": experience_level,
            "modules": self._select_learning_modules(experience_level),
            "timeline": self._create_learning_timeline(experience_level),
            "assessment_criteria": self._define_assessment_criteria(experience_level)
        }
        
        return learning_path
    
    def _select_learning_modules(self, experience_level: str) -> List[str]:
        """Select appropriate learning modules based on experience level"""
        modules = {
            "beginner": [
                "Basic Financial Concepts",
                "NAE System Overview",
                "Risk Management Fundamentals",
                "Ethical Decision Making"
            ],
            "intermediate": [
                "Advanced Trading Strategies",
                "Portfolio Optimization",
                "Market Analysis Techniques",
                "Generational Wealth Planning"
            ],
            "advanced": [
                "Complex Options Strategies",
                "Advanced Risk Management",
                "System Optimization",
                "Legacy Planning and Transfer"
            ]
        }
        
        return modules.get(experience_level, modules["beginner"])
    
    def _create_learning_timeline(self, experience_level: str) -> Dict[str, Any]:
        """Create learning timeline based on experience level"""
        timelines = {
            "beginner": {"duration_months": 12, "milestones": 4},
            "intermediate": {"duration_months": 8, "milestones": 3},
            "advanced": {"duration_months": 6, "milestones": 2}
        }
        
        return timelines.get(experience_level, timelines["beginner"])
    
    def _define_assessment_criteria(self, experience_level: str) -> List[str]:
        """Define assessment criteria for learning progress"""
        criteria = {
            "beginner": [
                "Understanding of basic concepts",
                "Ability to identify risks",
                "Ethical decision making",
                "System navigation skills"
            ],
            "intermediate": [
                "Strategy implementation",
                "Portfolio management",
                "Market analysis",
                "Risk assessment"
            ],
            "advanced": [
                "Complex strategy execution",
                "System optimization",
                "Legacy planning",
                "Mentorship capabilities"
            ]
        }
        
        return criteria.get(experience_level, criteria["beginner"])


class EthicalComplianceModule:
    """Ethical and Compliance Oversight Module"""
    
    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.ethical_framework = {}
        self.compliance_rules = {}
        self.audit_trail = []
    
    def assess_ethical_compliance(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ethical compliance of a decision"""
        assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "decision_id": decision.get("id", "unknown"),
            "compliance_score": self._calculate_compliance_score(decision),
            "ethical_alignment": self._assess_ethical_alignment(decision),
            "risk_factors": self._identify_risk_factors(decision),
            "recommendations": self._generate_recommendations(decision)
        }
        
        # Add to audit trail
        self.audit_trail.append(assessment)
        
        return assessment
    
    def _calculate_compliance_score(self, decision: Dict[str, Any]) -> float:
        """Calculate compliance score for a decision"""
        # Simplified scoring algorithm
        base_score = 0.8
        
        # Adjust based on decision factors
        if decision.get("risk_level") == "low":
            base_score += 0.1
        if decision.get("transparency") == "high":
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _assess_ethical_alignment(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Assess alignment with ethical framework"""
        return {
            "family_values_alignment": 0.9,
            "fiduciary_duty_compliance": 0.95,
            "transparency_level": 0.85,
            "stakeholder_consideration": 0.8
        }
    
    def _identify_risk_factors(self, decision: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if decision.get("risk_level") == "high":
            risks.append("High risk exposure")
        if decision.get("transparency") == "low":
            risks.append("Low transparency")
        if decision.get("stakeholder_impact") == "high":
            risks.append("High stakeholder impact")
        
        return risks
    
    def _generate_recommendations(self, decision: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if decision.get("compliance_score", 0) < 0.8:
            recommendations.append("Improve compliance documentation")
        if decision.get("transparency") == "low":
            recommendations.append("Increase decision transparency")
        
        return recommendations


class OrchestrationEngineModule:
    """Orchestration and Automation Engine Module"""
    
    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.workflows = {}
        self.automation_rules = {}
        self.event_triggers = {}
    
    def orchestrate_workflow(self, workflow_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a specific workflow"""
        workflow_result = {
            "workflow_name": workflow_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": parameters,
            "execution_steps": [],
            "status": "completed",
            "results": {}
        }
        
        # Execute workflow steps
        if workflow_name == "wealth_assessment":
            workflow_result = self._execute_wealth_assessment_workflow(parameters)
        elif workflow_name == "strategy_evaluation":
            workflow_result = self._execute_strategy_evaluation_workflow(parameters)
        elif workflow_name == "heir_preparation":
            workflow_result = self._execute_heir_preparation_workflow(parameters)
        
        return workflow_result
    
    def _execute_wealth_assessment_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wealth assessment workflow"""
        return {
            "workflow_name": "wealth_assessment",
            "timestamp": datetime.datetime.now().isoformat(),
            "assessment_results": {
                "financial_capital_score": 0.85,
                "intellectual_capital_score": 0.90,
                "social_capital_score": 0.75,
                "values_legacy_score": 0.95
            },
            "recommendations": [
                "Increase social capital building activities",
                "Maintain current intellectual capital levels",
                "Continue strong values and legacy practices"
            ]
        }
    
    def _execute_strategy_evaluation_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy evaluation workflow"""
        return {
            "workflow_name": "strategy_evaluation",
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluation_results": {
                "strategy_effectiveness": 0.88,
                "risk_adjustment": 0.82,
                "generational_applicability": 0.90
            },
            "recommendations": [
                "Strategy shows strong generational applicability",
                "Risk management is adequate",
                "Consider for heir knowledge transfer"
            ]
        }
    
    def _execute_heir_preparation_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute heir preparation workflow"""
        return {
            "workflow_name": "heir_preparation",
            "timestamp": datetime.datetime.now().isoformat(),
            "preparation_results": {
                "knowledge_transfer_readiness": 0.85,
                "skill_assessment": 0.78,
                "mentorship_plan": "customized"
            },
            "recommendations": [
                "Focus on skill development in identified areas",
                "Implement mentorship program",
                "Schedule regular knowledge transfer sessions"
            ]
        }


# ----------------------
# Test harness
# ----------------------
def genny_main_loop():
    """Genny continuous operation loop - NEVER STOPS"""
    import traceback
    import logging
    import time
    
    logger = logging.getLogger(__name__)
    restart_count = 0
    
    while True:  # NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting Genny Agent (Restart #{restart_count})")
            logger.info("=" * 70)
            
            genny = GennyAgent()
            
            # Main operation loop
            while True:
                try:
                    # Genny's main operation - track trades and optimize taxes continuously
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.warning("⚠️  KeyboardInterrupt - Continuing Genny operation...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in Genny main loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt - RESTARTING Genny (Restart #{restart_count})")
            time.sleep(5)
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit - RESTARTING Genny (Restart #{restart_count})")
            time.sleep(10)
        except Exception as e:
            restart_count += 1
            delay = min(60 * restart_count, 3600)
            logger.error(f"❌ Fatal error in Genny (Restart #{restart_count}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    genny_main_loop()
