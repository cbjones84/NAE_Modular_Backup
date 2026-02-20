# agents/shredder.py
"""
ShredderAgent v6 - Wealth Allocator & Treasury Agent for NAE.

Enhancements over v5:
* Kalshi prediction market integration (CFTC-regulated)
* Section 1256 tax treatment for prediction market profits (60/40 split)
* Risk management for prediction market exposure
* Unified prediction market profit routing (Kalshi + Polymarket)

Previous features (v5):
* Secure credential retrieval via the shared ``SecretManager`` utility.
* Optional PayPal sandbox connector for automated profit deposits.
* Compliance guardrails (approval workflow, audit logging, thresholds).

KALSHI ADVANTAGES:
- CFTC regulated = legal certainty in US
- USD-denominated (no crypto required)
- Proper tax reporting (1099 forms)
- FDIC-insured funds custody
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from tools.payments import (
    PaypalClientError,
    PaypalCredentials,
    PaypalPayoutClient,
)
from tools.security.secret_manager import (
    SecretManager,
    SecretRetrievalError,
    load_paypal_sandbox_credentials,
)


# --------------------------
# Embedded Mission & 3 Main Goals
# --------------------------
MISSION_STATEMENT = (
    "To manage and preserve financial resources for the Neural Agency Engine (NAE), "
    "ensuring smart allocation, long-term stability, and contribution toward generational wealth."
)

CORE_FUNCTION_STATEMENT = (
    "Shredder intelligently allocates trading profits from Optimus: "
    "25% to Bitcoin accumulation, 25% to taxes, and 50% to wealth growth."
)

GOALS = [
    "Achieve generational wealth.",
    "Generate $5,000,000 within eight years.",
    "Optimize NAE and all agents for successful, sustainable trading.",
]


class ShredderAgent:
    def __init__(
        self,
        *,
        secret_manager: Optional[SecretManager] = None,
        payout_client: Optional[PaypalPayoutClient] = None,
        sandbox_mode: bool = False,  # LIVE MODE ONLY
    ):
        self.mission = MISSION_STATEMENT
        self.core_function = CORE_FUNCTION_STATEMENT
        self.goals = GOALS
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"
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

        self.log_file = "logs/shredder.log"
        self.audit_log_file = "logs/shredder_audit.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_log_file), exist_ok=True)

        # Wallets & Accounts (can be real API keys or sandbox placeholders)
        self.bitcoin_wallet = os.getenv("NAE_BITCOIN_WALLET", "placeholder_btc_wallet")
        self.tax_account = os.getenv("NAE_TAX_ACCOUNT", "placeholder_tax_account")
        self.wealth_account = os.getenv("NAE_WEALTH_ACCOUNT", "placeholder_wealth_account")
        self.reserve_account = os.getenv("NAE_RESERVE_ACCOUNT", "placeholder_reserve_account")

        self.transaction_history: List[Dict[str, Any]] = []
        self.last_profit_processed = 0.0

        # Messaging hooks
        self.inbox: List[Dict[str, Any]] = []
        self.outbox: List[Dict[str, Any]] = []

        # Agent references (set by scheduler)
        self.april_agent = None  # For Bitcoin conversion coordination
        self.genny_agent = None  # For generational wealth tracking

        # Compliance / payout configuration
        self.paypal_secret_manager = secret_manager
        self.paypal_client = payout_client
        self.paypal_credentials: Optional[PaypalCredentials] = None
        self.sandbox_mode = sandbox_mode
        self.paypal_receiver_email = os.getenv(
            "NAE_PAYPAL_SANDBOX_RECEIVER", "sandbox-merchant@example.com"
        )
        self.compliance_threshold = float(
            os.getenv("NAE_PAYPAL_APPROVAL_THRESHOLD", "500.0")
        )
        self.pending_payouts: List[Dict[str, Any]] = []
        self.completed_payouts: List[Dict[str, Any]] = []

        self._initialise_paypal_client()
        
        # ----------------------
        # Polymarket Integration (Prediction Market Profits)
        # ----------------------
        self.polymarket_profits = 0.0
        self.polymarket_enabled = False
        try:
            from agents.polymarket_trader import get_polymarket_trader
            self.polymarket_trader = get_polymarket_trader()
            self.polymarket_enabled = True
            self.log_action("Polymarket profit routing enabled")
        except ImportError:
            self.polymarket_trader = None
        except Exception as e:
            self.polymarket_trader = None
            self.log_action(f"Polymarket init: {e}")
        
        # ----------------------
        # Kalshi Integration (CFTC-Regulated Prediction Markets)
        # ----------------------
        self.kalshi_profits = 0.0
        self.kalshi_enabled = False
        self.kalshi_trader = None
        try:
            from agents.kalshi_trader import KalshiTrader, get_kalshi_trader
            self.kalshi_trader = get_kalshi_trader()
            self.kalshi_enabled = True
            self.log_action("✅ Kalshi profit routing enabled (CFTC-regulated)")
        except ImportError:
            self.log_action("ℹ️ Kalshi trading not available")
        except Exception as e:
            self.log_action(f"Kalshi init: {e}")
        
        # Kalshi-specific allocation rules (different from traditional trading)
        # Prediction markets have unique risk/reward profiles
        self.kalshi_allocation_rules = {
            "reinvestment_to_kalshi": 0.30,    # 30% back to prediction markets
            "to_traditional_trading": 0.40,    # 40% to Optimus stock trading
            "to_bitcoin": 0.15,                # 15% to BTC (via April)
            "to_reserve": 0.10,                # 10% to stable reserve
            "to_taxes": 0.05                   # 5% tax withholding (Section 1256)
        }
        
        # Risk limits for prediction market exposure
        self.kalshi_risk_limits = {
            "max_nav_exposure_pct": 0.15,      # Max 15% of NAV in Kalshi
            "max_single_position_pct": 0.05,  # Max 5% per position
            "daily_loss_limit_pct": 0.02,     # Stop trading if 2% NAV loss in a day
            "correlation_limit": 0.70          # Max correlation between positions
        }
        
        self.log_action("Shredder v6 initialised with prediction market expertise.")

    # --------------------------
    # Logging & auditing
    # --------------------------
    def log_action(self, message: str) -> None:
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Shredder LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Shredder LOG] {safe_message}")

    def _record_audit(self, payload: Dict[str, Any]) -> None:
        with open(self.audit_log_file, "a") as audit_file:
            audit_payload = {"timestamp": datetime.datetime.utcnow().isoformat(), **payload}
            audit_file.write(json.dumps(audit_payload) + "\n")

    # --------------------------
    # PayPal helper methods
    # --------------------------
    def _initialise_paypal_client(self) -> None:
        if self.paypal_client is not None:
            return

        secret_payload: Optional[Dict[str, Any]] = None

        if self.paypal_secret_manager:
            try:
                secret_payload = self.paypal_secret_manager.get_secret()
            except SecretRetrievalError as exc:
                self.log_action(
                    "Secret manager did not return PayPal credentials "
                    f"({exc}). Falling back to environment variables."
                )

        if secret_payload is None:
            try:
                secret_payload = load_paypal_sandbox_credentials()
            except SecretRetrievalError:
                secret_payload = None

        if secret_payload is None:
            client_id = os.getenv("PAYPAL_CLIENT_ID")
            client_secret = os.getenv("PAYPAL_SECRET") or os.getenv("PAYPAL_CLIENT_SECRET")
            if client_id and client_secret:
                secret_payload = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                }

        if secret_payload is None:
            self.log_action(
                "PayPal credentials unavailable; payouts will remain queued until "
                "credentials are supplied via the secret manager or PAYPAL_CLIENT_ID/PAYPAL_SECRET."
            )
            return

        try:
            self.paypal_credentials = PaypalCredentials(
                client_id=secret_payload["client_id"],
                client_secret=secret_payload["client_secret"],
                mode="sandbox" if self.sandbox_mode else "live",
            )
            self.paypal_client = PaypalPayoutClient(self.paypal_credentials)
            self.log_action(
                f"PayPal {'sandbox' if self.sandbox_mode else 'live'} client initialised successfully."
            )
        except (KeyError, PaypalClientError) as exc:
            self.paypal_client = None
            self.log_action(
                f"Failed to initialise PayPal client ({exc}). Payouts will remain queued."
            )

    # --------------------------
    # Allocation Logic
    # --------------------------
    def allocate_profits(self, profit_amount: float, current_nav: float = 25000.0) -> bool:
        """Allocate Optimus' profits according to long-term plan phases."""
        try:
            if profit_amount <= 0:
                self.log_action(f"Invalid profit amount: ${profit_amount:.2f}")
                return False

            if current_nav < 5000:
                phase = "Phase 1-2"
                reinvestment_pct = 1.0
                reserve_pct = 0.0
                btc_pct = 0.0
                tax_pct = 0.0
            elif current_nav < 25000:
                phase = "Phase 3"
                reinvestment_pct = 0.9
                reserve_pct = 0.1
                btc_pct = 0.0
                tax_pct = 0.0
            else:
                phase = "Phase 4"
                reinvestment_pct = 0.7
                reserve_pct = 0.3
                btc_pct = 0.25  # 25% of reserve to BTC
                tax_pct = 0.05  # 5% of reserve to taxes

            self.log_action(
                f"Processing profit distribution for ${profit_amount:.2f} ({phase}) — "
                f"{reinvestment_pct*100:.0f}% reinvestment, {reserve_pct*100:.0f}% reserve."
            )

            reinvestment = profit_amount * reinvestment_pct
            reserve = profit_amount * reserve_pct
            btc_portion = reserve * btc_pct if reserve > 0 else 0.0
            tax_portion = reserve * tax_pct if reserve > 0 else 0.0
            stable_reserve = max(reserve - btc_portion - tax_portion, 0.0)
            wealth_portion = reinvestment

            if btc_portion > 0:
                self.convert_to_bitcoin(btc_portion)
            if tax_portion > 0:
                self.transfer_to_tax_account(tax_portion)
            if stable_reserve > 0:
                self.transfer_to_reserve_account(stable_reserve)
            if wealth_portion > 0:
                self.transfer_to_wealth_account(wealth_portion)
                self._queue_paypal_payout(wealth_portion, "wealth_allocation")

            self.last_profit_processed = profit_amount
            distribution_record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "profit": profit_amount,
                "btc": btc_portion,
                "tax": tax_portion,
                "reserve": stable_reserve,
                "wealth": wealth_portion,
                "phase": phase,
            }
            self.transaction_history.append(distribution_record)

            self.log_action(
                f"Distribution complete for ${profit_amount:.2f} — "
                f"BTC: ${btc_portion:.2f}, Tax: ${tax_portion:.2f}, "
                f"Reserve: ${stable_reserve:.2f}, Wealth: ${wealth_portion:.2f}"
            )

            if self.april_agent and hasattr(self.april_agent, "receive_message"):
                self.send_message(
                    {"type": "bitcoin_migration", "content": {"allocation_amount": btc_portion}},
                    self.april_agent,
                )

            if self.genny_agent and hasattr(self.genny_agent, "receive_message"):
                self.send_message(
                    {
                        "type": "wealth_allocation",
                        "content": {
                            "amount": wealth_portion,
                            "source": "shredder_profit_allocation",
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                        },
                    },
                    self.genny_agent,
                )

            self._record_audit({"event": "profit_distribution", **distribution_record})
            self._process_pending_payouts()
            return True

        except Exception as exc:
            self.log_action(f"Error in allocate_profits: {exc}")
            return False

    # --------------------------
    # Conversion Functions
    # --------------------------
    def convert_to_bitcoin(self, usd_amount: float) -> None:
        self.log_action(f"Converting ${usd_amount:.2f} → Bitcoin wallet ({self.bitcoin_wallet})...")

    def transfer_to_tax_account(self, usd_amount: float) -> None:
        self.log_action(f"Transferring ${usd_amount:.2f} → Tax account ({self.tax_account})...")

    def transfer_to_wealth_account(self, usd_amount: float) -> None:
        self.log_action(f"Allocating ${usd_amount:.2f} → Wealth account ({self.wealth_account})...")

    def transfer_to_reserve_account(self, usd_amount: float) -> None:
        self.log_action(f"Allocating ${usd_amount:.2f} → Reserve account ({self.reserve_account})...")

    # --------------------------
    # PayPal payout workflow
    # --------------------------
    def _queue_paypal_payout(self, usd_amount: float, reason: str) -> None:
        payout_id = f"payout_{uuid.uuid4().hex}"
        status = "pending_approval" if usd_amount >= self.compliance_threshold else "auto"
        entry = {
            "payout_id": payout_id,
            "amount": round(float(usd_amount), 2),
            "currency": "USD",
            "reason": reason,
            "status": status,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        self.pending_payouts.append(entry)
        self._record_audit({"event": "payout_queued", **entry})
        self.log_action(
            f"Queued PayPal payout ${usd_amount:.2f} ({reason}) — status '{status}'."
        )

    def approve_payout(self, payout_id: str, approver: str) -> bool:
        for payout in self.pending_payouts:
            if payout["payout_id"] == payout_id:
                payout["status"] = "approved"
                payout["approved_by"] = approver
                payout["approved_at"] = datetime.datetime.utcnow().isoformat()
                self._record_audit({"event": "payout_approved", **payout})
                self.log_action(f"Payout {payout_id} approved by {approver}.")
                self._process_pending_payouts()
                return True
        self.log_action(f"Payout {payout_id} not found for approval.")
        return False

    def list_pending_payouts(self) -> List[Dict[str, Any]]:
        return [
            payout
            for payout in self.pending_payouts
            if payout["status"] in {"pending_approval", "auto", "approved", "error"}
        ]

    def _process_pending_payouts(self) -> None:
        if not self.paypal_client:
            self._initialise_paypal_client()
        if not self.paypal_client:
            return

        for payout in list(self.pending_payouts):
            status = payout["status"]
            if status == "pending_approval":
                continue
            if status == "auto":
                payout["status"] = "approved"

            if payout["status"] != "approved":
                continue

            try:
                response = self.paypal_client.create_payout(
                    amount=f"{payout['amount']:.2f}",
                    currency=payout.get("currency", "USD"),
                    recipient_email=self.paypal_receiver_email,
                    note=f"NAE Shredder payout ({payout['reason']})",
                    sender_batch_id=payout["payout_id"],
                )
                payout["status"] = "completed"
                payout["completed_at"] = datetime.datetime.utcnow().isoformat()
                payout["response"] = response
                self.completed_payouts.append(payout)
                self.pending_payouts.remove(payout)
                self._record_audit({"event": "payout_completed", **payout})
                self.log_action(
                    f"PayPal payout {payout['payout_id']} completed for ${payout['amount']:.2f}."
                )
            except PaypalClientError as exc:
                payout["status"] = "error"
                payout["error"] = str(exc)
                self._record_audit({"event": "payout_error", **payout})
                self.log_action(
                    f"PayPal payout {payout['payout_id']} failed: {exc}. Manual review required."
                )

    # --------------------------
    # Profit Intake (from Optimus)
    # --------------------------
    def receive_profit_data(self, data: Dict[str, Any]) -> bool:
        try:
            if not isinstance(data, dict):
                self.log_action(f"Invalid profit data format: expected dict, got {type(data)}")
                return False

            profit = data.get("profit", 0)
            trade_id = data.get("trade_id", "unknown")

            if not isinstance(profit, (int, float)):
                self.log_action(f"Invalid profit value: {profit}")
                return False

            self.log_action(f"Received profit data from Optimus — Trade {trade_id}: ${profit:.2f}")

            if profit > 0:
                return self.allocate_profits(float(profit))
            self.log_action(f"No profit to allocate for trade {trade_id}.")
            return False

        except Exception as exc:
            self.log_action(f"Error in receive_profit_data: {exc}")
            return False

    # --------------------------
    # Polymarket Profit Collection
    # --------------------------
    def collect_polymarket_profits(self) -> Dict[str, Any]:
        """
        Collect and allocate profits from Polymarket prediction market trading.
        
        Returns:
            Dict with profit info and allocation status
        """
        if not self.polymarket_enabled or not self.polymarket_trader:
            return {"status": "disabled", "profit": 0.0}
        
        try:
            # Get realized profits from Polymarket
            profit = self.polymarket_trader.get_profit_for_shredder()
            
            if profit <= 0:
                return {"status": "no_profit", "profit": 0.0}
            
            self.log_action(f"Collecting Polymarket profit: ${profit:.2f}")
            
            # Allocate using standard profit allocation
            allocated = self.allocate_profits(profit)
            
            if allocated:
                self.polymarket_profits += profit
                # Notify Polymarket trader of allocation
                self.polymarket_trader.record_profit_allocation(
                    profit, "shredder_standard_allocation"
                )
                
                return {
                    "status": "allocated",
                    "profit": profit,
                    "total_polymarket_profits": self.polymarket_profits
                }
            
            return {"status": "allocation_failed", "profit": profit}
            
        except Exception as e:
            self.log_action(f"Error collecting Polymarket profits: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_polymarket_status(self) -> Dict[str, Any]:
        """Get Polymarket integration status"""
        return {
            "enabled": self.polymarket_enabled,
            "total_profits_collected": self.polymarket_profits,
            "trader_available": self.polymarket_trader is not None
        }

    # --------------------------
    # Kalshi Profit Collection & Risk Management
    # --------------------------
    def allocate_kalshi_profits(self, profit_amount: float, current_nav: float = 25000.0) -> Dict[str, Any]:
        """
        Specialized allocation for Kalshi prediction market profits.
        
        Kalshi profits have unique characteristics:
        - Section 1256 tax treatment (60/40 long-term/short-term)
        - CFTC-regulated = proper 1099 reporting
        - Different risk profile than traditional trading
        
        Args:
            profit_amount: Profit in USD from Kalshi trades
            current_nav: Current NAV for phase-based allocation
            
        Returns:
            Dict with allocation breakdown and status
        """
        if profit_amount <= 0:
            self.log_action(f"Invalid Kalshi profit amount: ${profit_amount:.2f}")
            return {"status": "invalid_amount", "profit": profit_amount}
        
        try:
            # Calculate allocations based on Kalshi-specific rules
            allocation = {
                "to_kalshi": profit_amount * self.kalshi_allocation_rules["reinvestment_to_kalshi"],
                "to_optimus": profit_amount * self.kalshi_allocation_rules["to_traditional_trading"],
                "to_bitcoin": profit_amount * self.kalshi_allocation_rules["to_bitcoin"],
                "to_reserve": profit_amount * self.kalshi_allocation_rules["to_reserve"],
                "to_taxes": profit_amount * self.kalshi_allocation_rules["to_taxes"],
                "total_profit": profit_amount,
                "tax_treatment": "section_1256_60_40"
            }
            
            self.log_action(
                f"💰 Kalshi profit ${profit_amount:.2f} allocation: "
                f"Kalshi: ${allocation['to_kalshi']:.2f}, "
                f"Optimus: ${allocation['to_optimus']:.2f}, "
                f"BTC: ${allocation['to_bitcoin']:.2f}, "
                f"Reserve: ${allocation['to_reserve']:.2f}, "
                f"Taxes: ${allocation['to_taxes']:.2f}"
            )
            
            # Route Bitcoin allocation to April
            if allocation["to_bitcoin"] > 0 and self.april_agent:
                self.send_message({
                    "type": "bitcoin_migration",
                    "content": {
                        "allocation_amount": allocation["to_bitcoin"],
                        "source": "kalshi_profits",
                        "tax_treatment": "section_1256"
                    }
                }, self.april_agent)
            
            # Route tax reserve (with Section 1256 notation)
            if allocation["to_taxes"] > 0:
                self.transfer_to_tax_account(allocation["to_taxes"])
            
            # Route reserve
            if allocation["to_reserve"] > 0:
                self.transfer_to_reserve_account(allocation["to_reserve"])
            
            # Notify Genny for tax tracking and wealth management
            if self.genny_agent:
                self.send_message({
                    "type": "kalshi_profit_allocation",
                    "content": {
                        **allocation,
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "regulatory_status": "CFTC_regulated"
                    }
                }, self.genny_agent)
            
            # Track profit
            self.kalshi_profits += profit_amount
            
            # Record in transaction history
            transaction_record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "kalshi",
                "profit": profit_amount,
                **allocation
            }
            self.transaction_history.append(transaction_record)
            self._record_audit({"event": "kalshi_profit_allocation", **transaction_record})
            
            allocation["status"] = "allocated"
            allocation["total_kalshi_profits"] = self.kalshi_profits
            
            return allocation
            
        except Exception as e:
            self.log_action(f"Error allocating Kalshi profits: {e}")
            return {"status": "error", "error": str(e), "profit": profit_amount}
    
    def collect_kalshi_profits(self) -> Dict[str, Any]:
        """
        Collect and allocate profits from Kalshi prediction market trading.
        
        Returns:
            Dict with profit info and allocation status
        """
        if not self.kalshi_enabled or not self.kalshi_trader:
            return {"status": "disabled", "profit": 0.0}
        
        try:
            # Get realized profits from Kalshi trader
            if hasattr(self.kalshi_trader, 'get_profit_for_shredder'):
                profit = self.kalshi_trader.get_profit_for_shredder()
            elif hasattr(self.kalshi_trader, 'get_realized_pnl'):
                profit = self.kalshi_trader.get_realized_pnl()
            else:
                # Fallback: check positions
                profit = self._calculate_kalshi_realized_profit()
            
            if profit <= 0:
                return {"status": "no_profit", "profit": 0.0}
            
            self.log_action(f"📈 Collecting Kalshi profit: ${profit:.2f}")
            
            # Allocate using Kalshi-specific allocation
            result = self.allocate_kalshi_profits(profit)
            
            if result.get("status") == "allocated":
                # Notify Kalshi trader of allocation
                if hasattr(self.kalshi_trader, 'record_profit_allocation'):
                    self.kalshi_trader.record_profit_allocation(
                        profit, "shredder_kalshi_allocation"
                    )
            
            return result
            
        except Exception as e:
            self.log_action(f"Error collecting Kalshi profits: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_kalshi_realized_profit(self) -> float:
        """Calculate realized profit from Kalshi positions (fallback method)"""
        try:
            if not self.kalshi_trader or not hasattr(self.kalshi_trader, 'adapter'):
                return 0.0
            
            adapter = self.kalshi_trader.adapter
            if not adapter:
                return 0.0
            
            # Get settled positions
            positions = adapter.get_positions() if hasattr(adapter, 'get_positions') else []
            
            realized_pnl = 0.0
            for pos in positions:
                if hasattr(pos, 'realized_pnl'):
                    realized_pnl += pos.realized_pnl
                elif isinstance(pos, dict):
                    realized_pnl += pos.get('realized_pnl', 0)
            
            return realized_pnl / 100  # Convert cents to dollars
            
        except Exception:
            return 0.0
    
    def manage_kalshi_risk_limits(self, current_nav: float = 25000.0) -> Dict[str, Any]:
        """
        Enforce risk limits for Kalshi trading across NAE.
        Prevents over-concentration in prediction markets.
        
        Args:
            current_nav: Current total NAV
            
        Returns:
            Dict with risk status and available capacity
        """
        try:
            # Calculate maximum allowed Kalshi exposure
            max_exposure = current_nav * self.kalshi_risk_limits["max_nav_exposure_pct"]
            
            # Get current Kalshi exposure
            current_exposure = self._get_kalshi_exposure()
            
            # Calculate utilization
            utilization_pct = (current_exposure / max_exposure * 100) if max_exposure > 0 else 0
            
            # Determine if more positions can be added
            available_capital = max(0, max_exposure - current_exposure)
            can_add_positions = current_exposure < max_exposure
            
            # Check daily loss limit
            daily_pnl = self._get_kalshi_daily_pnl()
            daily_loss_limit = current_nav * self.kalshi_risk_limits["daily_loss_limit_pct"]
            within_daily_limit = daily_pnl > -daily_loss_limit
            
            risk_status = {
                "max_exposure": max_exposure,
                "current_exposure": current_exposure,
                "utilization_pct": round(utilization_pct, 2),
                "available_capital": available_capital,
                "can_add_positions": can_add_positions and within_daily_limit,
                "daily_pnl": daily_pnl,
                "daily_loss_limit": daily_loss_limit,
                "within_daily_limit": within_daily_limit,
                "risk_level": self._assess_kalshi_risk_level(utilization_pct, daily_pnl, daily_loss_limit)
            }
            
            # Log if approaching limits
            if utilization_pct > 80:
                self.log_action(f"⚠️ Kalshi exposure at {utilization_pct:.1f}% of limit")
            if not within_daily_limit:
                self.log_action(f"🛑 Kalshi daily loss limit reached: ${daily_pnl:.2f}")
            
            return risk_status
            
        except Exception as e:
            self.log_action(f"Error checking Kalshi risk limits: {e}")
            return {
                "status": "error",
                "error": str(e),
                "can_add_positions": False
            }
    
    def _get_kalshi_exposure(self) -> float:
        """Get current Kalshi position exposure in USD"""
        try:
            if not self.kalshi_trader:
                return 0.0
            
            if hasattr(self.kalshi_trader, 'adapter') and self.kalshi_trader.adapter:
                adapter = self.kalshi_trader.adapter
                
                # Try to get balance/positions
                if hasattr(adapter, 'get_balance'):
                    balance_info = adapter.get_balance()
                    if isinstance(balance_info, dict):
                        # Return portfolio value in dollars
                        return balance_info.get('portfolio_value', 0) / 100
                
                # Fallback: sum position values
                if hasattr(adapter, 'get_positions'):
                    positions = adapter.get_positions()
                    total_value = 0.0
                    for pos in positions:
                        if hasattr(pos, 'market_exposure'):
                            total_value += pos.market_exposure
                        elif isinstance(pos, dict):
                            total_value += pos.get('market_exposure', 0)
                    return total_value / 100
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_kalshi_daily_pnl(self) -> float:
        """Get Kalshi P&L for current day"""
        try:
            if not self.kalshi_trader:
                return 0.0
            
            if hasattr(self.kalshi_trader, 'get_daily_pnl'):
                return self.kalshi_trader.get_daily_pnl()
            
            # Fallback: check today's transactions
            today = datetime.datetime.utcnow().date().isoformat()
            daily_pnl = 0.0
            
            for txn in self.transaction_history:
                if txn.get("source") == "kalshi" and txn.get("timestamp", "").startswith(today):
                    daily_pnl += txn.get("profit", 0)
            
            return daily_pnl
            
        except Exception:
            return 0.0
    
    def _assess_kalshi_risk_level(self, utilization_pct: float, daily_pnl: float, daily_limit: float) -> str:
        """Assess overall Kalshi risk level"""
        if daily_pnl < -daily_limit:
            return "CRITICAL"
        if utilization_pct > 90 or daily_pnl < -daily_limit * 0.8:
            return "HIGH"
        if utilization_pct > 70 or daily_pnl < -daily_limit * 0.5:
            return "MODERATE"
        return "LOW"
    
    def get_kalshi_status(self) -> Dict[str, Any]:
        """Get comprehensive Kalshi integration status"""
        return {
            "enabled": self.kalshi_enabled,
            "total_profits_collected": self.kalshi_profits,
            "trader_available": self.kalshi_trader is not None,
            "allocation_rules": self.kalshi_allocation_rules,
            "risk_limits": self.kalshi_risk_limits,
            "regulatory_status": "CFTC_regulated",
            "tax_treatment": "Section 1256 (60/40)"
        }
    
    def get_prediction_market_summary(self) -> Dict[str, Any]:
        """Get summary of all prediction market activities (Kalshi + Polymarket)"""
        return {
            "kalshi": self.get_kalshi_status(),
            "polymarket": self.get_polymarket_status(),
            "total_prediction_market_profits": self.kalshi_profits + self.polymarket_profits,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

    # --------------------------
    # Agent loop
    # --------------------------
    def run(self) -> Dict[str, Any]:
        try:
            self.log_action("Shredder v6 run cycle started")

            processed_count = 0
            total_profit = 0.0
            prediction_market_profits = 0.0

            # Process inbox messages (Optimus profits)
            while self.inbox:
                message = self.inbox.pop(0)
                if not isinstance(message, dict):
                    continue
                msg_type = message.get("type", "")
                content = message.get("content", {})
                if msg_type == "profit_data" or "profit" in message:
                    profit_data = content if content else message
                    if self.receive_profit_data(profit_data):
                        processed_count += 1
                        total_profit += float(profit_data.get("profit", 0))
            
            # Collect Kalshi prediction market profits
            if self.kalshi_enabled:
                kalshi_result = self.collect_kalshi_profits()
                if kalshi_result.get("status") == "allocated":
                    kalshi_profit = kalshi_result.get("total_profit", 0)
                    prediction_market_profits += kalshi_profit
                    total_profit += kalshi_profit
                    processed_count += 1
                    self.log_action(f"📊 Kalshi profits collected: ${kalshi_profit:.2f}")
            
            # Collect Polymarket prediction market profits
            if self.polymarket_enabled:
                poly_result = self.collect_polymarket_profits()
                if poly_result.get("status") == "allocated":
                    poly_profit = poly_result.get("profit", 0)
                    prediction_market_profits += poly_profit
                    total_profit += poly_profit
                    processed_count += 1

            self._process_pending_payouts()

            result = {
                "status": "success",
                "agent": "Shredder",
                "version": "v6",
                "transactions_processed": processed_count,
                "total_profit_allocated": total_profit,
                "prediction_market_profits": prediction_market_profits,
                "kalshi_total": self.kalshi_profits,
                "polymarket_total": self.polymarket_profits,
                "transaction_history_size": len(self.transaction_history),
                "pending_payouts": len(self.list_pending_payouts()),
                "completed_payouts": len(self.completed_payouts),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
            self.log_action(f"Shredder v6 run cycle completed: {result}")
            return result

        except Exception as exc:
            self.log_action(f"Error in Shredder run cycle: {exc}")
            return {
                "status": "error",
                "agent": "Shredder",
                "error": str(exc),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }

    # --------------------------
    # Messaging
    # --------------------------
    def receive_message(self, message: dict) -> None:
        try:
            if not isinstance(message, dict):
                self.log_action(f"Invalid message format: expected dict, got {type(message)}")
                return

            self.inbox.append(message)
            self.log_action(f"Received message: {message}")

            if message.get("type") == "profit_data" or "profit" in message:
                self.receive_profit_data(message.get("content", message))

        except Exception as exc:
            self.log_action(f"Error receiving message: {exc}")

    def send_message(self, message: dict, recipient_agent) -> bool:
        """Send a message to another agent"""
        try:
            if hasattr(recipient_agent, "receive_message"):
                # Try (sender, message) signature first
                try:
                    recipient_agent.receive_message(self.__class__.__name__, message)
                except TypeError:
                    # Fall back to (message) signature
                    recipient_agent.receive_message(message)
                self.log_action(f"Sent message to {recipient_agent.__class__.__name__}")
                return True
            else:
                self.log_action(f"Recipient agent does not support receive_message")
                return False
        except Exception as e:
            self.log_action(f"Error sending message: {e}")
            return False
