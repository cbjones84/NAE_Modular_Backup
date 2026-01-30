#!/usr/bin/env python3
"""
Genny Tax Preparation and Assessment Module
Comprehensive tax tracking, calculation, and preparation for NAE
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import sys

logger = logging.getLogger(__name__)


class CostBasisMethod(Enum):
    """Cost basis calculation methods"""
    FIFO = "FIFO"  # First In, First Out
    LIFO = "LIFO"  # Last In, First Out
    HIFO = "HIFO"  # Highest In, First Out
    AVERAGE = "AVERAGE"  # Average Cost


class HoldingPeriod(Enum):
    """Holding period for capital gains"""
    SHORT_TERM = "SHORT_TERM"  # <= 1 year
    LONG_TERM = "LONG_TERM"  # > 1 year


@dataclass
class TradeRecord:
    """Record of a trade for tax purposes"""
    trade_id: str
    timestamp: str
    symbol: str
    asset_type: str  # "stock", "option", "crypto", "other"
    action: str  # "buy", "sell", "exercise", "expire"
    quantity: float
    price: float
    cost_basis: float
    proceeds: float
    fees: float
    agent: str  # "optimus", "shredder", "april", "donnie", "mikey"
    holding_period_days: int
    is_day_trade: bool
    notes: str = ""


@dataclass
class Position:
    """Current position for cost basis tracking"""
    symbol: str
    asset_type: str
    quantity: float
    cost_basis: float
    average_cost: float
    lots: List[Dict[str, Any]]  # List of purchase lots
    first_purchase_date: str
    last_purchase_date: str


@dataclass
class ExpenseRecord:
    """Deductible expense record"""
    expense_id: str
    timestamp: str
    category: str  # "software", "hardware", "subscription", "professional", "other"
    description: str
    amount: float
    deductible_pct: float  # Percentage deductible (0-100)
    business_use_pct: float  # Business use percentage
    receipt_path: Optional[str] = None
    tax_year: int = None


@dataclass
class CapitalGain:
    """Capital gain/loss calculation"""
    symbol: str
    asset_type: str
    sale_date: str
    sale_proceeds: float
    cost_basis: float
    fees: float
    gain_loss: float
    holding_period: HoldingPeriod
    is_day_trade: bool
    short_term_gain: float
    long_term_gain: float


class TaxPreparer:
    """
    Comprehensive tax preparation and assessment system
    """
    
    def __init__(self, data_dir: str = "tools/data/genny/tax"):
        """Initialize tax preparer"""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tax data storage
        self.trades: List[TradeRecord] = []
        self.positions: Dict[str, Position] = {}
        self.expenses: List[ExpenseRecord] = []
        self.capital_gains: List[CapitalGain] = []
        
        # Cost basis method (default FIFO, can be changed)
        self.cost_basis_method = CostBasisMethod.FIFO
        
        # Tax year
        self.current_tax_year = datetime.now().year
        
        # North Carolina tax rates (2024)
        self.nc_tax_rates = {
            "individual_income": {
                "rate": 0.0475,  # 4.75% flat rate
                "description": "North Carolina individual income tax rate"
            },
            "capital_gains": {
                "short_term_rate": 0.0475,  # Same as income tax
                "long_term_rate": 0.0475,  # Same as income tax (NC doesn't differentiate)
                "description": "NC treats all capital gains as ordinary income"
            }
        }
        
        # Federal tax rates (2024)
        self.federal_tax_rates = {
            "capital_gains": {
                "short_term": {
                    "brackets": [
                        (0, 0.10),
                        (11000, 0.12),
                        (44725, 0.22),
                        (95375, 0.24),
                        (201050, 0.32),
                        (243725, 0.35),
                        (609350, 0.37)
                    ],
                    "description": "Short-term capital gains taxed as ordinary income"
                },
                "long_term": {
                    "brackets": [
                        (0, 0.0),
                        (44625, 0.15),
                        (492300, 0.20),
                        (553850, 0.20)  # Additional 3.8% NIIT may apply
                    ],
                    "description": "Long-term capital gains preferential rates"
                }
            }
        }
        
        # Load existing data
        self._load_tax_data()
        
        logger.info("Tax Preparer initialized")
    
    def _load_tax_data(self):
        """Load existing tax data from files"""
        try:
            # Load trades
            trades_file = os.path.join(self.data_dir, "trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, "r") as f:
                    trades_data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in trades_data]
            
            # Load positions
            positions_file = os.path.join(self.data_dir, "positions.json")
            if os.path.exists(positions_file):
                with open(positions_file, "r") as f:
                    positions_data = json.load(f)
                    self.positions = {
                        k: Position(**v) for k, v in positions_data.items()
                    }
            
            # Load expenses
            expenses_file = os.path.join(self.data_dir, "expenses.json")
            if os.path.exists(expenses_file):
                with open(expenses_file, "r") as f:
                    expenses_data = json.load(f)
                    self.expenses = [ExpenseRecord(**e) for e in expenses_data]
            
            # Load capital gains
            gains_file = os.path.join(self.data_dir, "capital_gains.json")
            if os.path.exists(gains_file):
                with open(gains_file, "r") as f:
                    gains_data = json.load(f)
                    self.capital_gains = [CapitalGain(**g) for g in gains_data]
        
        except Exception as e:
            logger.error(f"Error loading tax data: {e}")
    
    def _save_tax_data(self):
        """Save tax data to files"""
        try:
            # Save trades
            trades_file = os.path.join(self.data_dir, "trades.json")
            with open(trades_file, "w") as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2, default=str)
            
            # Save positions
            positions_file = os.path.join(self.data_dir, "positions.json")
            with open(positions_file, "w") as f:
                json.dump(
                    {k: asdict(v) for k, v in self.positions.items()},
                    f,
                    indent=2,
                    default=str
                )
            
            # Save expenses
            expenses_file = os.path.join(self.data_dir, "expenses.json")
            with open(expenses_file, "w") as f:
                json.dump([asdict(e) for e in self.expenses], f, indent=2, default=str)
            
            # Save capital gains
            gains_file = os.path.join(self.data_dir, "capital_gains.json")
            with open(gains_file, "w") as f:
                json.dump([asdict(g) for g in self.capital_gains], f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error saving tax data: {e}")
    
    def record_trade(
        self,
        trade_id: str,
        timestamp: str,
        symbol: str,
        asset_type: str,
        action: str,
        quantity: float,
        price: float,
        fees: float,
        agent: str,
        notes: str = ""
    ) -> TradeRecord:
        """
        Record a trade for tax purposes
        
        Args:
            trade_id: Unique trade identifier
            timestamp: Trade timestamp (ISO format)
            symbol: Asset symbol
            asset_type: Type of asset (stock, option, crypto, etc.)
            action: Trade action (buy, sell, etc.)
            quantity: Number of shares/units
            price: Price per share/unit
            fees: Trading fees/commissions
            agent: Agent that executed trade
            notes: Additional notes
        
        Returns:
            TradeRecord object
        """
        trade_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        
        # Calculate cost basis and proceeds
        if action.lower() == "buy":
            cost_basis = (quantity * price) + fees
            proceeds = 0.0
        elif action.lower() == "sell":
            proceeds = (quantity * price) - fees
            # Calculate cost basis based on method
            cost_basis = self._calculate_cost_basis(symbol, quantity, trade_time)
        else:
            cost_basis = (quantity * price) + fees
            proceeds = 0.0
        
        # Calculate holding period
        holding_period_days = self._calculate_holding_period(symbol, trade_time)
        
        # Check if day trade
        is_day_trade = self._is_day_trade(symbol, trade_time)
        
        # Create trade record
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            quantity=quantity,
            price=price,
            cost_basis=cost_basis,
            proceeds=proceeds,
            fees=fees,
            agent=agent,
            holding_period_days=holding_period_days,
            is_day_trade=is_day_trade,
            notes=notes
        )
        
        self.trades.append(trade)
        
        # Update positions
        self._update_position(symbol, asset_type, action, quantity, price, fees, trade_time)
        
        # Calculate capital gain if sale
        if action.lower() == "sell":
            capital_gain = self._calculate_capital_gain(trade)
            if capital_gain:
                self.capital_gains.append(capital_gain)
        
        # Save data
        self._save_tax_data()
        
        logger.info(f"Recorded trade: {symbol} {action} {quantity} @ ${price:.2f}")
        
        return trade
    
    def _calculate_cost_basis(
        self,
        symbol: str,
        quantity: float,
        sale_date: datetime
    ) -> float:
        """
        Calculate cost basis using selected method (FIFO/LIFO/HIFO)
        
        Args:
            symbol: Asset symbol
            quantity: Quantity being sold
            sale_date: Sale date
        
        Returns:
            Cost basis for the sale
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}, using zero cost basis")
            return 0.0
        
        position = self.positions[symbol]
        
        if not position.lots:
            return position.average_cost * quantity
        
        remaining_quantity = quantity
        total_cost_basis = 0.0
        
        # Sort lots based on method
        if self.cost_basis_method == CostBasisMethod.FIFO:
            # First In, First Out - oldest lots first
            sorted_lots = sorted(position.lots, key=lambda x: x["purchase_date"])
        elif self.cost_basis_method == CostBasisMethod.LIFO:
            # Last In, First Out - newest lots first
            sorted_lots = sorted(position.lots, key=lambda x: x["purchase_date"], reverse=True)
        elif self.cost_basis_method == CostBasisMethod.HIFO:
            # Highest In, First Out - highest cost lots first
            sorted_lots = sorted(position.lots, key=lambda x: x["cost_per_unit"], reverse=True)
        else:  # AVERAGE
            # Average cost method
            return position.average_cost * quantity
        
        # Allocate cost basis from lots
        for lot in sorted_lots:
            if remaining_quantity <= 0:
                break
            
            lot_quantity = lot["quantity"]
            lot_cost_per_unit = lot["cost_per_unit"]
            
            if lot_quantity <= remaining_quantity:
                # Use entire lot
                total_cost_basis += lot_quantity * lot_cost_per_unit
                remaining_quantity -= lot_quantity
            else:
                # Use partial lot
                total_cost_basis += remaining_quantity * lot_cost_per_unit
                remaining_quantity = 0
        
        return total_cost_basis
    
    def _calculate_holding_period(self, symbol: str, sale_date: datetime) -> int:
        """Calculate holding period in days"""
        if symbol not in self.positions:
            return 0
        
        position = self.positions[symbol]
        first_purchase = datetime.fromisoformat(position.first_purchase_date.replace("Z", "+00:00"))
        
        return (sale_date - first_purchase).days
    
    def _is_day_trade(self, symbol: str, trade_date: datetime) -> bool:
        """Check if trade is a day trade (bought and sold same day)"""
        # Check if there was a purchase of same symbol on same day
        same_day_trades = [
            t for t in self.trades
            if t.symbol == symbol
            and datetime.fromisoformat(t.timestamp.replace("Z", "+00:00")).date() == trade_date.date()
        ]
        
        buy_trades = [t for t in same_day_trades if t.action.lower() == "buy"]
        sell_trades = [t for t in same_day_trades if t.action.lower() == "sell"]
        
        return len(buy_trades) > 0 and len(sell_trades) > 0
    
    def _update_position(
        self,
        symbol: str,
        asset_type: str,
        action: str,
        quantity: float,
        price: float,
        fees: float,
        trade_date: datetime
    ):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                asset_type=asset_type,
                quantity=0.0,
                cost_basis=0.0,
                average_cost=0.0,
                lots=[],
                first_purchase_date=trade_date.isoformat(),
                last_purchase_date=trade_date.isoformat()
            )
        
        position = self.positions[symbol]
        trade_date_str = trade_date.isoformat()
        
        if action.lower() == "buy":
            # Add to position
            cost_per_unit = (price * quantity + fees) / quantity
            position.lots.append({
                "purchase_date": trade_date_str,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "cost_per_unit": cost_per_unit,
                "total_cost": price * quantity + fees
            })
            
            position.quantity += quantity
            position.cost_basis += (price * quantity + fees)
            position.average_cost = position.cost_basis / position.quantity
            position.last_purchase_date = trade_date_str
        
        elif action.lower() == "sell":
            # Reduce position
            position.quantity -= quantity
            
            # Remove lots based on cost basis method
            remaining_quantity = quantity
            
            if self.cost_basis_method == CostBasisMethod.FIFO:
                sorted_lots = sorted(position.lots, key=lambda x: x["purchase_date"])
            elif self.cost_basis_method == CostBasisMethod.LIFO:
                sorted_lots = sorted(position.lots, key=lambda x: x["purchase_date"], reverse=True)
            elif self.cost_basis_method == CostBasisMethod.HIFO:
                sorted_lots = sorted(position.lots, key=lambda x: x["cost_per_unit"], reverse=True)
            else:
                sorted_lots = position.lots
            
            for lot in sorted_lots:
                if remaining_quantity <= 0:
                    break
                
                if lot["quantity"] <= remaining_quantity:
                    # Remove entire lot
                    position.cost_basis -= lot["total_cost"]
                    remaining_quantity -= lot["quantity"]
                    position.lots.remove(lot)
                else:
                    # Reduce lot
                    reduction_pct = remaining_quantity / lot["quantity"]
                    position.cost_basis -= lot["total_cost"] * reduction_pct
                    lot["quantity"] -= remaining_quantity
                    lot["total_cost"] = lot["quantity"] * lot["cost_per_unit"]
                    remaining_quantity = 0
            
            # Recalculate average cost
            if position.quantity > 0:
                position.average_cost = position.cost_basis / position.quantity
            else:
                position.average_cost = 0.0
                position.cost_basis = 0.0
    
    def _calculate_capital_gain(self, trade: TradeRecord) -> Optional[CapitalGain]:
        """Calculate capital gain/loss for a sale"""
        if trade.action.lower() != "sell":
            return None
        
        sale_date = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
        holding_period = HoldingPeriod.LONG_TERM if trade.holding_period_days > 365 else HoldingPeriod.SHORT_TERM
        
        gain_loss = trade.proceeds - trade.cost_basis
        
        if holding_period == HoldingPeriod.SHORT_TERM:
            short_term_gain = gain_loss
            long_term_gain = 0.0
        else:
            short_term_gain = 0.0
            long_term_gain = gain_loss
        
        capital_gain = CapitalGain(
            symbol=trade.symbol,
            asset_type=trade.asset_type,
            sale_date=trade.timestamp,
            sale_proceeds=trade.proceeds,
            cost_basis=trade.cost_basis,
            fees=trade.fees,
            gain_loss=gain_loss,
            holding_period=holding_period,
            is_day_trade=trade.is_day_trade,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain
        )
        
        return capital_gain
    
    def record_expense(
        self,
        expense_id: str,
        timestamp: str,
        category: str,
        description: str,
        amount: float,
        deductible_pct: float = 100.0,
        business_use_pct: float = 100.0,
        receipt_path: Optional[str] = None
    ) -> ExpenseRecord:
        """
        Record a deductible expense
        
        Args:
            expense_id: Unique expense identifier
            timestamp: Expense timestamp
            category: Expense category
            description: Expense description
            amount: Expense amount
            deductible_pct: Percentage deductible (0-100)
            business_use_pct: Business use percentage (0-100)
            receipt_path: Path to receipt if available
        
        Returns:
            ExpenseRecord object
        """
        expense_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        tax_year = expense_time.year
        
        expense = ExpenseRecord(
            expense_id=expense_id,
            timestamp=timestamp,
            category=category,
            description=description,
            amount=amount,
            deductible_pct=deductible_pct,
            business_use_pct=business_use_pct,
            receipt_path=receipt_path,
            tax_year=tax_year
        )
        
        self.expenses.append(expense)
        self._save_tax_data()
        
        logger.info(f"Recorded expense: {category} - ${amount:.2f}")
        
        return expense
    
    def get_tax_summary(
        self,
        tax_year: Optional[int] = None,
        include_unrealized: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive tax summary
        
        Args:
            tax_year: Tax year (defaults to current year)
            include_unrealized: Include unrealized gains/losses
        
        Returns:
            Comprehensive tax summary
        """
        if tax_year is None:
            tax_year = self.current_tax_year
        
        # Filter trades and gains for tax year
        year_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t.timestamp.replace("Z", "+00:00")).year == tax_year
        ]
        
        year_gains = [
            g for g in self.capital_gains
            if datetime.fromisoformat(g.sale_date.replace("Z", "+00:00")).year == tax_year
        ]
        
        year_expenses = [e for e in self.expenses if e.tax_year == tax_year]
        
        # Calculate totals
        total_short_term_gain = sum(g.short_term_gain for g in year_gains)
        total_long_term_gain = sum(g.long_term_gain for g in year_gains)
        total_capital_gain = total_short_term_gain + total_long_term_gain
        
        # Calculate deductible expenses
        total_expenses = sum(e.amount for e in year_expenses)
        deductible_expenses = sum(
            e.amount * (e.deductible_pct / 100.0) * (e.business_use_pct / 100.0)
            for e in year_expenses
        )
        
        # Calculate unrealized gains if requested
        unrealized_gains = {}
        if include_unrealized:
            unrealized_gains = self._calculate_unrealized_gains()
        
        # Calculate estimated taxes
        estimated_taxes = self._calculate_estimated_taxes(
            total_short_term_gain,
            total_long_term_gain,
            deductible_expenses
        )
        
        summary = {
            "tax_year": tax_year,
            "generated_at": datetime.now().isoformat(),
            "trades": {
                "total_trades": len(year_trades),
                "buy_trades": len([t for t in year_trades if t.action.lower() == "buy"]),
                "sell_trades": len([t for t in year_trades if t.action.lower() == "sell"]),
                "day_trades": len([t for t in year_trades if t.is_day_trade]),
                "by_agent": self._group_trades_by_agent(year_trades)
            },
            "capital_gains": {
                "total_realized_gain": total_capital_gain,
                "short_term_gain": total_short_term_gain,
                "long_term_gain": total_long_term_gain,
                "total_transactions": len(year_gains),
                "by_asset_type": self._group_gains_by_type(year_gains)
            },
            "expenses": {
                "total_expenses": total_expenses,
                "deductible_expenses": deductible_expenses,
                "by_category": self._group_expenses_by_category(year_expenses),
                "total_deductions": deductible_expenses
            },
            "unrealized_gains": unrealized_gains if include_unrealized else {},
            "tax_estimates": estimated_taxes,
            "north_carolina_tax": {
                "estimated_state_tax": estimated_taxes.get("nc_state_tax", 0),
                "rate": self.nc_tax_rates["individual_income"]["rate"],
                "notes": "NC treats all capital gains as ordinary income"
            },
            "federal_tax": {
                "estimated_federal_tax": estimated_taxes.get("federal_tax", 0),
                "short_term_rate_applied": "Ordinary income rates",
                "long_term_rate_applied": "Preferential capital gains rates"
            },
            "total_estimated_tax": estimated_taxes.get("total_tax", 0),
            "net_after_tax": total_capital_gain - estimated_taxes.get("total_tax", 0)
        }
        
        return summary
    
    def _calculate_unrealized_gains(self) -> Dict[str, Any]:
        """Calculate unrealized gains for current positions"""
        unrealized = {}
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                # Would need current market price - placeholder for now
                current_price = 0.0  # Would fetch from market data
                current_value = position.quantity * current_price
                unrealized_gain = current_value - position.cost_basis
                
                unrealized[symbol] = {
                    "quantity": position.quantity,
                    "cost_basis": position.cost_basis,
                    "current_value": current_value,
                    "unrealized_gain": unrealized_gain,
                    "unrealized_gain_pct": (unrealized_gain / position.cost_basis * 100) if position.cost_basis > 0 else 0
                }
        
        return unrealized
    
    def _calculate_estimated_taxes(
        self,
        short_term_gain: float,
        long_term_gain: float,
        deductions: float
    ) -> Dict[str, Any]:
        """Calculate estimated tax liability"""
        # Net capital gains after deductions
        net_short_term = max(0, short_term_gain - deductions)
        net_long_term = max(0, long_term_gain)
        
        # Federal tax calculation (simplified - would need actual income brackets)
        # Short-term taxed as ordinary income
        federal_short_term_tax = self._calculate_federal_tax(net_short_term, is_long_term=False)
        
        # Long-term taxed at preferential rates
        federal_long_term_tax = self._calculate_federal_tax(net_long_term, is_long_term=True)
        
        federal_tax = federal_short_term_tax + federal_long_term_tax
        
        # North Carolina tax (flat rate on all income)
        nc_taxable_income = net_short_term + net_long_term
        nc_state_tax = nc_taxable_income * self.nc_tax_rates["individual_income"]["rate"]
        
        total_tax = federal_tax + nc_state_tax
        
        return {
            "federal_tax": federal_tax,
            "nc_state_tax": nc_state_tax,
            "total_tax": total_tax,
            "effective_rate": (total_tax / (short_term_gain + long_term_gain) * 100) if (short_term_gain + long_term_gain) > 0 else 0
        }
    
    def _calculate_federal_tax(self, income: float, is_long_term: bool = False) -> float:
        """Calculate federal tax (simplified bracket calculation)"""
        if income <= 0:
            return 0.0
        
        brackets = self.federal_tax_rates["capital_gains"]["long_term"]["brackets"] if is_long_term else self.federal_tax_rates["capital_gains"]["short_term"]["brackets"]
        
        tax = 0.0
        remaining_income = income
        
        for i, (threshold, rate) in enumerate(brackets):
            if i == len(brackets) - 1:
                # Top bracket
                if remaining_income > threshold:
                    tax += (remaining_income - threshold) * rate
                break
            
            next_threshold = brackets[i + 1][0]
            bracket_income = min(remaining_income, next_threshold - threshold)
            
            if bracket_income > 0:
                tax += bracket_income * rate
                remaining_income -= bracket_income
            
            if remaining_income <= 0:
                break
        
        return tax
    
    def _group_trades_by_agent(self, trades: List[TradeRecord]) -> Dict[str, int]:
        """Group trades by agent"""
        agent_counts = defaultdict(int)
        for trade in trades:
            agent_counts[trade.agent] += 1
        return dict(agent_counts)
    
    def _group_gains_by_type(self, gains: List[CapitalGain]) -> Dict[str, float]:
        """Group capital gains by asset type"""
        type_totals = defaultdict(float)
        for gain in gains:
            type_totals[gain.asset_type] += gain.gain_loss
        return dict(type_totals)
    
    def _group_expenses_by_category(self, expenses: List[ExpenseRecord]) -> Dict[str, float]:
        """Group expenses by category"""
        category_totals = defaultdict(float)
        for expense in expenses:
            category_totals[expense.category] += expense.amount
        return dict(category_totals)
    
    def export_turbo_tax_format(self, tax_year: Optional[int] = None) -> Dict[str, Any]:
        """Export data in TurboTax-compatible format"""
        if tax_year is None:
            tax_year = self.current_tax_year
        
        summary = self.get_tax_summary(tax_year)
        
        # Format for TurboTax import
        turbo_tax_data = {
            "tax_year": tax_year,
            "schedule_d": {
                "short_term_transactions": [
                    {
                        "description": f"{g.symbol} - {g.asset_type}",
                        "date_acquired": self._get_acquisition_date(g.symbol),
                        "date_sold": g.sale_date,
                        "cost_basis": g.cost_basis,
                        "sales_price": g.sale_proceeds,
                        "gain_loss": g.short_term_gain
                    }
                    for g in self.capital_gains
                    if datetime.fromisoformat(g.sale_date.replace("Z", "+00:00")).year == tax_year
                    and g.short_term_gain != 0
                ],
                "long_term_transactions": [
                    {
                        "description": f"{g.symbol} - {g.asset_type}",
                        "date_acquired": self._get_acquisition_date(g.symbol),
                        "date_sold": g.sale_date,
                        "cost_basis": g.cost_basis,
                        "sales_price": g.sale_proceeds,
                        "gain_loss": g.long_term_gain
                    }
                    for g in self.capital_gains
                    if datetime.fromisoformat(g.sale_date.replace("Z", "+00:00")).year == tax_year
                    and g.long_term_gain != 0
                ]
            },
            "schedule_c": {
                "business_expenses": [
                    {
                        "category": e.category,
                        "description": e.description,
                        "amount": e.amount,
                        "deductible_amount": e.amount * (e.deductible_pct / 100.0) * (e.business_use_pct / 100.0)
                    }
                    for e in self.expenses
                    if e.tax_year == tax_year
                ]
            },
            "summary": summary
        }
        
        return turbo_tax_data
    
    def _get_acquisition_date(self, symbol: str) -> str:
        """Get acquisition date for a symbol"""
        if symbol in self.positions:
            return self.positions[symbol].first_purchase_date
        return ""
    
    def export_cpa_format(self, tax_year: Optional[int] = None) -> Dict[str, Any]:
        """Export data in CPA-friendly format"""
        if tax_year is None:
            tax_year = self.current_tax_year
        
        summary = self.get_tax_summary(tax_year, include_unrealized=True)
        
        cpa_data = {
            "client_info": {
                "tax_year": tax_year,
                "prepared_by": "NAE Genny Tax Module",
                "prepared_date": datetime.now().isoformat()
            },
            "trading_activity": {
                "total_trades": summary["trades"]["total_trades"],
                "trades_by_agent": summary["trades"]["by_agent"],
                "day_trades": summary["trades"]["day_trades"]
            },
            "capital_gains_losses": {
                "short_term": {
                    "total": summary["capital_gains"]["short_term_gain"],
                    "transactions": len([g for g in self.capital_gains if g.short_term_gain != 0])
                },
                "long_term": {
                    "total": summary["capital_gains"]["long_term_gain"],
                    "transactions": len([g for g in self.capital_gains if g.long_term_gain != 0])
                },
                "by_asset_type": summary["capital_gains"]["by_asset_type"]
            },
            "business_expenses": {
                "total": summary["expenses"]["total_expenses"],
                "deductible": summary["expenses"]["deductible_expenses"],
                "by_category": summary["expenses"]["by_category"]
            },
            "unrealized_gains": summary["unrealized_gains"],
            "tax_estimates": summary["tax_estimates"],
            "north_carolina_compliance": {
                "state_tax_estimate": summary["north_carolina_tax"]["estimated_state_tax"],
                "rate": summary["north_carolina_tax"]["rate"],
                "notes": "NC treats all capital gains as ordinary income - no preferential rate"
            },
            "detailed_transactions": [
                {
                    "trade_id": t.trade_id,
                    "date": t.timestamp,
                    "symbol": t.symbol,
                    "type": t.asset_type,
                    "action": t.action,
                    "quantity": t.quantity,
                    "price": t.price,
                    "cost_basis": t.cost_basis,
                    "proceeds": t.proceeds,
                    "fees": t.fees,
                    "agent": t.agent
                }
                for t in self.trades
                if datetime.fromisoformat(t.timestamp.replace("Z", "+00:00")).year == tax_year
            ]
        }
        
        return cpa_data


if __name__ == "__main__":
    # Test the tax module
    preparer = TaxPreparer()
    
    # Record some test trades
    preparer.record_trade(
        trade_id="test_001",
        timestamp="2024-01-15T10:00:00Z",
        symbol="AAPL",
        asset_type="stock",
        action="buy",
        quantity=10,
        price=150.00,
        fees=1.00,
        agent="optimus"
    )
    
    preparer.record_trade(
        trade_id="test_002",
        timestamp="2024-06-15T10:00:00Z",
        symbol="AAPL",
        asset_type="stock",
        action="sell",
        quantity=10,
        price=175.00,
        fees=1.00,
        agent="optimus"
    )
    
    # Record an expense
    preparer.record_expense(
        expense_id="exp_001",
        timestamp="2024-03-01T00:00:00Z",
        category="software",
        description="Trading platform subscription",
        amount=99.00,
        deductible_pct=100.0,
        business_use_pct=100.0
    )
    
    # Generate tax summary
    summary = preparer.get_tax_summary(2024)
    print("\nTax Summary:")
    print(json.dumps(summary, indent=2, default=str))

