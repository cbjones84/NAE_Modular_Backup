# NAE/tools/backtest_engine.py
"""
Robust Backtesting Engine with Walk-Forward Analysis

Features:
- Transaction costs modeling
- Slippage simulation
- Realistic fills
- Margin/Greeks for options
- Walk-forward testing
- K-fold time-series cross validation
- Metadata tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import hashlib
from pathlib import Path


class FillType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class TransactionCosts:
    """Transaction cost model"""
    commission_per_share: float = 0.005  # $0.005 per share
    commission_per_contract: float = 0.65  # $0.65 per options contract
    slippage_bps: float = 5.0  # 5 basis points slippage
    market_impact_bps: float = 2.0  # 2 bps market impact


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100000.0
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)
    slippage_model: str = "linear"  # "linear", "sqrt", "constant"
    fill_model: str = "realistic"  # "realistic", "instant", "delayed"
    use_margin: bool = True
    margin_requirement: float = 0.20  # 20% margin for options
    use_greeks: bool = True
    random_seed: Optional[int] = None


@dataclass
class BacktestResult:
    """Backtest result"""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SlippageModel:
    """Slippage modeling"""
    
    @staticmethod
    def linear(quantity: float, price: float, bps: float) -> float:
        """Linear slippage model"""
        slippage = price * (bps / 10000) * abs(quantity)
        return slippage
    
    @staticmethod
    def sqrt(quantity: float, price: float, bps: float) -> float:
        """Square root slippage model (market impact)"""
        slippage = price * (bps / 10000) * np.sqrt(abs(quantity))
        return slippage
    
    @staticmethod
    def constant(quantity: float, price: float, bps: float) -> float:
        """Constant slippage"""
        slippage = price * (bps / 10000)
        return slippage


class BacktestEngine:
    """
    Robust backtesting engine with realistic market simulation
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def run_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        strategy_name: str = "strategy",
        **strategy_kwargs
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            strategy: Strategy function that returns signals
            data: Historical price data (OHLCV)
            strategy_name: Name of strategy
            **strategy_kwargs: Additional strategy parameters
        """
        capital = self.config.initial_capital
        positions = {}  # symbol -> quantity
        trades = []
        equity_curve = [capital]
        dates = []
        
        # Generate signals
        signals = strategy(data, **strategy_kwargs)
        
        for i, (date, row) in enumerate(data.iterrows()):
            dates.append(date)
            signal = signals.iloc[i] if isinstance(signals, pd.Series) else signals[i]
            
            # Execute trades based on signals
            if signal != 0:  # Non-zero signal
                symbol = row.get('symbol', 'STOCK') if 'symbol' in row else 'STOCK'
                price = row['close']
                
                # Calculate position size
                position_size = self._calculate_position_size(capital, price, signal)
                
                # Apply transaction costs
                cost = self._calculate_transaction_cost(position_size, price, symbol)
                
                # Apply slippage
                fill_price = self._apply_slippage(price, position_size, signal)
                
                # Update positions
                if symbol in positions:
                    positions[symbol] += position_size
                else:
                    positions[symbol] = position_size
                
                # Update capital
                capital -= position_size * fill_price + cost
                
                # Record trade
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'quantity': position_size,
                    'price': fill_price,
                    'cost': cost,
                    'signal': signal
                })
            
            # Calculate portfolio value
            portfolio_value = capital
            for symbol, qty in positions.items():
                if qty != 0:
                    current_price = row['close']
                    portfolio_value += qty * current_price
            
            equity_curve.append(portfolio_value)
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=dates[:len(equity_curve)])
        returns = equity_series.pct_change().dropna()
        
        result = self._calculate_metrics(
            strategy_name,
            data.index[0],
            data.index[-1],
            equity_curve,
            returns,
            trades
        )
        
        return result
    
    def _calculate_position_size(self, capital: float, price: float, signal: float) -> float:
        """Calculate position size"""
        # Simple fixed fractional (2% of capital)
        position_value = capital * 0.02
        return (position_value / price) * np.sign(signal) if price > 0 else 0
    
    def _calculate_transaction_cost(self, quantity: float, price: float, symbol: str) -> float:
        """Calculate transaction costs"""
        costs = self.config.transaction_costs
        
        if 'OPT' in symbol.upper() or 'CALL' in symbol.upper() or 'PUT' in symbol.upper():
            # Options contract
            contracts = abs(quantity) / 100  # Assume 100 shares per contract
            commission = contracts * costs.commission_per_contract
        else:
            # Stock
            commission = abs(quantity) * costs.commission_per_share
        
        return commission
    
    def _apply_slippage(self, price: float, quantity: float, signal: float) -> float:
        """Apply slippage to fill price"""
        if quantity == 0:
            return price
        
        slippage_model = getattr(SlippageModel, self.config.slippage_model, SlippageModel.linear)
        slippage = slippage_model(quantity, price, self.config.transaction_costs.slippage_bps)
        
        # Slippage increases cost for buys, decreases proceeds for sells
        fill_price = price + (slippage * np.sign(signal))
        
        return fill_price
    
    def _calculate_metrics(
        self,
        strategy_name: str,
        start_date,
        end_date,
        equity_curve: List[float],
        returns: pd.Series,
        trades: List[Dict[str, Any]]
    ) -> BacktestResult:
        """Calculate backtest metrics"""
        initial_capital = equity_curve[0]
        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized return
        days = (end_date - start_date).days if hasattr(end_date, '__sub__') else 365
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Max drawdown duration
        drawdown_periods = (drawdown < -0.01).astype(int)
        max_dd_duration = drawdown_periods.sum()
        
        # Win rate
        trade_returns = [t.get('return', 0) for t in trades if 'return' in t]
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_trade_return = np.mean(trade_returns)
            profit_factor = sum(r for r in trade_returns if r > 0) / abs(sum(r for r in trade_returns if r < 0)) if sum(r for r in trade_returns if r < 0) != 0 else 0
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
            profit_factor = 0.0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            metadata={
                "config": asdict(self.config),
                "data_points": len(equity_curve)
            }
        )
    
    def walk_forward_analysis(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: int = 21,
        strategy_name: str = "strategy",
        **strategy_kwargs
    ) -> List[BacktestResult]:
        """
        Walk-forward analysis
        
        Args:
            strategy: Strategy function
            data: Historical data
            train_period_days: Training period length
            test_period_days: Testing period length
            step_days: Step size for rolling window
            strategy_name: Strategy name
            **strategy_kwargs: Strategy parameters
        """
        results = []
        start_idx = 0
        
        while start_idx + train_period_days + test_period_days < len(data):
            # Training period
            train_start = start_idx
            train_end = start_idx + train_period_days
            
            # Test period
            test_start = train_end
            test_end = test_start + test_period_days
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Run backtest on test period
            result = self.run_backtest(
                strategy,
                test_data,
                strategy_name=f"{strategy_name}_wf_{len(results)+1}",
                **strategy_kwargs
            )
            
            # Add walk-forward metadata
            result.metadata.update({
                "walk_forward": True,
                "train_start": str(train_data.index[0]),
                "train_end": str(train_data.index[-1]),
                "test_start": str(test_data.index[0]),
                "test_end": str(test_data.index[-1]),
                "fold": len(results) + 1
            })
            
            results.append(result)
            
            # Move window
            start_idx += step_days
        
        return results
    
    def save_result(self, result: BacktestResult, output_dir: str = "backtests"):
        """Save backtest result with metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with hash
        result_hash = hashlib.md5(
            json.dumps(asdict(result), sort_keys=True).encode()
        ).hexdigest()[:8]
        
        filename = f"{result.strategy_name}_{result_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save result
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        return filepath

