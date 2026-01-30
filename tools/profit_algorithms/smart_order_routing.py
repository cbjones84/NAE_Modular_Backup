# NAE/tools/profit_algorithms/smart_order_routing.py
"""
Smart Order Routing (SOR): Routes orders to best execution venue
Analyzes multiple trading venues to optimize execution price, liquidity, and speed
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from dataclasses import dataclass

@dataclass
class ExecutionVenue:
    """Represents a trading venue with execution metrics"""
    name: str
    base_url: str
    spread: float  # Bid-ask spread
    liquidity_score: float  # 0.0 to 1.0
    latency_ms: float  # Execution latency
    fee_pct: float  # Trading fee percentage
    min_order_size: float
    max_order_size: float
    available: bool

class SmartOrderRouter:
    """
    Smart Order Router - selects best execution venue
    Enhanced with RL-based execution optimization
    """
    
    def __init__(self):
        self.venues = {}
        self.execution_history = []
        
        # RL-based execution optimizer
        try:
            from .rl_execution_optimizer import RLExecutionOptimizer
            self.rl_optimizer = RLExecutionOptimizer()
        except ImportError:
            self.rl_optimizer = None
        
    def register_venue(self, venue: ExecutionVenue):
        """Register a trading venue"""
        self.venues[venue.name] = venue
    
    def _calculate_execution_score(self, venue: ExecutionVenue, 
                                   order_data: Dict[str, Any]) -> float:
        """
        Calculate execution quality score for a venue
        Higher score = better execution
        """
        if not venue.available:
            return 0.0
        
        score = 1.0
        
        # Penalize high spreads (worse execution price)
        spread_penalty = venue.spread * 10.0  # Penalty multiplier
        score -= min(spread_penalty, 0.3)
        
        # Reward high liquidity
        score += venue.liquidity_score * 0.2
        
        # Penalize high latency
        latency_penalty = min(venue.latency_ms / 1000.0, 0.2)  # Penalty for >1s latency
        score -= latency_penalty
        
        # Penalize fees
        fee_penalty = venue.fee_pct * 2.0
        score -= min(fee_penalty, 0.2)
        
        # Check if order size is within venue limits
        order_size = order_data.get('quantity', 0) * order_data.get('price', 0)
        if order_size < venue.min_order_size or order_size > venue.max_order_size:
            return 0.0
        
        return max(0.0, min(1.0, score))
    
    def select_best_venue(self, order_data: Dict[str, Any]) -> Optional[ExecutionVenue]:
        """
        Select best execution venue for an order
        Returns venue with highest execution score
        """
        if not self.venues:
            return None
        
        best_venue = None
        best_score = 0.0
        
        for venue in self.venues.values():
            score = self._calculate_execution_score(venue, order_data)
            if score > best_score:
                best_score = score
                best_venue = venue
        
        return best_venue
    
    def route_order(self, order_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route order to best execution venue
        Enhanced with RL-based execution optimization
        
        Args:
            order_data: Order details
            market_data: Optional market microstructure data for RL optimization
            
        Returns:
            Order data with selected venue and execution strategy
        """
        # Use RL optimizer if available
        execution_decision = None
        if self.rl_optimizer and market_data:
            try:
                execution_decision = self.rl_optimizer.optimize_execution(order_data, market_data)
                # Override venue selection with RL decision
                if execution_decision:
                    order_data['execution_strategy'] = {
                        'urgency': execution_decision.urgency,
                        'order_pacing': execution_decision.order_pacing,
                        'order_type': execution_decision.order_type,
                        'limit_price_offset': execution_decision.limit_price_offset,
                        'expected_slippage': execution_decision.expected_slippage
                    }
            except Exception:
                pass  # Fall back to standard routing
        
        best_venue = self.select_best_venue(order_data)
        
        if not best_venue:
            return {
                "status": "error",
                "error": "No suitable execution venue available",
                "order_data": order_data
            }
        
        # Add routing information to order
        routed_order = order_data.copy()
        routed_order["execution_venue"] = best_venue.name
        routed_order["routing_timestamp"] = time.time()
        routed_order["estimated_fee"] = order_data.get('quantity', 0) * order_data.get('price', 0) * best_venue.fee_pct
        
        # Log routing decision
        self.execution_history.append({
            "timestamp": time.time(),
            "order_id": order_data.get('order_id', 'unknown'),
            "venue": best_venue.name,
            "score": self._calculate_execution_score(best_venue, order_data)
        })
        
        return {
            "status": "routed",
            "venue": best_venue.name,
            "order_data": routed_order,
            "execution_score": self._calculate_execution_score(best_venue, order_data)
        }

# Pre-configured venues for common brokers
def create_default_venues() -> SmartOrderRouter:
    """Create SmartOrderRouter with default broker venues"""
    router = SmartOrderRouter()
    
    # E*TRADE (Sandbox/Production)
    router.register_venue(ExecutionVenue(
        name="etrade_sandbox",
        base_url="https://apisb.etrade.com",
        spread=0.001,  # 0.1% spread
        liquidity_score=0.8,
        latency_ms=150.0,
        fee_pct=0.0,  # $0 commission
        min_order_size=1.0,
        max_order_size=1000000.0,
        available=True
    ))
    
    router.register_venue(ExecutionVenue(
        name="etrade_prod",
        base_url="https://api.etrade.com",
        spread=0.001,
        liquidity_score=0.9,
        latency_ms=120.0,
        fee_pct=0.0,
        min_order_size=1.0,
        max_order_size=1000000.0,
        available=True
    ))
    
    # Alpaca (Paper/Live)
    router.register_venue(ExecutionVenue(
        name="alpaca_paper",
        base_url="https://paper-api.alpaca.markets",
        spread=0.0005,  # Tighter spread
        liquidity_score=0.85,
        latency_ms=100.0,
        fee_pct=0.0,
        min_order_size=1.0,
        max_order_size=1000000.0,
        available=True
    ))
    
    router.register_venue(ExecutionVenue(
        name="alpaca_live",
        base_url="https://api.alpaca.markets",
        spread=0.0005,
        liquidity_score=0.9,
        latency_ms=80.0,  # Faster
        fee_pct=0.0,
        min_order_size=1.0,
        max_order_size=1000000.0,
        available=True
    ))
    
    # Interactive Brokers (Paper/Live)
    router.register_venue(ExecutionVenue(
        name="ibkr_paper",
        base_url="https://api.ibkr.com/v1/paper",
        spread=0.0008,
        liquidity_score=0.95,  # High liquidity
        latency_ms=90.0,
        fee_pct=0.001,  # 0.1% fee
        min_order_size=1.0,
        max_order_size=1000000.0,
        available=True
    ))
    
    return router


