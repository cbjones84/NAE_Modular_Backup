# THRML Quick Start Guide

## Quick Installation

```bash
# Install dependencies
pip install jax>=0.4.0 jaxlib>=0.4.0

# Install THRML (if available)
pip install thrml
# OR: pip install git+https://github.com/extropic-ai/thrml.git
```

## Quick Usage Examples

### 1. Optimus - Probabilistic Trading Scenarios

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=True)

# Simulate market scenarios
scenarios = optimus.simulate_trading_scenarios(
    symbol="AAPL",
    current_price=150.0,
    volatility=0.25,
    volume=1000000,
    num_trajectories=100
)

print(f"Prob profit: {scenarios['statistics']['prob_profit']:.2%}")

# Estimate tail risk
risk = optimus.estimate_tail_risk(
    symbol="AAPL",
    current_price=150.0,
    volatility=0.25,
    volume=1000000
)
print(f"Tail risk: {risk['risk_metrics']['tail_probability']:.2%}")
```

### 2. Ralph - Energy-Based Strategy Learning

```python
from agents.ralph import RalphAgent

ralph = RalphAgent()

# Train on strategies
ralph.train_strategy_ebm()

# Evaluate a strategy
strategy = {
    "name": "Test",
    "backtest_score": 75.0,
    "trust_score": 80.0,
    "sharpe": 1.5,
    "win_rate": 0.65,
    "max_drawdown": 0.15,
    "perf": 0.20,
    "volatility": 0.18,
    "total_trades": 150,
    "consensus_count": 3
}

eval_result = ralph.evaluate_strategy_with_ebm(strategy)
print(f"Energy: {eval_result['energy']:.2f}")
print(f"Interpretation: {eval_result['interpretation']}")

# Generate new strategies
generated = ralph.generate_strategy_samples(num_samples=5)
```

## Key Features

✅ **Probabilistic Decision Models** - Simulate trading scenarios under uncertainty  
✅ **Energy-Based Learning** - Learn strategy patterns from historical data  
✅ **Tail Risk Estimation** - Model risk states and estimate tail probabilities  
✅ **Strategy Generation** - Generate new strategies via sampling  
✅ **Performance Profiling** - Benchmark THRML vs conventional methods  

## Integration Status

- ✅ **Optimus**: Probabilistic trading scenarios & risk modeling
- ✅ **Ralph**: Energy-based strategy pattern recognition
- ⏳ **Donnie**: Probabilistic validation (future enhancement)

## Documentation

See `docs/THRML_INTEGRATION_GUIDE.md` for comprehensive documentation.

## Troubleshooting

If THRML is not available, NAE will automatically fall back to JAX-based implementations. Check logs for details:

```bash
tail -f logs/optimus.log
tail -f logs/ralph.log
```

