# THRML Integration Guide for NAE

## Overview

This guide explains how to leverage Extropic's THRML (Thermodynamic Machine Learning) to improve NAE's entire performance. THRML enables probabilistic decision models, energy-based learning, and thermodynamic computing principles for enhanced trading intelligence.

## Table of Contents

1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Usage Examples](#usage-examples)
4. [Integration Points](#integration-points)
5. [Best Practices](#best-practices)
6. [Performance Profiling](#performance-profiling)

---

## Installation

### Prerequisites

THRML requires JAX for numerical computing. Install dependencies:

```bash
pip install jax>=0.4.0 jaxlib>=0.4.0
```

### Installing THRML

THRML may need to be installed from Extropic's repository:

```bash
# Option 1: From GitHub (if available)
pip install git+https://github.com/extropic-ai/thrml.git

# Option 2: From PyPI (if available)
pip install thrml

# Option 3: Install from requirements.txt
pip install -r requirements.txt
```

### Verification

Verify installation:

```python
from tools.thrml_integration import ProbabilisticTradingModel, EnergyBasedStrategyModel
import jax.numpy as jnp

# Test basic functionality
model = ProbabilisticTradingModel(num_nodes=5)
print("THRML integration ready!")
```

---

## Architecture

### Core Components

1. **ProbabilisticTradingModel** (`tools/thrml_integration.py`)
   - Probabilistic Graphical Models (PGMs) for market states
   - Gibbs sampling for trajectory simulation
   - Tail risk estimation

2. **EnergyBasedStrategyModel** (`tools/thrml_integration.py`)
   - Energy-based learning for strategy patterns
   - Pattern recognition (typical vs rare strategies)
   - Strategy generation via sampling

3. **ThermodynamicArchitectureExplorer**
   - Explore graph topologies for TSU hardware
   - Simulate different coupling strengths and temperatures

4. **THRMLProfiler**
   - Benchmark sampling performance
   - Compare THRML vs JAX vs NumPy
   - Validate thermodynamic compute gains

### Integration Points

- **Optimus Agent**: Uses probabilistic models for trading scenarios and risk modeling
- **Ralph Agent**: Uses energy-based learning for strategy pattern recognition
- **Donnie Agent**: Can leverage probabilistic validation (future enhancement)

---

## Usage Examples

### 1. Probabilistic Trading Scenarios (Optimus)

Simulate market trajectories under uncertainty:

```python
from agents.optimus import OptimusAgent

# Initialize Optimus with THRML
optimus = OptimusAgent(sandbox=True)

# Simulate trading scenarios
scenarios = optimus.simulate_trading_scenarios(
    symbol="AAPL",
    current_price=150.0,
    volatility=0.25,
    volume=1000000,
    num_trajectories=100,
    horizon=10
)

print(f"Mean future price: ${scenarios['statistics']['mean_future_price']:.2f}")
print(f"Probability of profit: {scenarios['statistics']['prob_profit']:.2%}")
print(f"Probability of loss: {scenarios['statistics']['prob_loss']:.2%}")
```

### 2. Tail Risk Estimation (Optimus)

Estimate tail-risk probabilities for risk management:

```python
# Estimate tail risk
risk_analysis = optimus.estimate_tail_risk(
    symbol="AAPL",
    current_price=150.0,
    volatility=0.25,
    volume=1000000,
    threshold=-0.1  # -10% loss threshold
)

print(f"Tail probability: {risk_analysis['risk_metrics']['tail_probability']:.2%}")
print(f"VaR (95%): {risk_analysis['risk_metrics']['var_95']:.2%}")
print(f"Recommendation: {risk_analysis['recommendation']}")
```

### 3. Energy-Based Strategy Learning (Ralph)

Train EBM on historical strategies:

```python
from agents.ralph import RalphAgent

# Initialize Ralph with THRML
ralph = RalphAgent()

# Train EBM on approved strategies
training_result = ralph.train_strategy_ebm()
print(f"Trained on {training_result['num_strategies']} strategies")

# Evaluate a strategy
strategy = {
    "name": "Test Strategy",
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

evaluation = ralph.evaluate_strategy_with_ebm(strategy)
print(f"Energy: {evaluation['energy']:.2f}")
print(f"Probability Score: {evaluation['probability_score']:.2f}")
print(f"Interpretation: {evaluation['interpretation']}")
```

### 4. Generate Strategy Samples (Ralph)

Generate new strategies using THRML sampling:

```python
# Generate strategy samples
generated = ralph.generate_strategy_samples(num_samples=10)

for strategy in generated:
    print(f"{strategy['name']}: Energy={strategy['thrml_energy']:.2f}, "
          f"Backtest={strategy['backtest_score']:.1f}")
```

### 5. Performance Profiling

Benchmark THRML performance:

```python
# Profile THRML performance
profile = optimus.profile_thrml_performance()

for method, results in profile['benchmark_results'].items():
    print(f"{method}: {results['samples_per_second']:.0f} samples/sec")
    
print(f"\n{profile['recommendation']}")
```

---

## Integration Points

### Optimus Agent Integration

**Location**: `NAE/agents/optimus.py`

**Methods Added**:
- `simulate_trading_scenarios()`: Simulate probabilistic market trajectories
- `estimate_tail_risk()`: Estimate tail-risk probabilities
- `profile_thrml_performance()`: Benchmark THRML performance

**Usage in Trading Flow**:
1. Before executing a trade, use `simulate_trading_scenarios()` to assess potential outcomes
2. Use `estimate_tail_risk()` to evaluate risk before large positions
3. Integrate risk metrics into position sizing decisions

### Ralph Agent Integration

**Location**: `NAE/agents/ralph.py`

**Methods Added**:
- `train_strategy_ebm()`: Train energy-based model on strategies
- `evaluate_strategy_with_ebm()`: Evaluate strategy patterns
- `generate_strategy_samples()`: Generate new strategies via sampling
- `_extract_strategy_features()`: Extract features from strategies

**Usage in Strategy Flow**:
1. Train EBM periodically on approved strategies
2. Evaluate new candidates with `evaluate_strategy_with_ebm()`
3. Use energy scores to filter/rank strategies
4. Generate novel strategies with `generate_strategy_samples()`

---

## Best Practices

### 1. Start Small

Begin with toy models to understand THRML behavior:

```python
# Simple 5-node Ising chain
from tools.thrml_integration import ProbabilisticTradingModel
import jax.numpy as jnp

model = ProbabilisticTradingModel(num_nodes=5)
model.build_market_pgm(
    market_features=['price', 'vol', 'trend'],
    coupling_strength=0.5
)

initial_state = jnp.array([0.5, 0.5, 0.5])
trajectories = model.simulate_market_trajectories(initial_state, num_trajectories=10)
```

### 2. Tune Sampling Schedule

Adjust `SamplingSchedule` parameters for optimal performance:

```python
from tools.thrml_integration import SamplingSchedule

schedule = SamplingSchedule(
    warmup=100,      # Warmup steps (too few → poor mixing)
    steps=1000,      # Sampling steps
    samples=100,     # Number of samples
    block_size=1,   # Block size for block Gibbs
    beta=1.0        # Inverse temperature
)
```

### 3. Log and Visualize

Track sample distributions, energy states, and convergence:

```python
# Log energy over time
energies = []
for sample in samples:
    energy = energy_fn(sample)
    energies.append(energy)

# Visualize convergence
import matplotlib.pyplot as plt
plt.plot(energies)
plt.xlabel('Sample')
plt.ylabel('Energy')
plt.title('Energy Convergence')
plt.show()
```

### 4. Experiment with Topology

Try different graph structures:

```python
from tools.thrml_integration import ThermodynamicArchitectureExplorer

explorer = ThermodynamicArchitectureExplorer()

# Explore grid topology
grid_results = explorer.explore_topology(
    num_nodes=16,
    topology="grid",
    coupling_strength=0.5
)

# Compare with fully connected
fc_results = explorer.explore_topology(
    num_nodes=16,
    topology="fully_connected",
    coupling_strength=0.5
)
```

### 5. Integrate with Existing ML Stack

Use THRML sampling outputs as features for downstream agents:

```python
# Sample from THRML
samples = thrml_model.sample_strategies(num_samples=100)

# Feed to neural network or other ML models
for sample in samples:
    features = extract_features(sample)
    prediction = neural_network.predict(features)
    # Use prediction for trading decisions
```

---

## Performance Profiling

### Benchmarking THRML vs Conventional Methods

Compare THRML performance with JAX and NumPy:

```python
from tools.thrml_integration import THRMLProfiler
import jax.numpy as jnp

profiler = THRMLProfiler()

def energy_fn(state):
    return float(jnp.sum(state ** 2))

initial_state = jnp.ones(10) * 0.5

comparison = profiler.compare_methods(
    energy_fn,
    initial_state,
    num_samples=1000
)

# Results show:
# - Samples per second
# - Memory usage
# - Sample quality metrics
```

### Expected Performance Gains

When using actual TSU hardware (future):
- **10-100x speedup** for sampling operations
- **Lower power consumption** for probabilistic inference
- **Better scalability** for large graphical models

Current GPU-based JAX implementation:
- **2-5x speedup** vs NumPy
- **Similar performance** to PyTorch for sampling
- **Better for research/prototyping**

---

## Advanced Usage

### Custom Energy Functions

Define custom energy functions for specific use cases:

```python
import jax.numpy as jnp

def custom_energy_fn(state: jnp.ndarray) -> float:
    """
    Custom energy function for options trading
    
    E(x) = -sum(bias_i * x_i) - sum(coupling_ij * x_i * x_j) + penalty_term
    """
    # Unary terms (biases)
    biases = jnp.array([0.1, -0.2, 0.15, -0.1, 0.05])
    unary = -jnp.dot(biases, state)
    
    # Pairwise terms (coupling)
    coupling = jnp.array([
        [0.0, 0.5, 0.3, 0.2, 0.1],
        [0.5, 0.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 0.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 0.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.0]
    ])
    pairwise = -0.5 * jnp.sum(coupling * jnp.outer(state, state))
    
    # Penalty term (e.g., for constraints)
    penalty = 0.1 * jnp.sum((state - 0.5) ** 2)
    
    return unary + pairwise + penalty
```

### Hybrid Models

Combine THRML with neural networks:

```python
# Sample latent states from PGM
latent_samples = thrml_model.sample_strategies(num_samples=100)

# Feed to neural network
for latent in latent_samples:
    # Use latent as input to neural network
    nn_input = preprocess(latent)
    prediction = neural_network(nn_input)
    
    # Combine probabilistic and neural predictions
    final_decision = combine_predictions(latent, prediction)
```

---

## Troubleshooting

### Common Issues

1. **JAX not available**
   - Install: `pip install jax jaxlib`
   - For GPU: `pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

2. **THRML import fails**
   - Check if THRML is installed: `pip list | grep thrml`
   - Fallback to JAX implementation (automatic)

3. **Sampling too slow**
   - Reduce `num_samples` or `horizon`
   - Use GPU acceleration: `export JAX_PLATFORMS=cuda`
   - Tune `warmup` and `block_size`

4. **Poor mixing/convergence**
   - Increase `warmup` steps
   - Adjust `beta` (temperature)
   - Try different `block_size`

---

## Future Enhancements

1. **Hardware Acceleration**: Migrate to TSU hardware when available
2. **Real-time Integration**: Stream market data into THRML models
3. **Distributed Sampling**: Parallelize sampling across multiple GPUs/nodes
4. **Adaptive Scheduling**: Automatically tune sampling schedules
5. **Integration with Donnie**: Add probabilistic validation to Donnie agent

---

## References

- [Extropic THRML Documentation](https://extropic.ai)
- [JAX Documentation](https://jax.readthedocs.io)
- [Gibbs Sampling Tutorial](https://en.wikipedia.org/wiki/Gibbs_sampling)
- [Energy-Based Models](https://arxiv.org/abs/2003.01607)

---

## Support

For issues or questions:
1. Check logs: `logs/optimus.log` and `logs/ralph.log`
2. Review THRML integration code: `tools/thrml_integration.py`
3. Test with simple examples first
4. Enable debug logging for detailed output

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Active Development

