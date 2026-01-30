# NAE/experiments/thrml_sampling_experiment.py
"""
THRML Sampling Prototype Experiment

Compares THRML Gibbs sampling vs conventional Monte Carlo for:
- Market state scenario generation
- Options valuation
- Performance benchmarking
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX not available")

try:
    from tools.thrml_integration import ProbabilisticTradingModel, block_gibbs_sampling
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    print("Warning: THRML integration not available")


@dataclass
class SamplingBenchmark:
    """Sampling benchmark results"""
    method: str
    num_samples: int
    time_seconds: float
    samples_per_second: float
    mixing_time: Optional[float] = None
    effective_sample_size: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None


class THRMLExperiment:
    """
    THRML sampling experiment for market scenario generation
    """
    
    def __init__(self):
        self.results: List[SamplingBenchmark] = []
    
    def run_market_state_sampler_experiment(
        self,
        num_samples: int = 1000,
        num_nodes: int = 10
    ) -> Dict[str, Any]:
        """
        Compare THRML vs Monte Carlo for market state sampling
        
        Returns benchmark results
        """
        print(f"Running market state sampler experiment ({num_samples} samples, {num_nodes} nodes)...")
        
        # Define energy function (market state model)
        def energy_fn(state):
            # Simple Ising-like model: E = -sum(coupling * s_i * s_j) - sum(bias * s_i)
            coupling = 0.5
            bias = 0.1
            pairwise = -coupling * jnp.sum(state[:, None] * state[None, :]) / 2
            unary = -bias * jnp.sum(state)
            return pairwise + unary
        
        initial_state = jnp.ones(num_nodes) * 0.5
        
        results = {}
        
        # THRML Gibbs sampling
        if THRML_AVAILABLE and JAX_AVAILABLE:
            thrml_result = self._benchmark_thrml_sampling(
                energy_fn, initial_state, num_samples
            )
            results['thrml'] = asdict(thrml_result)
            print(f"✅ THRML: {thrml_result.samples_per_second:.0f} samples/sec")
        
        # Conventional Monte Carlo (Metropolis-Hastings)
        mc_result = self._benchmark_mc_sampling(
            energy_fn, initial_state, num_samples
        )
        results['monte_carlo'] = asdict(mc_result)
        print(f"✅ Monte Carlo: {mc_result.samples_per_second:.0f} samples/sec")
        
        # Calculate speedup
        if 'thrml' in results:
            speedup = results['monte_carlo']['samples_per_second'] / results['thrml']['samples_per_second']
            results['speedup'] = speedup
            print(f"📊 Speedup: {speedup:.2f}x")
        
        return results
    
    def _benchmark_thrml_sampling(
        self,
        energy_fn,
        initial_state: jnp.ndarray,
        num_samples: int
    ) -> SamplingBenchmark:
        """Benchmark THRML Gibbs sampling"""
        start_time = time.time()
        
        samples = block_gibbs_sampling(
            energy_fn,
            initial_state,
            num_samples=num_samples,
            warmup=100,
            beta=1.0,
            rng_key=random.PRNGKey(42)
        )
        
        elapsed = time.time() - start_time
        
        # Calculate energy statistics
        energies = [float(energy_fn(s)) for s in samples]
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        
        return SamplingBenchmark(
            method="THRML_Gibbs",
            num_samples=num_samples,
            time_seconds=elapsed,
            samples_per_second=num_samples / elapsed if elapsed > 0 else 0,
            energy_mean=energy_mean,
            energy_std=energy_std
        )
    
    def _benchmark_mc_sampling(
        self,
        energy_fn,
        initial_state: jnp.ndarray,
        num_samples: int
    ) -> SamplingBenchmark:
        """Benchmark conventional Monte Carlo (Metropolis-Hastings)"""
        start_time = time.time()
        
        # Convert to numpy for MC
        state = np.array(initial_state)
        samples = [state.copy()]
        
        for _ in range(num_samples):
            # Propose new state
            proposed = state + np.random.normal(0, 0.1, size=state.shape)
            
            # Metropolis-Hastings
            current_energy = float(energy_fn(jnp.array(state)))
            proposed_energy = float(energy_fn(jnp.array(proposed)))
            
            acceptance = min(1.0, np.exp(-(proposed_energy - current_energy)))
            if np.random.random() < acceptance:
                state = proposed
            
            samples.append(state.copy())
        
        elapsed = time.time() - start_time
        
        # Calculate energy statistics
        energies = [float(energy_fn(jnp.array(s))) for s in samples]
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        
        return SamplingBenchmark(
            method="Monte_Carlo_MH",
            num_samples=num_samples,
            time_seconds=elapsed,
            samples_per_second=num_samples / elapsed if elapsed > 0 else 0,
            energy_mean=energy_mean,
            energy_std=energy_std
        )
    
    def run_options_valuation_experiment(
        self,
        spot_price: float = 100.0,
        strike: float = 100.0,
        time_to_expiry: float = 0.25,
        risk_free_rate: float = 0.05,
        num_paths: int = 10000
    ) -> Dict[str, Any]:
        """
        Compare THRML vs MC for options valuation
        
        Uses sampled market trajectories for Monte Carlo options pricing
        """
        print(f"Running options valuation experiment ({num_paths} paths)...")
        
        # Use THRML to sample market trajectories
        if THRML_AVAILABLE:
            model = ProbabilisticTradingModel(num_nodes=5)
            market_features = ['price', 'volatility', 'trend']
            model.build_market_pgm(market_features=market_features)
            
            # Sample trajectories
            initial_state = jnp.array([spot_price / 100.0, 0.20, 0.0])  # Normalized
            
            start_time = time.time()
            trajectories = model.simulate_market_trajectories(
                initial_state,
                num_trajectories=num_paths,
                horizon=int(time_to_expiry * 252)  # Daily steps
            )
            thrml_time = time.time() - start_time
            
            # Extract final prices
            final_prices = [float(traj[-1][0] * 100.0) for traj in trajectories]
            
            # Calculate option payoff (call)
            payoffs = [max(0, price - strike) for price in final_prices]
            thrml_price = np.mean(payoffs) * np.exp(-risk_free_rate * time_to_expiry)
        else:
            thrml_price = None
            thrml_time = None
        
        # Conventional Monte Carlo
        start_time = time.time()
        mc_prices = []
        for _ in range(num_paths):
            # Geometric Brownian Motion
            dt = time_to_expiry / 252
            price = spot_price
            for _ in range(int(time_to_expiry * 252)):
                price *= np.exp((risk_free_rate - 0.5 * 0.2**2) * dt + 0.2 * np.sqrt(dt) * np.random.normal())
            mc_prices.append(max(0, price - strike))
        
        mc_price = np.mean(mc_prices) * np.exp(-risk_free_rate * time_to_expiry)
        mc_time = time.time() - start_time
        
        results = {
            "monte_carlo": {
                "option_price": float(mc_price),
                "time_seconds": mc_time,
                "paths_per_second": num_paths / mc_time if mc_time > 0 else 0
            }
        }
        
        if thrml_price is not None:
            results["thrml"] = {
                "option_price": float(thrml_price),
                "time_seconds": thrml_time,
                "paths_per_second": num_paths / thrml_time if thrml_time > 0 else 0
            }
            results["price_difference"] = abs(thrml_price - mc_price)
            results["price_difference_pct"] = abs(thrml_price - mc_price) / mc_price * 100
        
        print(f"✅ Monte Carlo price: ${mc_price:.4f} ({mc_time:.2f}s)")
        if thrml_price:
            print(f"✅ THRML price: ${thrml_price:.4f} ({thrml_time:.2f}s)")
            print(f"📊 Price difference: {results.get('price_difference_pct', 0):.2f}%")
        
        return results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save experiment results"""
        output_dir = "experiments/results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filepath}")
        return filepath


if __name__ == "__main__":
    experiment = THRMLExperiment()
    
    # Run market state sampler experiment
    print("\n" + "="*60)
    print("Market State Sampler Experiment")
    print("="*60)
    sampler_results = experiment.run_market_state_sampler_experiment(num_samples=1000)
    experiment.save_results(sampler_results, "market_state_sampler")
    
    # Run options valuation experiment
    print("\n" + "="*60)
    print("Options Valuation Experiment")
    print("="*60)
    options_results = experiment.run_options_valuation_experiment(num_paths=5000)
    experiment.save_results(options_results, "options_valuation")

