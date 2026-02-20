# Changelog

All notable changes to the Neural Agency Engine (NAE) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### [2026-01-26] - Automated Update


### [2026-01-22] - Automated Update


### [2026-01-16] - Automated Update


### [2026-01-16] - Automated Update


### [2026-01-16] - Automated Update


### [2026-01-16] - Automated Update


### [2026-01-09] - Automated Update


### [2026-01-09] - Automated Update


### [2026-01-09] - Automated Update


### [2026-01-09] - Automated Update


### [2026-01-08] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2026-01-06] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-16] - Automated Update


### [2025-12-12] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-09] - Automated Update


### [2025-12-02] - Automated Update


### [2025-12-02] - Automated Update


### [2025-12-02] - Automated Update


### [2025-12-02] - Automated Update


### [2025-12-02] - Automated Update


### [2025-12-02] - Automated Update


### Added
- Micro-Scalp Accelerator Strategy for rapid account growth ($100 → $8000-$10000)
- Dual-mode operation (sandbox testing + live production simultaneously)
- Settlement cash tracking system to prevent free-riding violations
- Advanced risk management features (volatility filters, spread-aware exits, Kelly criterion)
- Ralph signal integration (`get_intraday_direction_probability`)
- Session-based retraining hooks for continuous learning
- Accelerator controller for managing dual-mode operation
- Automated deployment script (`deploy_accelerator.sh`)
- Comprehensive documentation (ACCELERATOR_STRATEGY.md, DEPLOYMENT_CHECKLIST.md)

### Changed
- Updated accelerator target account size from $500-$1000 to $8000-$10000
- Enhanced Tradier adapter with settled cash tracking methods
- Integrated accelerator into NAE master controller

### Fixed
- Settlement cash tracking to prevent free-riding violations
- Path resolution in accelerator controller

## [2025-01-XX] - Accelerator Strategy Deployment

### Added
- **Settlement Ledger** (`tools/settlement_utils.py`)
  - Tracks settled vs unsettled cash
  - Prevents free-riding violations
  - T+1 settlement for options, T+2 for stocks
  
- **Advanced Micro-Scalp Accelerator** (`tools/profit_algorithms/advanced_micro_scalp.py`)
  - SPY 0DTE options scalping strategy
  - Volatility filters (IV percentile, ATR)
  - Time-of-day filters (9:45-10:30 AM, 1:00-3:30 PM)
  - Spread-aware exits
  - Kelly criterion position sizing
  - Risk-of-ruin calculations
  - 4.3% weekly returns target

- **Accelerator Controller** (`execution/integration/accelerator_controller.py`)
  - Dual-mode operation (sandbox + live)
  - Performance tracking for both environments
  - Automated cycle execution

- **Ralph Signal Methods** (`agents/ralph.py`)
  - `get_intraday_direction_probability()` - Directional probability signals
  - `retrain_hook()` - Session-based learning

- **Optimus Integration** (`agents/optimus.py`)
  - `enable_accelerator_mode()` - Enable accelerator
  - `disable_accelerator_mode()` - Disable accelerator
  - `run_accelerator_cycle()` - Execute cycle

- **Tradier Adapter Enhancements** (`execution/broker_adapters/tradier_adapter.py`)
  - `get_account_balance()` - Get total balance
  - `get_buying_power()` - Get buying power
  - `get_settled_cash()` - Get settled cash
  - `get_unsettled_cash()` - Get unsettled cash

### Changed
- Accelerator target account size: $500 → $8000-$10000
- NAE master controller now includes accelerator process

### Documentation
- Created `docs/ACCELERATOR_STRATEGY.md` - Comprehensive strategy guide
- Created `DEPLOYMENT_CHECKLIST.md` - Deployment verification steps
- Created `ACCELERATOR_DEPLOYMENT_SUMMARY.md` - Quick reference
- Created `CHANGELOG.md` - This file

## [2025-01-15] - Accelerator Strategy Deployment

### Added
- Micro-Scalp Accelerator Strategy (`tools/profit_algorithms/advanced_micro_scalp.py`)
  - SPY 0DTE options scalping for rapid account growth
  - Target: $8000-$10000 account growth
  - 4.3% weekly returns target
- Settlement Ledger (`tools/settlement_utils.py`)
  - Tracks settled vs unsettled cash
  - Prevents free-riding violations
- Accelerator Controller (`execution/integration/accelerator_controller.py`)
  - Dual-mode operation (sandbox + live)
  - Performance tracking
- Ralph Signal Integration (`agents/ralph.py`)
  - `get_intraday_direction_probability()` method
  - `retrain_hook()` for session-based learning
- Tradier Adapter Enhancements (`execution/broker_adapters/tradier_adapter.py`)
  - Settled cash tracking methods
  - Account balance methods
- Deployment Automation (`scripts/deploy_accelerator.sh`)
  - Automated GitHub push and deployment
  - Dual-mode startup
- CHANGELOG.md tracking system
  - Comprehensive change history
  - Automated updates during deployment

### Changed
- Accelerator target: $500-$1000 → $8000-$10000
- NAE master controller includes accelerator process
- Deployment process now maintains changelog

## [2024-12-XX] - Production Environment & Robustness

### Added
- Tradier broker adapter integration (`execution/broker_adapters/tradier_adapter.py`)
- Self-healing diagnostic engine (`execution/self_healing/`)
- Excellence protocols (`agents/*_excellence_protocol.py`)
  - Optimus Excellence Protocol
  - Ralph Excellence Protocol
  - Donnie Excellence Protocol
- Genius communication protocol (`agents/genius_communication_protocol.py`)
- Genny tax optimization system (`agents/genny_tax_*.py`)
- THRML probabilistic trading models (`tools/thrml_integration.py`)
- Robustness systems
  - Metrics collector (`tools/metrics_collector.py`)
  - Risk controls (`tools/risk_controls.py`)
  - Decision ledger (`tools/decision_ledger.py`)
  - Ensemble framework (`tools/ensemble_framework.py`)
  - Regime detection (`tools/regime_detection.py`)
- Online learning framework (`tools/online_learning.py`)
- Enhanced RL trading agents (`tools/profit_algorithms/enhanced_rl_agent.py`)
- IV surface modeling (`tools/profit_algorithms/iv_surface_model.py`)
- Volatility ensemble forecasting (`tools/profit_algorithms/volatility_ensemble.py`)
- Dispersion engine (`tools/profit_algorithms/dispersion_engine.py`)
- Hedging optimizer (`tools/profit_algorithms/hedging_optimizer.py`)
- Execution cost modeling (`tools/profit_algorithms/execution_costs.py`)
- Smart order routing (`tools/profit_algorithms/smart_order_routing.py`)
- Timing strategy engine (`tools/profit_algorithms/timing_strategies.py`)
- Kelly criterion position sizing (`tools/profit_algorithms/kelly_criterion.py`)
- Feedback loops (`tools/feedback_loops/`)
  - Performance feedback
  - Risk feedback
  - Research feedback

### Changed
- Migrated to production environment
- Enhanced Casey with continuous learning
- Improved Donnie orchestration
- Upgraded Ralph with market data integration
- Enhanced Optimus with legal compliance

### Fixed
- Tradier connection issues
- Account balance visibility
- Funds activation system
- Order execution errors

## [2024-11-XX] - Core Infrastructure

### Added
- Core agent architecture
  - Optimus (`agents/optimus.py`) - Trading execution agent
  - Ralph (`agents/ralph.py`) - Research and strategy agent
  - Donnie (`agents/donnie.py`) - Orchestration agent
  - Casey (`agents/casey.py`) - Financial optimization agent
  - Genny (`agents/genny.py`) - Tax and compliance agent
- Broker adapters
  - Alpaca adapter (`adapters/alpaca.py`)
  - Tradier adapter (`execution/broker_adapters/tradier_adapter.py`)
- Secure vault system (`secure_vault.py`)
- Goal management system (`goal_manager.py`)
- Audit logging system
- Safety controls and kill switches
- Legal compliance checker (`legal_compliance_checker.py`)
- Redis integration for state management
- Mac Auto-Updater Service (`scripts/mac_auto_updater.py`)
- Ralph GitHub Research System (`agents/ralph_github_*.py`)
- Production mode setup (`com.nae.production.plist`)
- Windows service scripts
- Dual Machine Setup (Mac Dev + HP OmniBook X Prod)
- AutoGen Studio integration (`autogen_studio_*.py`)

## [2024-11-XX] - Foundation Updates

### Added
- Core agent architecture (Optimus, Ralph, Donnie, Casey, Genny)
- Broker adapters (Alpaca, Tradier)
- Secure vault system
- Goal management system
- Audit logging
- Safety controls and kill switches
- Legal compliance checker
- Redis integration for state management

### Changed
- Initial project structure
- Agent communication protocols

---

## Version History

- **v0.4.0** - Current version (see =0.4.0 file)
- **v0.3.0** - Major robustness improvements
- **v0.2.0** - Broker integration and trading capabilities
- **v0.1.0** - Initial release

---

## How to Update This Changelog

When making changes:

1. Add entries under `[Unreleased]` section
2. Use categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
3. Include relevant details and file paths
4. When releasing, move `[Unreleased]` to a new versioned section with date
5. Update version number if following semantic versioning

### Example Entry Format

```markdown
### Added
- Feature name (`path/to/file.py`)
  - Description of feature
  - Key methods or capabilities
```

---

## Notes

- This changelog tracks all significant updates to NAE
- Minor bug fixes and internal refactoring may not be listed
- Breaking changes will be clearly marked
- Security updates will be prioritized

