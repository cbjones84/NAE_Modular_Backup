# NAE/QUICK_START_GUIDE.md
"""
Quick Start Guide for Enhanced NAE Systems
"""

# 🚀 NAE Quick Start Guide

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Systems
```bash
python3 nae_integration.py
```

### 3. Migrate API Keys to Vault
```python
from nae_integration import get_nae_integration
nae = get_nae_integration()
nae.migrate_api_keys_to_vault()
```

### 4. Configure Environment Variables
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export NAE_ENVIRONMENT="sandbox"  # or paper, live, test
```

## Usage Examples

### Secure Vault
```python
from secure_vault import get_vault
vault = get_vault()
vault.set_secret("polygon", "api_key", "your_key")
value = vault.get_secret("polygon", "api_key")
```

### Environment Management
```python
from environment_manager import get_env_manager, Environment
manager = get_env_manager()
manager.set_environment(Environment.PAPER)
profile = manager.get_current_profile()
```

### Model Assignment
```python
from model_config import get_model_manager
manager = get_model_manager()
llm_config = manager.get_llm_config("RalphAgent")
```

### AutoTest Framework
```python
from autotest_framework import AutoTestFramework
framework = AutoTestFramework()
framework.run_all_agent_tests()
framework.print_summary()
```

### Command Execution
```python
from command_executor import get_executor
executor = get_executor()
result = executor.execute_python_code("print('Hello NAE')")
```

### Multi-Step Planning
```python
from multi_step_planner import get_planner
planner = get_planner()
plan = planner.create_autogen_workflow("Generate and execute strategy")
result = planner.execute_plan(plan.plan_id)
```

### Debugging
```python
from debug_tools import get_debug_tools
debug = get_debug_tools()
info = debug.debug_agent("Ralph")
info = debug.inspect_method("Ralph", "generate_strategies")
```

## Testing

### Run All Tests
```bash
python3 autotest_framework.py
```

### Run Integration Tests
```python
from nae_integration import get_nae_integration
nae = get_nae_integration()
nae.run_full_test_suite()
```

## Status Reports

### Ralph Learning Status
See `RALPH_LEARNING_STATUS.md`

### Comprehensive Assessment
See `NAE_COMPREHENSIVE_ASSESSMENT.md`

### Implementation Summary
See `IMPLEMENTATION_SUMMARY.md`


