# AutoGen Studio Integration for NAE

This document describes the integration of Microsoft AutoGen Studio with the Neural Agency Engine (NAE).

## Overview

AutoGen Studio is a low-code interface for building, testing, and deploying multi-agent AI workflows. By integrating AutoGen Studio with NAE, you can:

- **Visual Workflow Design**: Design agent workflows visually without extensive coding
- **Interactive Testing**: Test and debug agent interactions in real-time
- **Rapid Prototyping**: Quickly prototype new trading strategies and agent configurations
- **Seamless Deployment**: Export workflows as JSON and integrate into NAE

## Installation

### 1. Install AutoGen Studio

```bash
pip install autogenstudio
```

Or update requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate NAE Configurations

```bash
python3 autogen_studio_nae_integration.py
```

This will create:
- Agent configurations in `.autogenstudio/agents.json`
- Workflow configurations in `.autogenstudio/workflows.json`

## Usage

### Starting AutoGen Studio

**Option 1: Using the startup script**
```bash
./scripts/start_autogen_studio_nae.sh
```

**Option 2: Manual start**
```bash
autogenstudio ui --port 8080
```

Then access the UI at: http://localhost:8080

### Available NAE Agents in AutoGen Studio

1. **Casey** - System orchestrator
   - File operations
   - Codebase search
   - Code execution
   - Debugging

2. **Optimus** - Trading agent
   - Execute trades
   - Monitor positions
   - Check balances

3. **Ralph** - Strategy research
   - Backtest strategies
   - Analyze market data
   - Recommend strategies

4. **Donnie** - Market data agent
   - Fetch market data
   - Analyze trends
   - Monitor news

5. **Genny** - Wealth management
   - Track trades
   - Calculate taxes
   - Generate reports

### Pre-configured Workflows

1. **NAE Trading Workflow**
   - Agents: Casey, Optimus, Ralph
   - Purpose: Execute trading strategies

2. **NAE Research Workflow**
   - Agents: Casey, Ralph, Donnie
   - Purpose: Research and develop strategies

3. **NAE Wealth Management**
   - Agents: Casey, Genny, Optimus
   - Purpose: Manage wealth and taxes

4. **NAE Full System**
   - Agents: All NAE agents
   - Purpose: Complete system coordination

## Creating Custom Workflows

### In AutoGen Studio UI

1. Open AutoGen Studio at http://localhost:8080
2. Navigate to "Team Builder"
3. Create new agents or use existing NAE agents
4. Configure agent roles, tools, and models
5. Set termination conditions
6. Test in the Playground
7. Export as JSON configuration

### Programmatically

```python
from autogen_studio_nae_integration import NAEStudioIntegration

# Create integration
integration = NAEStudioIntegration()

# Create custom workflow
workflow = integration.create_workflow_config(
    "My Custom Workflow",
    ["Casey", "Optimus"]
)

# Run workflow
result = integration.run_workflow("My Custom Workflow", "Execute trading strategy")
```

## Integration Architecture

```
┌─────────────────┐
│ AutoGen Studio  │
│      UI         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TeamManager    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NAEAgentBridge  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   NAE Agents    │
│ (Casey, Optimus,│
│  Ralph, etc.)   │
└─────────────────┘
```

## Configuration Files

### Agent Configuration (`.autogenstudio/agents.json`)

Each agent configuration includes:
- Name and type
- System message
- Available tools
- Code execution settings
- Model configuration

### Workflow Configuration (`.autogenstudio/workflows.json`)

Each workflow includes:
- Workflow name
- Agent list
- Max rounds
- Speaker selection method
- Termination conditions

## Advanced Usage

### Custom Agent Tools

You can extend agents with custom tools:

```python
def custom_trading_tool(symbol: str, action: str) -> str:
    """Custom trading tool"""
    # Implementation
    return f"Executed {action} on {symbol}"

# Add to agent config
agent_config["tools"].append("custom_trading_tool")
```

### Workflow Execution

```python
from autogenstudio import TeamManager

tm = TeamManager()
result = tm.run(
    task="Execute trading strategy for AAPL",
    team_config="path/to/workflow.json"
)
```

## Troubleshooting

### AutoGen Studio Not Starting

1. Check installation:
   ```bash
   pip list | grep autogenstudio
   ```

2. Check port availability:
   ```bash
   lsof -i :8080
   ```

3. Try different port:
   ```bash
   autogenstudio ui --port 8081
   ```

### Agents Not Available

1. Verify NAE agents are importable:
   ```bash
   python3 -c "from agents.casey import CaseyAgent; print('OK')"
   ```

2. Regenerate configurations:
   ```bash
   python3 autogen_studio_nae_integration.py
   ```

### Workflow Execution Errors

1. Check agent configurations
2. Verify all required tools are available
3. Check model API keys are set
4. Review workflow termination conditions

## Resources

- [AutoGen Studio GitHub](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-studio)
- [AutoGen Studio Documentation](https://microsoft.github.io/autogen/dev/user-guide/autogenstudio-user-guide/)
- [NAE Documentation](../docs/README.md)

## Support

For issues or questions:
1. Check NAE logs: `logs/nae_autonomous_master.log`
2. Check AutoGen Studio logs in the UI
3. Review agent configurations
4. Test individual agents before workflows

