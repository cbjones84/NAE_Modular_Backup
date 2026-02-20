# Quick Start: AutoGen Studio with NAE

## 🚀 Quick Setup (5 minutes)

### 1. Generate NAE Configurations

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 autogen_studio_nae_integration.py
```

This creates:
- Agent configs: `.autogenstudio/agents.json`
- Workflow configs: `.autogenstudio/workflows.json`

### 2. Start AutoGen Studio

**Option A: Using startup script**
```bash
./scripts/start_autogen_studio_nae.sh
```

**Option B: Manual start**
```bash
autogenstudio ui --port 8080
```

### 3. Access AutoGen Studio

Open your browser to: **http://localhost:8080**

### 4. Import NAE Agents

1. In AutoGen Studio UI, go to **"Team Builder"**
2. Click **"Import"** or **"Add Agent"**
3. Select agents from `.autogenstudio/agents.json`:
   - Casey (System Orchestrator)
   - Optimus (Trading Agent)
   - Ralph (Strategy Research)
   - Donnie (Market Data)
   - Genny (Wealth Management)

### 5. Create or Use Pre-configured Workflows

**Pre-configured workflows available:**
- **NAE Trading Workflow**: Casey + Optimus + Ralph
- **NAE Research Workflow**: Casey + Ralph + Donnie  
- **NAE Wealth Management**: Casey + Genny + Optimus
- **NAE Full System**: All agents

Import from `.autogenstudio/workflows.json` or create custom workflows in the UI.

## 📋 What You Can Do

### Visual Workflow Design
- Drag and drop agents into workflows
- Configure agent interactions
- Set termination conditions
- Test workflows in real-time

### Interactive Testing
- Use the Playground to test agent conversations
- Monitor agent interactions
- Debug workflow issues
- Analyze performance

### Export & Deploy
- Export workflows as JSON
- Integrate into NAE programmatically
- Deploy workflows to production

## 🎯 Example: Trading Workflow

1. **Create Workflow**: "Daily Trading Strategy"
2. **Add Agents**: Casey, Optimus, Ralph
3. **Configure**:
   - Casey: Analyzes market conditions
   - Ralph: Generates strategy recommendations
   - Optimus: Executes trades
4. **Test**: Run in Playground with sample task
5. **Deploy**: Export JSON and integrate into NAE

## 📚 Full Documentation

See [AUTOGEN_STUDIO_NAE_INTEGRATION.md](AUTOGEN_STUDIO_NAE_INTEGRATION.md) for complete documentation.

## 🔗 Resources

- [AutoGen Studio GitHub](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-studio)
- [AutoGen Studio Docs](https://microsoft.github.io/autogen/dev/user-guide/autogenstudio-user-guide/)

