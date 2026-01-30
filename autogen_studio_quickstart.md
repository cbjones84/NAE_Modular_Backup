# Quick Start: AutoGen Studio with NAE

## 🚀 Quick Setup (3 Steps)

### Step 1: Install AutoGen Studio
```bash
pip install autogenstudio
```

### Step 2: Start AutoGen Studio
```bash
cd NAE
./start_autogen_studio.sh
```

Or manually:
```bash
autogenstudio ui --port 8081
```

### Step 3: Open Browser
Navigate to: **http://localhost:8081/**

## 📋 Import Configuration

Once AutoGen Studio is running:

1. Click **"Agents"** in the sidebar
2. Click **"Import"** or **"Create New"**
3. Use the configuration from `autogen_studio_config.json`:
   - Name: `Casey`
   - Type: `AssistantAgent`
   - System Message: (from config file)
   - Tools: Enable all available tools

4. Click **"Workflows"** in the sidebar
5. Create a new workflow:
   - Name: `NAE Casey Workflow`
   - Type: `GroupChat`
   - Agents: Add `Casey` and `User`
   - Max Rounds: `10`

## 💬 Start Chatting

1. Click **"Chats"** in the sidebar
2. Click **"New Chat"**
3. Select the **"NAE Casey Workflow"**
4. Start typing commands like:
   - `Read agents/optimus.py`
   - `List directory agents`
   - `Search for trading strategies`
   - `Get system status`
   - `Execute Python: print("Hello NAE")`

## 🔧 Optional: API Bridge

For programmatic access, start the bridge server:

```bash
python3 autogen_studio_bridge.py
```

Then AutoGen Studio can connect via API at `http://localhost:8080`

## 📚 Available Commands

Casey understands these commands through AutoGen Studio:

- **File Operations:**
  - `Read [file_path]` - Read a file
  - `Write [file_path] [content]` - Write to a file
  - `List [directory]` - List files in directory

- **Code Operations:**
  - `Search [query]` - Search codebase
  - `Execute Python: [code]` - Execute Python code
  - `Debug [file_path]` - Debug a file

- **System Operations:**
  - `Get status` - Get system status
  - `Monitor agents` - Monitor all agents
  - `Check health` - Check system health

## 🎯 Example Conversation

**You:** `Read agents/optimus.py`

**Casey:** 
```
**File:** agents/optimus.py

```python
# OptimusAgent implementation
class OptimusAgent:
    ...
```

**You:** `Search for trading strategies`

**Casey:**
```
**Search Results:**
- tools/profit_algorithms/timing_strategies.py: Advanced entry/exit timing...
- agents/optimus.py: Trading execution logic...
```

## 🆘 Troubleshooting

**AutoGen Studio won't start:**
- Check if port 8081 is available: `lsof -i :8081`
- Try a different port: `autogenstudio ui --port 8082`

**Casey not responding:**
- Check if Casey agent is properly configured
- Verify system message includes NAE goals
- Check logs in `logs/casey.log`

**Tools not working:**
- Ensure file paths are relative to NAE root
- Check file permissions
- Verify Casey has access to filesystem

## 📖 More Information

See `AUTOGEN_STUDIO_INTEGRATION.md` for detailed documentation.

