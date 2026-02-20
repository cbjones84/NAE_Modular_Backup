# AutoGen Studio Integration for NAE

This guide explains how to connect NAE to AutoGen Studio so you can communicate with Casey through AutoGen Studio's interface.

## Overview

AutoGen Studio provides a visual interface for interacting with AutoGen agents. This integration allows you to:
- Chat with Casey through AutoGen Studio
- Send commands and receive responses
- Use Casey's tools (file operations, code execution, etc.)
- Monitor NAE system status

## Setup

### 1. Install AutoGen Studio

```bash
pip install autogenstudio
```

### 2. Run Setup Script

```bash
cd NAE
python3 autogen_studio_setup.py
```

This will create:
- `autogen_studio_config.json` - AutoGen Studio configuration
- `autogen_functions.json` - Function definitions
- `autogen_workflow.json` - Workflow configuration

### 3. Start AutoGen Studio Bridge (Optional)

For API-based integration:

```bash
python3 autogen_studio_bridge.py
```

This starts a bridge server on `http://localhost:8080` that AutoGen Studio can connect to.

### 4. Start AutoGen Studio

```bash
autogenstudio ui
```

AutoGen Studio will start on `http://localhost:8081` (default).

### 5. Import Configuration

In AutoGen Studio:
1. Go to Settings/Configuration
2. Import `autogen_studio_config.json`
3. Import `autogen_functions.json`
4. Import `autogen_workflow.json`

## Usage

### Through AutoGen Studio UI

1. Open AutoGen Studio in your browser
2. Select the "NAE Casey Workflow"
3. Start chatting with Casey
4. Use commands like:
   - "Read agents/optimus.py"
   - "List directory agents"
   - "Search for trading strategies"
   - "Get system status"
   - "Execute Python code: print('Hello')"

### Through API Bridge

If using the bridge server, AutoGen Studio can connect via API:

```python
import requests

response = requests.post('http://localhost:8080/api/casey/process', json={
    'message': 'Read agents/optimus.py',
    'sender': 'User'
})

print(response.json()['response'])
```

## Available Tools

Casey exposes these tools to AutoGen Studio:

1. **read_file** - Read files from codebase
2. **write_file** - Write files to codebase
3. **list_directory** - List files and directories
4. **search_codebase** - Semantic code search
5. **execute_python** - Execute Python code
6. **get_system_status** - Get NAE system status

## Configuration

### Agent Configuration

Casey is configured as an AssistantAgent with:
- System message describing Casey's capabilities
- Tools for file operations, code execution, etc.
- Code execution enabled (no Docker)

### Workflow Configuration

The workflow uses a GroupChat with:
- Casey (AssistantAgent)
- User (UserProxyAgent)
- Round-robin speaker selection
- Max 10 rounds per conversation

## Troubleshooting

### AutoGen Studio not connecting

1. Check if bridge server is running: `python3 autogen_studio_bridge.py`
2. Verify AutoGen Studio can reach `http://localhost:8080`
3. Check firewall settings

### Casey not responding

1. Verify Casey agent is initialized correctly
2. Check logs in `logs/casey.log`
3. Ensure all dependencies are installed

### Tools not working

1. Verify file paths are relative to NAE root
2. Check file permissions
3. Ensure Casey has access to the filesystem

## Advanced Configuration

### Custom Tools

Add custom tools by modifying `autogen_studio_integration.py`:

```python
def custom_tool(param: str) -> str:
    """Your custom tool"""
    casey = get_casey().casey
    # Use Casey's methods
    return result
```

### Multiple Agents

Add more NAE agents to AutoGen Studio:

```python
optimus_agent = AssistantAgent(
    name="Optimus",
    system_message="Optimus trading agent...",
    # ...
)
```

## Integration Points

### Casey → AutoGen Studio

- Commands received from AutoGen Studio
- Responses sent back through bridge
- Tool execution results

### AutoGen Studio → Casey

- User messages
- Tool calls
- Workflow orchestration

## Benefits

1. **Visual Interface**: Use AutoGen Studio's UI instead of command line
2. **Workflow Management**: Organize conversations and workflows
3. **Tool Integration**: Easy access to Casey's capabilities
4. **Multi-Agent**: Can add other NAE agents to conversations
5. **History**: AutoGen Studio maintains conversation history

## Example Workflow

1. User sends: "Read agents/optimus.py"
2. AutoGen Studio routes to Casey
3. Casey processes command using intelligence engine
4. Casey reads file and returns content
5. AutoGen Studio displays response
6. User can continue conversation or use tools

## Next Steps

- Add more NAE agents to AutoGen Studio
- Create custom workflows
- Integrate with other AutoGen Studio features
- Set up persistent storage for conversations

