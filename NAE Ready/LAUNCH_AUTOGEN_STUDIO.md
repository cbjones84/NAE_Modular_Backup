# 🚀 Launch AutoGen Studio

You have **3 easy ways** to launch AutoGen Studio and start chatting with Casey:

## Option 1: Double-Click (macOS) ⭐ Recommended

**File:** `Open AutoGen Studio.command`

1. Double-click `Open AutoGen Studio.command` in Finder
2. AutoGen Studio will start automatically
3. Your browser will open to http://localhost:8081
4. Start chatting with Casey!

## Option 2: Python Script (Cross-Platform)

**File:** `Open AutoGen Studio.py`

1. Double-click `Open AutoGen Studio.py` (or run: `python3 "Open AutoGen Studio.py"`)
2. AutoGen Studio will start automatically
3. Your browser will open automatically
4. Start chatting with Casey!

## Option 3: HTML Launcher

**File:** `Open AutoGen Studio.html`

1. Double-click `Open AutoGen Studio.html` to open in your browser
2. Click "Start Server & Open" button
3. AutoGen Studio will start and open automatically
4. Start chatting with Casey!

## 📋 After Launching

Once AutoGen Studio is open:

1. **Create Casey Agent:**
   - Click "Agents" → "Create New"
   - Name: `Casey`
   - Type: `AssistantAgent`
   - Copy system message from `autogen_studio_config.json`
   - Enable tools

2. **Create Workflow:**
   - Click "Workflows" → "Create New"
   - Type: `GroupChat`
   - Add `Casey` and `User` agents
   - Max Rounds: `10`

3. **Start Chatting:**
   - Click "Chats" → "New Chat"
   - Select your workflow
   - Try commands like:
     - `Read agents/optimus.py`
     - `Get system status`
     - `Search for trading strategies`

## 🆘 Troubleshooting

**Port already in use:**
- The launcher will detect if AutoGen Studio is already running
- It will just open your browser to the existing instance

**AutoGen Studio not installed:**
- The launcher will automatically install it for you
- Or install manually: `pip install autogenstudio`

**Browser doesn't open:**
- Manually navigate to: http://localhost:8081

## 🎯 Quick Commands

```bash
# Manual start (if launchers don't work)
cd NAE
./start_autogen_studio.sh

# Or with Python
python3 "Open AutoGen Studio.py"

# Or with command file (macOS)
open "Open AutoGen Studio.command"
```

---

**Choose your preferred method and start chatting with Casey!** 🚀

