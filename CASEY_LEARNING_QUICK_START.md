# Casey Enhanced Learning - Quick Start

## Setup

### 1. Install Required Packages

```bash
# OpenAI (for ChatGPT)
pip install openai

# Google Generative AI (for Gemini)
pip install google-generativeai

# Grok (if available)
# Check Grok API documentation for installation
```

### 2. Set Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
export GROK_API_KEY="your_grok_key"  # If available
```

### 3. Restart Casey

The learning system automatically starts when Casey initializes:

```python
from agents.casey import CaseyAgent

casey = CaseyAgent()
# Learning system is now active!
```

## What Happens Automatically

### Every Hour
- ✅ Queries ChatGPT, Grok, and Gemini with learning prompts
- ✅ Extracts insights from responses
- ✅ Synthesizes knowledge from all sources
- ✅ Generates improvement actions
- ✅ Auto-applies safe improvements

### Every 5 Minutes
- ✅ Checks system health
- ✅ Detects issues
- ✅ Auto-heals common problems
- ✅ Updates self-awareness score

### Continuously
- ✅ Analyzes financial optimization opportunities
- ✅ Suggests safe optimizations
- ✅ Tracks gains

## Check Status

```python
# Get learning report
report = casey.get_learning_report()
print(f"Insights learned: {report['insights_learned']}")
print(f"Self-awareness: {report['self_awareness']['self_awareness_score']}")

# Get improvement suggestions
suggestions = casey.get_improvement_suggestions()
for s in suggestions:
    print(f"- {s['title']}: {s['description']}")
```

## Learn from Interactions

Casey can learn from any AI interaction:

```python
# Learn from Cursor/Auto
casey.learn_from_interaction(
    prompt="How can I improve NAE?",
    response="Use async/await for better performance...",
    source="cursor_auto"
)

# Learn from ChatGPT
casey.learn_from_interaction(
    prompt="Optimize trading strategies",
    response="Implement Kelly Criterion...",
    source="chatgpt"
)
```

## View Logs

```bash
# Casey learning logs
tail -f logs/casey.log | grep "Learning\|Self-Healing\|Optimization"

# Check for insights
grep "📚 Learned" logs/casey.log
```

## Status

✅ **Fully Integrated** - Starts automatically
✅ **Production Ready** - Comprehensive error handling
✅ **Self-Aware** - Monitors and heals itself
✅ **Compliance Safe** - All suggestions verified

---

**Casey is now continuously learning and improving!** 🧠

