# /Users/melissabishop/NAE_Ready/core/mcp/nae_research_mcp.py
"""
NAE Research MCP Server
Modular version using dynamic skill loading.
"""

import sys
import os
import logging
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nae-research-mcp")

# Initialize FastMCP Server
mcp = FastMCP("NAE Research")

# Add project root to sys.path for internal imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import skills
try:
    from core.mcp.skills import github_skill, analysis_skill, market_skill
    
    # Register GitHub Tool
    @mcp.tool()
    async def hunt_github(query: str, limit: int = 5) -> str:
        """Search GitHub for trading strategies and algorithms."""
        return await github_skill.hunt_github(query, limit)

    # Register Analysis Tool
    @mcp.tool()
    async def analyze_strategy(params: str) -> str:
        """Analyze a potential trading strategy based on provided parameters."""
        return await analysis_skill.analyze_strategy(params)

    # Register Web Research Tool
    @mcp.tool()
    async def web_research(ticker: str) -> str:
        """Perform market research on a specific ticker."""
        return await market_skill.web_research(ticker)

except ImportError as e:
    logger.error(f"Failed to load MCP skills: {e}")

if __name__ == "__main__":
    mcp.run()
