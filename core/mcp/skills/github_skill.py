# /Users/melissabishop/NAE_Ready/core/mcp/skills/github_skill.py
import json
import os
import sys

# Ensure project root is in path for agents.ralph import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

async def hunt_github(query: str, limit: int = 5) -> str:
    """
    Search GitHub for trading strategies and algorithms.
    """
    try:
        from agents.ralph import GitHubClient
        client = GitHubClient() 
        results = client.search_repositories(query, sort="stars")[:limit]
        
        formatted_results = []
        for repo in results:
            formatted_results.append({
                "name": repo.get("full_name"),
                "description": repo.get("description"),
                "stars": repo.get("stargazers_count"),
                "url": repo.get("html_url")
            })
            
        return json.dumps(formatted_results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error hunting GitHub: {str(e)}"})
