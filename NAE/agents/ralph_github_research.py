#!/usr/bin/env python3
"""
Ralph GitHub Research & Auto-Implementation System
Discovers and implements improvements from GitHub for NAE

Uses GitHub API (not scraping) to:
- Search for options trading algorithms
- Find ML/AI improvements
- Discover better risk management systems
- Auto-implement useful code patterns
- Enhance NAE for maximum profitability

ALIGNED WITH GOALS:
- Maximize options trading profits
- Improve M/L ratios
- Enhance AI capabilities
- Build the BEST autonomous options trading system
- Accelerate generational wealth goal
"""

import os
import sys
import json
import time
import requests
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubResearchEngine:
    """GitHub API-based research engine for finding improvements"""
    
    def __init__(self, github_token: Optional[str] = None):
        # Try to load token from secure config file first
        if not github_token:
            github_token = self._load_token_from_config()
        
        # Fall back to environment variable
        if not github_token:
            github_token = os.getenv('GITHUB_TOKEN', '')
        
        self.github_token = github_token
        self.api_base = "https://api.github.com"
        self.session = requests.Session()
        
        if self.github_token:
            self.session.headers.update({
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
            logger.info("✅ GitHub token loaded successfully")
        else:
            logger.warning("⚠️  No GitHub token found - using limited API access")
        
        # Rate limiting
        self.rate_limit_remaining = 60
        self.rate_limit_reset = time.time() + 3600
        
        # Search categories
        self.search_categories = {
            'options_trading': [
                'options trading algorithm',
                'options strategy backtesting',
                'options greeks calculation',
                'options volatility trading',
                'options risk management',
                'options portfolio optimization'
            ],
            'ml_ai': [
                'machine learning trading',
                'reinforcement learning options',
                'neural network trading',
                'deep learning finance',
                'AI trading strategy',
                'predictive modeling options'
            ],
            'risk_management': [
                'trading risk management',
                'portfolio risk calculation',
                'value at risk options',
                'position sizing algorithm',
                'drawdown management',
                'stop loss optimization'
            ],
            'systems_thinking': [
                'trading system architecture',
                'automated trading framework',
                'trading bot framework',
                'system design trading',
                'scalable trading system'
            ],
            'profit_optimization': [
                'profit maximization trading',
                'sharpe ratio optimization',
                'kelly criterion trading',
                'optimal position sizing',
                'trade execution optimization'
            ]
        }
    
    def _load_token_from_config(self) -> Optional[str]:
        """Load GitHub token from secure config file"""
        try:
            config_file = Path(__file__).parent.parent / 'config' / 'secrets.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('github_token')
        except Exception as e:
            logger.debug(f"Could not load token from config: {e}")
        return None
    
    def _rate_limit_check(self):
        """Check and respect GitHub API rate limits"""
        if time.time() > self.rate_limit_reset:
            # Reset rate limit (check actual limit via API)
            try:
                response = self.session.get(f"{self.api_base}/rate_limit")
                if response.status_code == 200:
                    data = response.json()
                    self.rate_limit_remaining = data['resources']['search']['remaining']
                    self.rate_limit_reset = data['resources']['search']['reset']
            except:
                pass
        
        if self.rate_limit_remaining < 5:
            wait_time = max(0, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.warning(f"Rate limit low, waiting {wait_time:.0f} seconds")
                time.sleep(wait_time)
    
    def search_repositories(self, query: str, language: str = 'python', 
                          sort: str = 'stars', order: str = 'desc',
                          per_page: int = 10) -> List[Dict]:
        """Search GitHub repositories"""
        self._rate_limit_check()
        
        try:
            url = f"{self.api_base}/search/repositories"
            params = {
                'q': f'{query} language:{language}',
                'sort': sort,
                'order': order,
                'per_page': per_page
            }
            
            response = self.session.get(url, params=params, timeout=30)
            self.rate_limit_remaining -= 1
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            elif response.status_code == 403:
                logger.error("GitHub API rate limit exceeded")
                return []
            else:
                logger.error(f"GitHub API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    def search_code(self, query: str, language: str = 'python',
                   per_page: int = 10) -> List[Dict]:
        """Search GitHub code"""
        self._rate_limit_check()
        
        try:
            url = f"{self.api_base}/search/code"
            params = {
                'q': f'{query} language:{language}',
                'per_page': per_page
            }
            
            response = self.session.get(url, params=params, timeout=30)
            self.rate_limit_remaining -= 1
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            elif response.status_code == 403:
                logger.error("GitHub API rate limit exceeded")
                return []
            else:
                logger.error(f"GitHub API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching code: {e}")
            return []
    
    def get_repository_content(self, owner: str, repo: str, path: str = '') -> Optional[Dict]:
        """Get repository file/directory content"""
        self._rate_limit_check()
        
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
            response = self.session.get(url, timeout=30)
            self.rate_limit_remaining -= 1
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting repository content: {e}")
            return None
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get file content from repository"""
        content = self.get_repository_content(owner, repo, path)
        if content and content.get('type') == 'file':
            import base64
            try:
                decoded = base64.b64decode(content['content']).decode('utf-8')
                return decoded
            except:
                return None
        return None
    
    def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code for quality and usefulness"""
        analysis = {
            'has_algorithm': False,
            'has_ml_components': False,
            'has_risk_management': False,
            'complexity_score': 0,
            'usefulness_score': 0,
            'keywords_found': []
        }
        
        # Keywords for different categories
        algorithm_keywords = [
            'def ', 'class ', 'algorithm', 'strategy', 'backtest',
            'optimize', 'calculate', 'compute', 'simulate'
        ]
        
        ml_keywords = [
            'tensorflow', 'pytorch', 'sklearn', 'neural', 'model',
            'train', 'predict', 'fit', 'gradient', 'loss'
        ]
        
        risk_keywords = [
            'risk', 'sharpe', 'drawdown', 'volatility', 'var',
            'position', 'size', 'stop', 'loss', 'greek'
        ]
        
        # Check for keywords
        code_lower = code.lower()
        
        for keyword in algorithm_keywords:
            if keyword in code_lower:
                analysis['has_algorithm'] = True
                analysis['keywords_found'].append(keyword)
        
        for keyword in ml_keywords:
            if keyword in code_lower:
                analysis['has_ml_components'] = True
                analysis['keywords_found'].append(keyword)
        
        for keyword in risk_keywords:
            if keyword in code_lower:
                analysis['has_risk_management'] = True
                analysis['keywords_found'].append(keyword)
        
        # Calculate complexity (simple heuristic)
        lines = code.split('\n')
        analysis['complexity_score'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # Calculate usefulness score
        usefulness = 0
        if analysis['has_algorithm']:
            usefulness += 30
        if analysis['has_ml_components']:
            usefulness += 40
        if analysis['has_risk_management']:
            usefulness += 30
        
        # Bonus for having multiple components
        component_count = sum([
            analysis['has_algorithm'],
            analysis['has_ml_components'],
            analysis['has_risk_management']
        ])
        if component_count >= 2:
            usefulness += 20
        
        analysis['usefulness_score'] = min(100, usefulness)
        
        return analysis


class CodeImplementationEngine:
    """Engine for auto-implementing discovered code"""
    
    def __init__(self, nae_root: Path):
        self.nae_root = nae_root
        self.implementations_dir = nae_root / 'agents' / 'generated_scripts' / 'github_improvements'
        self.implementations_dir.mkdir(parents=True, exist_ok=True)
        
        self.implemented_hashes = set()
        self.load_implementation_history()
    
    def load_implementation_history(self):
        """Load history of implemented code"""
        history_file = self.implementations_dir / 'implementation_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.implemented_hashes = set(data.get('hashes', []))
            except:
                pass
    
    def save_implementation_history(self):
        """Save implementation history"""
        history_file = self.implementations_dir / 'implementation_history.json'
        data = {
            'hashes': list(self.implemented_hashes),
            'last_updated': datetime.now().isoformat()
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def hash_code(self, code: str) -> str:
        """Generate hash for code to avoid duplicates"""
        return hashlib.sha256(code.encode()).hexdigest()
    
    def extract_useful_functions(self, code: str, analysis: Dict) -> List[Dict]:
        """Extract useful functions/classes from code"""
        functions = []
        
        # Find function definitions
        function_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
        for match in re.finditer(function_pattern, code):
            func_name = match.group(1)
            start = match.start()
            
            # Find function end (simple heuristic)
            lines = code[start:].split('\n')
            func_lines = []
            indent_level = None
            
            for i, line in enumerate(lines):
                if i == 0:
                    func_lines.append(line)
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if line.strip() and not line.strip().startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                
                func_lines.append(line)
            
            func_code = '\n'.join(func_lines)
            
            # Check if function is useful
            if any(kw in func_code.lower() for kw in ['trade', 'option', 'risk', 'profit', 'calculate', 'optimize']):
                functions.append({
                    'name': func_name,
                    'code': func_code,
                    'start': start
                })
        
        return functions
    
    def implement_improvement(self, code: str, source_repo: str, 
                             source_path: str, analysis: Dict) -> bool:
        """Implement discovered code improvement"""
        code_hash = self.hash_code(code)
        
        # Skip if already implemented
        if code_hash in self.implemented_hashes:
            logger.info(f"Skipping already implemented code: {code_hash[:8]}")
            return False
        
        # Only implement high-quality code
        if analysis['usefulness_score'] < 50:
            logger.info(f"Code usefulness too low: {analysis['usefulness_score']}")
            return False
        
        try:
            # Extract useful functions
            functions = self.extract_useful_functions(code, analysis)
            
            if not functions:
                logger.info("No useful functions found in code")
                return False
            
            # Create implementation file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            impl_file = self.implementations_dir / f'improvement_{timestamp}.py'
            
            with open(impl_file, 'w') as f:
                f.write(f'"""\n')
                f.write(f'Auto-implemented improvement from GitHub\n')
                f.write(f'Source: {source_repo}/{source_path}\n')
                f.write(f'Implemented: {datetime.now().isoformat()}\n')
                f.write(f'Usefulness Score: {analysis["usefulness_score"]}\n')
                f.write(f'Keywords: {", ".join(analysis["keywords_found"])}\n')
                f.write(f'"""\n\n')
                f.write(f'# Original source: {source_repo}\n')
                f.write(f'# Path: {source_path}\n\n')
                
                for func in functions:
                    f.write(f'\n# Function: {func["name"]}\n')
                    f.write(func['code'])
                    f.write('\n\n')
            
            # Mark as implemented
            self.implemented_hashes.add(code_hash)
            self.save_implementation_history()
            
            logger.info(f"✅ Implemented improvement: {impl_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error implementing improvement: {e}")
            return False


class RalphGitHubResearch:
    """Main Ralph GitHub Research System"""
    
    def __init__(self, nae_root: Optional[Path] = None):
        self.nae_root = Path(nae_root) if nae_root else Path(__file__).parent.parent
        self.research_engine = GitHubResearchEngine()
        self.implementation_engine = CodeImplementationEngine(self.nae_root)
        
        self.results_dir = self.nae_root / 'logs' / 'github_research'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def research_category(self, category: str, queries: List[str]) -> List[Dict]:
        """Research a specific category"""
        logger.info(f"🔍 Researching category: {category}")
        
        all_results = []
        
        for query in queries:
            logger.info(f"  Searching: {query}")
            
            # Search repositories
            repos = self.research_engine.search_repositories(query, per_page=5)
            
            for repo in repos:
                repo_info = {
                    'category': category,
                    'query': query,
                    'repo_name': repo['full_name'],
                    'repo_url': repo['html_url'],
                    'stars': repo['stargazers_count'],
                    'description': repo.get('description', ''),
                    'language': repo.get('language', ''),
                    'files_analyzed': []
                }
                
                # Analyze repository files
                try:
                    # Get main files (limit to avoid rate limits)
                    content = self.research_engine.get_repository_content(
                        repo['owner']['login'], repo['name']
                    )
                    
                    if content and isinstance(content, list):
                        for item in content[:5]:  # Limit to 5 files
                            if item['type'] == 'file' and item['name'].endswith('.py'):
                                file_content = self.research_engine.get_file_content(
                                    repo['owner']['login'], repo['name'], item['path']
                                )
                                
                                if file_content:
                                    analysis = self.research_engine.analyze_code_quality(file_content)
                                    analysis['file_path'] = item['path']
                                    analysis['file_url'] = item['html_url']
                                    
                                    repo_info['files_analyzed'].append(analysis)
                                    
                                    # Auto-implement if useful
                                    if analysis['usefulness_score'] >= 70:
                                        self.implementation_engine.implement_improvement(
                                            file_content,
                                            repo['full_name'],
                                            item['path'],
                                            analysis
                                        )
                    
                    all_results.append(repo_info)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {repo['full_name']}: {e}")
                    continue
            
            # Rate limit protection
            time.sleep(2)
        
        return all_results
    
    def run_full_research(self) -> Dict[str, Any]:
        """Run full research across all categories"""
        logger.info("=" * 60)
        logger.info("🚀 Starting Ralph GitHub Research")
        logger.info("=" * 60)
        
        all_results = {}
        
        for category, queries in self.research_engine.search_categories.items():
            results = self.research_category(category, queries)
            all_results[category] = results
            
            logger.info(f"✅ {category}: Found {len(results)} repositories")
        
        # Save results
        results_file = self.results_dir / f'research_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info("=" * 60)
        
        return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ralph GitHub Research System')
    parser.add_argument('--category', help='Research specific category')
    parser.add_argument('--full', action='store_true', help='Run full research')
    parser.add_argument('--github-token', help='GitHub API token (or set GITHUB_TOKEN env var)')
    
    args = parser.parse_args()
    
    research = RalphGitHubResearch()
    
    if args.github_token:
        research.research_engine.github_token = args.github_token
        research.research_engine.session.headers.update({
            'Authorization': f'token {args.github_token}'
        })
    
    if args.full:
        research.run_full_research()
    elif args.category:
        if args.category in research.research_engine.search_categories:
            queries = research.research_engine.search_categories[args.category]
            research.research_category(args.category, queries)
        else:
            logger.error(f"Unknown category: {args.category}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

