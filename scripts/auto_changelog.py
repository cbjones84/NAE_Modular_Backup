#!/usr/bin/env python3
"""
Automated Changelog Generator for NAE

Analyzes git commits and automatically updates CHANGELOG.md
"""

import os
import re
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

class AutoChangelog:
    def __init__(self, repo_root: str = None):
        self.repo_root = repo_root or self._find_repo_root()
        self.changelog_path = Path(self.repo_root) / "NAE Ready" / "CHANGELOG.md"
        self.change_log_path = Path(self.repo_root) / "NAE Ready" / "logs" / "change_log.json"
        
    def _find_repo_root(self) -> str:
        """Find git repository root"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return os.getcwd()
    
    def get_recent_commits(self, since: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent git commits"""
        cmd = ["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso"]
        
        if since:
            cmd.extend([f"--since={since}"])
        
        cmd.extend([f"-{limit}"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.repo_root)
            commits = []
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('|', 3)
                if len(parts) >= 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    })
            
            return commits
        except Exception as e:
            print(f"Error getting commits: {e}")
            return []
    
    def analyze_commit(self, commit: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a commit to extract change information"""
        msg = commit["message"]
        
        # Categorize based on commit message
        categories = {
            "Added": ["add", "create", "new", "implement"],
            "Changed": ["update", "modify", "change", "improve", "enhance"],
            "Fixed": ["fix", "bug", "error", "issue", "resolve"],
            "Removed": ["remove", "delete", "drop"],
            "Deprecated": ["deprecate", "obsolete"],
            "Security": ["security", "vulnerability", "auth"]
        }
        
        change_type = "Changed"
        msg_lower = msg.lower()
        
        for cat, keywords in categories.items():
            if any(keyword in msg_lower for keyword in keywords):
                change_type = cat
                break
        
        # Extract files changed
        try:
            result = subprocess.run(
                ["git", "show", "--name-status", "--pretty=format:", commit["hash"]],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root
            )
            
            files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    status = line[0]
                    filepath = line[1:].strip()
                    if filepath.startswith("NAE Ready/"):
                        files.append({
                            "status": status,
                            "path": filepath
                        })
        except:
            files = []
        
        return {
            "type": change_type,
            "files": files,
            "summary": self._extract_summary(msg)
        }
    
    def _extract_summary(self, message: str) -> str:
        """Extract summary from commit message"""
        # Take first line, remove common prefixes
        summary = message.split('\n')[0]
        summary = re.sub(r'^(feat|fix|chore|docs|style|refactor|test|perf):\s*', '', summary, flags=re.I)
        return summary.strip()
    
    def update_changelog(self, commits: List[Dict[str, Any]], dry_run: bool = False):
        """Update CHANGELOG.md with recent commits"""
        if not self.changelog_path.exists():
            print(f"CHANGELOG.md not found at {self.changelog_path}")
            return
        
        # Read current changelog
        with open(self.changelog_path, 'r') as f:
            content = f.read()
        
        # Group commits by date
        commits_by_date = {}
        for commit in commits:
            date_str = commit["date"].split()[0]  # YYYY-MM-DD
            if date_str not in commits_by_date:
                commits_by_date[date_str] = []
            commits_by_date[date_str].append(commit)
        
        # Build changelog entries
        entries = []
        for date_str in sorted(commits_by_date.keys(), reverse=True):
            date_commits = commits_by_date[date_str]
            
            # Analyze commits
            changes = {
                "Added": [],
                "Changed": [],
                "Fixed": [],
                "Removed": [],
                "Deprecated": [],
                "Security": []
            }
            
            for commit in date_commits:
                analysis = self.analyze_commit(commit)
                change_type = analysis["type"]
                
                if change_type in changes:
                    entry = f"- {analysis['summary']}"
                    if analysis["files"]:
                        file_list = ", ".join([f["path"] for f in analysis["files"][:3]])
                        if len(analysis["files"]) > 3:
                            file_list += f" (+{len(analysis['files'])-3} more)"
                        entry += f" (`{file_list}`)"
                    changes[change_type].append(entry)
            
            # Build entry if there are changes
            if any(changes.values()):
                entry_text = f"\n### [{date_str}] - Automated Update\n\n"
                
                for change_type in ["Added", "Changed", "Fixed", "Removed", "Deprecated", "Security"]:
                    if changes[change_type]:
                        entry_text += f"#### {change_type}\n"
                        entry_text += "\n".join(changes[change_type]) + "\n\n"
                
                entries.append(entry_text)
        
        if not entries:
            print("No new changes to add to changelog")
            return
        
        # Insert entries after [Unreleased] header
        unreleased_pattern = r'(## \[Unreleased\]\n)'
        
        if re.search(unreleased_pattern, content):
            new_content = re.sub(
                unreleased_pattern,
                r'\1' + '\n'.join(entries),
                content,
                count=1
            )
            
            if not dry_run:
                with open(self.changelog_path, 'w') as f:
                    f.write(new_content)
                print(f"Updated CHANGELOG.md with {len(commits)} commits")
            else:
                print("DRY RUN - Would update CHANGELOG.md:")
                print('\n'.join(entries))
        else:
            print("No [Unreleased] section found in CHANGELOG.md")
    
    def sync_from_git(self, days: int = 7, dry_run: bool = False):
        """Sync changelog from git commits"""
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        commits = self.get_recent_commits(since=since_date)
        
        if commits:
            self.update_changelog(commits, dry_run=dry_run)
        else:
            print(f"No commits found since {since_date}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Changelog Generator")
    parser.add_argument("--sync", action="store_true", help="Sync from git commits")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    
    args = parser.parse_args()
    
    generator = AutoChangelog()
    
    if args.sync:
        generator.sync_from_git(days=args.days, dry_run=args.dry_run)
    else:
        print("Use --sync to sync changelog from git commits")


if __name__ == "__main__":
    main()

