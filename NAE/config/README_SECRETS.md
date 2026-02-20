# Secrets Configuration

## Security Notice

This directory contains sensitive credentials and should be:
- **Never committed to git** (already in .gitignore)
- **Protected with proper file permissions** (600 - owner read/write only)
- **Kept secure** on the local machine only

## Files

### `secrets.json`
Contains secure tokens and API keys:
- GitHub Personal Access Token for Ralph GitHub Research System
- Owner: Chris Jones

## File Permissions

The secrets file is set to `600` (read/write for owner only):
```bash
chmod 600 config/secrets.json
```

**Owner:** Chris Jones

## Usage

The GitHub token is automatically loaded by:
- `agents/ralph_github_research.py`
- `agents/ralph_github_continuous.py`

No manual configuration needed - the system will find and use the token automatically.

## Backup

If you need to backup secrets:
1. Use secure backup method (encrypted)
2. Never store in plain text
3. Never share publicly

## Rotation

To update the GitHub token:
1. Edit `config/secrets.json`
2. Update `github_token` field
3. Update `last_updated` timestamp
4. Save file (permissions will be preserved)

