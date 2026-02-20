# CHANGELOG Maintenance Guide

## Overview

The `CHANGELOG.md` file tracks all notable changes to the Neural Agency Engine (NAE). This guide explains how to maintain it.

## Format

The changelog follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format:

```markdown
## [Unreleased]

### Added
- New features

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features

### Deprecated
- Soon-to-be removed features

### Security
- Security fixes
```

## When to Update

Update the changelog when:

1. **Adding new features** - Document in `Added` section
2. **Modifying existing features** - Document in `Changed` section
3. **Fixing bugs** - Document in `Fixed` section
4. **Removing features** - Document in `Removed` section
5. **Deprecating features** - Document in `Deprecated` section
6. **Security updates** - Document in `Security` section

## How to Update

### Manual Update

1. Open `CHANGELOG.md`
2. Add entries under `[Unreleased]` section
3. Use appropriate category (Added, Changed, Fixed, etc.)
4. Include file paths and brief descriptions
5. Commit with your changes

### Using Update Script

```bash
# Basic update
./scripts/update_changelog.sh

# With version number
./scripts/update_changelog.sh "0.5.0"

# With description
./scripts/update_changelog.sh "" "Major feature update"
```

### Automatic Update (Deployment)

The `deploy_accelerator.sh` script automatically updates the changelog during deployment.

## Entry Format

### Good Entry

```markdown
### Added
- Micro-Scalp Accelerator Strategy (`tools/profit_algorithms/advanced_micro_scalp.py`)
  - SPY 0DTE options scalping
  - Volatility filters (IV percentile, ATR)
  - Target: $8000-$10000 account growth
```

### Bad Entry

```markdown
### Added
- New feature
```

## Categories Explained

### Added
New features, capabilities, or functionality.

**Examples:**
- New agent capabilities
- New broker integrations
- New algorithms or strategies
- New documentation

### Changed
Changes to existing functionality that don't break compatibility.

**Examples:**
- Updated target account sizes
- Modified risk parameters
- Enhanced existing features
- Performance improvements

### Fixed
Bug fixes and error corrections.

**Examples:**
- Fixed settlement tracking bugs
- Corrected calculation errors
- Resolved connection issues
- Fixed path resolution problems

### Removed
Features that have been removed.

**Examples:**
- Deprecated broker adapters
- Removed obsolete strategies
- Cleaned up unused code

### Deprecated
Features that will be removed in future versions.

**Examples:**
- Legacy API endpoints
- Old configuration formats
- Outdated strategies

### Security
Security-related updates.

**Examples:**
- Vulnerability fixes
- Authentication improvements
- Access control updates

## Versioning

When releasing a new version:

1. Move `[Unreleased]` entries to a new versioned section
2. Use date format: `[YYYY-MM-DD]`
3. Include version number if applicable
4. Update version history section

**Example:**

```markdown
## [2025-01-15] - Version 0.5.0

### Added
- [Previous Unreleased entries]

## [Unreleased]
```

## Best Practices

1. **Be Specific**: Include file paths and method names
2. **Be Concise**: Keep entries brief but informative
3. **Be Consistent**: Use same format throughout
4. **Be Timely**: Update changelog with each significant change
5. **Be Complete**: Don't skip important updates

## Examples

### Feature Addition

```markdown
### Added
- Settlement Ledger (`tools/settlement_utils.py`)
  - Tracks settled vs unsettled cash
  - Prevents free-riding violations
  - T+1 settlement for options, T+2 for stocks
```

### Bug Fix

```markdown
### Fixed
- Path resolution in accelerator controller (`execution/integration/accelerator_controller.py`)
  - Fixed incorrect path detection
  - Improved error handling
```

### Change

```markdown
### Changed
- Accelerator target account size (`tools/profit_algorithms/advanced_micro_scalp.py`)
  - Updated from $500-$1000 to $8000-$10000
  - Adjusted risk parameters accordingly
```

## Integration with Git

The changelog is automatically:
- Updated during deployment (`deploy_accelerator.sh`)
- Committed with other changes
- Pushed to GitHub

## Review Process

Before committing:
1. Review changelog entries for accuracy
2. Ensure all significant changes are documented
3. Check formatting and consistency
4. Verify file paths are correct

---

**Remember**: A well-maintained changelog helps track progress and makes it easier to understand what changed and when.

