# Dual Machine Setup: Mac (Dev) + HP OmniBook X (Prod)

## Overview

This setup enables a safe, efficient dual-machine workflow:
- **Mac**: Development, testing, strategy updates
- **HP OmniBook X**: Production trading only

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Mac (Dev)     │         │  HP OmniBook X   │
│                 │         │     (Prod)       │
│  - Development  │◄───────►│  - Live Trading  │
│  - Testing      │  Git    │  - Production    │
│  - Strategy Gen │         │  - Execution     │
│  - Monitoring   │         │                  │
└─────────────────┘         └──────────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │   GitHub    │
              │  Repository │
              └─────────────┘
```

## Git Branch Strategy

### Branches

- **`main`**: Stable, tested code (shared)
- **`dev`**: Mac development branch
- **`prod`**: HP production branch

### Workflow

1. **Mac (Dev)**:
   - Work on `dev` branch
   - Test all changes
   - Merge to `main` when stable

2. **HP (Prod)**:
   - Always on `prod` branch
   - Pulls from `main`
   - Never commits directly
   - Only runs production code

## Safety Features

### 1. Environment Separation
- Mac: `PRODUCTION=false`
- HP: `PRODUCTION=true`
- Separate `.env` files per machine

### 2. Anti-Double-Trade System
- Device ID tracking
- Trade nonce system
- Timestamp validation
- Production lock file
- Branch verification

### 3. Production Safety Checks
- Branch must be `prod` for live trading
- Device ID must match authorized device
- Production lock file must exist
- Last trade timestamp validation

## Setup Steps

### Step 1: Mac Setup (Development)

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
git checkout -b dev
git push -u origin dev
```

### Step 2: HP Setup (Production)

```bash
cd "/path/to/NAE"
git checkout -b prod
git push -u origin prod
```

### Step 3: Configure Environments

See `setup/configure_environments.sh`

### Step 4: Initialize Safety Systems

See `setup/initialize_safety_systems.sh`

## Next Steps

1. Run setup scripts on both machines
2. Configure environment variables
3. Test safety systems
4. Begin development workflow

