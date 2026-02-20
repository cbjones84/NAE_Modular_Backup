# Quick Start: Dual Machine Setup

## Mac (Development) Setup

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"

# 1. Configure environment
./setup/configure_environments.sh

# 2. Initialize safety systems
./setup/initialize_safety_systems.sh

# 3. Configure as Master
./setup/configure_master.sh

# 4. Create dev branch
git checkout -b dev
git push -u origin dev

# 5. Verify setup
python3 safety/production_guard.py
```

## HP OmniBook X (Production) Setup

```bash
cd "/path/to/NAE"

# 1. Clone repository
git clone https://github.com/cbjones84/NAE.git
cd NAE

# 2. Configure environment
./setup/configure_environments.sh

# 3. Initialize safety systems
./setup/initialize_safety_systems.sh

# 4. Configure as Node
./setup/configure_node.sh
# (Enter Master API key when prompted)

# 5. Create prod branch
git checkout -b prod
git push -u origin prod

# 6. Verify setup
python3 safety/production_guard.py
```

## Daily Workflow

### Mac (Development)
```bash
# Work on dev branch
git checkout dev
git pull origin dev

# Make changes, test
# ...

# Commit and push
git add .
git commit -m "Description"
git push origin dev

# When ready, merge to main
git checkout main
git merge dev
git push origin main
```

### HP OmniBook X (Production)
```bash
# Always on prod branch
git checkout prod
git pull origin main  # Pull from main (stable)

# Run production services
python3 nae_autonomous_master.py
```

## Safety Checks

Before any live trading:
1. Branch must be `prod`
2. Production lock file must exist
3. Device ID must match
4. Safety systems must be enabled

Run check:
```bash
python3 safety/production_guard.py
```

## Master-Node Communication

### Start Master (Mac)
```bash
python3 master_api/server.py
```

### Test Node Connection (HP)
```bash
python3 node_api/client.py
```

## Troubleshooting

### Branch Mismatch
```bash
git checkout prod  # On HP
git checkout dev   # On Mac
```

### Safety Check Failed
```bash
# Re-run initialization
./setup/initialize_safety_systems.sh
```

### API Connection Failed
- Check Master API is running
- Verify API key matches
- Check firewall settings

