# Phisher Security Alerting System

## Overview

Phisher has been enhanced to automatically alert Bebop and Rocksteady when vulnerabilities or threats are detected, enabling immediate defensive action to secure the NAE system.

## Features

### 1. **Automatic Threat Detection**
- Scans logs for anomalous activity
- Audits critical code files for vulnerabilities
- Analyzes scan results for security issues
- Classifies threats by severity (critical, high, medium, low)

### 2. **Automatic Alerting**
- **Critical/High Threats**: Alerts both Bebop and Rocksteady immediately
- **Medium/Low Threats**: Alerts Bebop for monitoring
- Automatic threat classification and prioritization

### 3. **Bebop Response**
- Receives security alerts from Phisher
- Updates system status based on threat severity
- Escalates critical threats immediately
- Increases monitoring for high-priority threats
- Logs all alerts for tracking

### 3. **Rocksteady Response**
- Receives security threats from Phisher
- Takes immediate defensive action:
  - **Critical**: Blocks suspicious entities, enters defensive mode
  - **High**: Blocks entities, runs security sweep, enhanced security
  - **Medium**: Monitors and investigates
  - **Low**: Logs for review
- Generates security reports

### 4. **Casey Response**
- Receives security improvement requests from Phisher
- Analyzes vulnerabilities and creates improvement plans
- Generates security improvements:
  - **Enhance Phisher**: Add new detection patterns
  - **Enhance Bebop**: Add vulnerability-specific monitoring
  - **Enhance Rocksteady**: Add vulnerability-specific blocking rules
  - **Security Patches**: Create recommendations for affected files
- Applies improvements automatically
- Tracks all improvements for future reference

## Threat Severity Levels

| Severity | Response | Actions |
|----------|----------|---------|
| **Critical** | Immediate | All three agents alerted, entities blocked, defensive mode activated, improvements generated |
| **High** | Urgent | All three agents alerted, entities blocked, security sweep, improvements generated |
| **Medium** | Monitoring | Bebop and Casey alerted, improvements generated |
| **Low** | Logged | Bebop and Casey alerted, improvements logged |

## Threat Detection

### Code Scanning
- Scans critical files: `agents/optimus.py`, `agents/ralph.py`, `agents/donnie.py`, `agents/casey.py`, `nae_master_scheduler.py`
- Uses Bandit (if available) or heuristic scanning
- Detects:
  - SQL injection vulnerabilities
  - XSS vulnerabilities
  - Code injection patterns
  - Suspicious code patterns

### Log Scanning
- Scans all `.log` files in `logs/` directory
- Detects:
  - Error spikes
  - Unauthorized access attempts
  - Anomalous activity patterns
  - Security-related keywords

## Alert Flow

```
Phisher detects threat
    ↓
Analyzes severity and vulnerability type
    ↓
Critical/High → Alert Bebop + Rocksteady + Casey
Medium/Low → Alert Bebop + Casey
    ↓
Bebop: Updates status, escalates if needed
Rocksteady: Takes defensive action, blocks entities
Casey: Creates improvement plan, generates enhancements
```

## Usage

### Automatic Operation

The system runs automatically when Phisher executes its cycle (every hour):

```python
# Phisher automatically scans and alerts
phisher.run()
# Returns: {"threats_detected": N, "threats": [...]}
```

### Manual Alerting

```python
from agents.phisher import PhisherAgent
from agents.bebop import BebopAgent
from agents.rocksteady import RocksteadyAgent
from agents.casey import CaseyAgent

# Initialize agents
phisher = PhisherAgent()
bebop = BebopAgent()
rocksteady = RocksteadyAgent()
casey = CaseyAgent()

# Connect Phisher to alert targets
phisher.bebop_agent = bebop
phisher.rocksteady_agent = rocksteady
phisher.casey_agent = casey

# Detect and alert
threat = {
    "threat": "Security vulnerability detected",
    "severity": "high",
    "vulnerability_type": ["sql_injection"],
    "details": {"file": "agents/test.py", "issue": "SQL injection"},
    "action_required": "code_review_and_fix"
}
phisher.alert_security_team(threat)
```

## Integration with Master Scheduler

The master scheduler automatically connects Phisher to Bebop, Rocksteady, and Casey:

```python
# In nae_master_scheduler.py
phisher = PhisherAgent()
phisher.bebop_agent = bebop_agent
phisher.rocksteady_agent = rocksteady_agent
phisher.casey_agent = casey_agent
```

## Threat Response Actions

### Bebop Actions
- Updates `SecurityStatus` based on severity
- Escalates critical threats
- Increases monitoring frequency
- Logs all alerts

### Casey Actions
- **Critical/High**: Creates improvement plan, generates enhancements for all agents
- **Medium/Low**: Creates improvement plan, generates targeted enhancements
- Generates security patches for affected files
- Enhances detection patterns (Phisher)
- Enhances monitoring (Bebop)
- Enhances defenses (Rocksteady)

## Example Threat Detection

```python
# Phisher detects threat in code scan
threat = {
    "threat": "Security vulnerabilities found in agents/optimus.py",
    "severity": "high",
    "vulnerability_type": ["sql_injection", "xss"],
    "details": {
        "file": "agents/optimus.py",
        "issue_count": 5,
        "high_severity": 3
    },
    "action_required": "code_review_and_fix"
}

# Phisher automatically alerts all three agents
phisher.alert_security_team(threat)

# Bebop: Updates status to HIGH_THREAT_DETECTED
# Rocksteady: Blocks file, runs security sweep, enters HIGH_ALERT mode
# Casey: Creates improvement plan, generates enhancements for Phisher, Bebop, and Rocksteady
```

## Testing

Run the test script to verify alerting:

```bash
python3 test_security_alerting.py
```

This tests:
- ✓ Critical threat alerting
- ✓ High priority threat alerting
- ✓ Medium priority threat alerting
- ✓ Phisher scanning and detection
- ✓ Bebop alert handling
- ✓ Rocksteady threat response
- ✓ Casey improvement generation

## Security Benefits

1. **Immediate Response**: Threats detected and responded to automatically
2. **Multi-Layer Defense**: Monitoring (Bebop), enforcement (Rocksteady), and improvement (Casey)
3. **Severity-Based Actions**: Appropriate response based on threat level
4. **Continuous Improvement**: Casey generates improvements to prevent future vulnerabilities
5. **Continuous Monitoring**: Phisher runs every hour
6. **Automatic Blocking**: Suspicious entities blocked automatically
7. **Comprehensive Logging**: All threats, responses, and improvements logged

## Status Indicators

- **Bebop**: Updates `SecurityStatus` based on alerts
- **Rocksteady**: Changes status to:
  - `DEFENSIVE_MODE_ACTIVE` (critical)
  - `HIGH_ALERT` (high)
  - `MONITORING` (medium/low)
- **Casey**: Generates and tracks security improvements

## Next Steps

The NAE system now has:
- ✅ Automatic threat detection
- ✅ Automatic alerting to security team (Bebop, Rocksteady, Casey)
- ✅ Immediate defensive actions
- ✅ Automatic security improvements
- ✅ Continuous security monitoring
- ✅ Comprehensive threat response
- ✅ Proactive vulnerability prevention

The system is now better protected with automated threat detection, response, and continuous improvement!

