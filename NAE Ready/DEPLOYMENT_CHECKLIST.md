# Accelerator Strategy Deployment Checklist

## Pre-Deployment Verification

- [x] Target account size updated to $8000-$10000
- [x] Accelerator integrated into NAE master controller
- [x] Dual-mode operation (sandbox + live) implemented
- [x] Settlement cash tracking implemented
- [x] Ralph signal integration added
- [x] Risk management features implemented
- [x] Deployment script created
- [x] Documentation updated

## Deployment Steps

### 1. Verify Environment

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check required packages
pip3 list | grep -E "numpy|pandas|requests"

# Verify Tradier credentials are set
echo $TRADIER_API_KEY
echo $TRADIER_ACCOUNT_ID
```

### 2. Test in Sandbox First

```bash
cd "NAE Ready"

# Run sandbox only for initial testing
python3 execution/integration/accelerator_controller.py \
    --sandbox \
    --no-live \
    --interval 60
```

**Monitor for 1-2 hours** to verify:
- [ ] Ralph signals are generating correctly
- [ ] Settlement tracking is working
- [ ] No errors in logs
- [ ] Trades are executing properly

### 3. Deploy to Production

Once sandbox testing is successful:

```bash
cd "NAE Ready"

# Run deployment script
./scripts/deploy_accelerator.sh
```

This will:
1. Commit and push to GitHub
2. Start sandbox testing
3. Start live production
4. Start NAE master controller

### 4. Monitor Both Environments

```bash
# Monitor sandbox logs
tail -f logs/accelerator_sandbox.log

# Monitor live logs
tail -f logs/accelerator_live.log

# Monitor master controller
tail -f logs/nae_master.log
```

### 5. Verify Ralph Signals

Check that Ralph is generating signals correctly:

```python
from agents.ralph import RalphAgent

ralph = RalphAgent()
signal = ralph.get_intraday_direction_probability("SPY")
print(f"Prob Up: {signal['prob_up']:.2%}")
print(f"Prob Down: {signal['prob_down']:.2%}")
print(f"Confidence: {signal['confidence']:.2%}")
```

### 6. Check Settlement Tracking

Verify settlement ledger is working:

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)
if optimus.accelerator:
    status = optimus.accelerator.ledger.get_settlement_status()
    print(f"Settled Cash: ${status['settled_cash']:.2f}")
    print(f"Available: ${status['available_settled_cash']:.2f}")
```

## Post-Deployment Monitoring

### Daily Checks

- [ ] Check daily P&L for both sandbox and live
- [ ] Verify no settlement violations
- [ ] Monitor Ralph signal quality
- [ ] Check account balance growth
- [ ] Review error logs

### Weekly Reviews

- [ ] Calculate weekly returns (target: 4.3%)
- [ ] Review trade performance
- [ ] Adjust parameters if needed
- [ ] Check if target ($8000-$10000) reached

### When to Disable Accelerator

Disable automatically when:
- [ ] Account reaches $8000-$10000
- [ ] After 6 weeks of operation
- [ ] Daily drawdown limit hit 3+ times

## Troubleshooting

### "NO_SETTLED_FUNDS" Error

**Solution**: Wait for previous trades to settle (T+1 for options)

### "NO_HIGH_CONFIDENCE_SIGNAL" Error

**Solution**: Normal - strategy only trades when probability >70%. Check Ralph signal quality.

### "DAILY_DRAWDOWN_EXCEEDED" Error

**Solution**: Daily loss limit (-25%) hit. Trading stopped for safety. Review strategy performance.

### Process Not Starting

**Solution**: Check logs for errors. Verify Python path and dependencies.

## Performance Targets

- **Weekly Return**: 4.3% (aligned with generational wealth goal)
- **Target Account Size**: $8000-$10000
- **Max Daily Drawdown**: -25%
- **Win Rate**: Monitor and adjust if <40%

## Support

For issues:
1. Check logs: `logs/accelerator_*.log`
2. Review status: `optimus.accelerator.get_status()`
3. Check settlement: `optimus.accelerator.ledger.get_settlement_status()`

---

**Remember**: This is a temporary bootstrap strategy. Transition to main generational wealth strategies once target is reached.

