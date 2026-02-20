# Runtime Debugging Plan - Neural Agency Engine

**Date:** 2025-01-09  
**Instrumentation Added:** Yes  
**Log File:** `/Users/melissabishop/Downloads/Neural Agency Engine/.cursor/debug.log`

---

## Hypotheses for Runtime Issues

I've identified 5 key hypotheses about potential runtime bugs that need verification through runtime evidence:

### Hypothesis A: Redis Connection Failures
**Issue:** Redis connections may fail silently or not be properly initialized, causing kill switch to fall back to local state without proper error handling.

**Instrumentation Added:**
- Logs Redis initialization start (line 666)
- Logs Redis module availability check (line 670)
- Logs Redis connection creation (line 671)
- Logs kill switch setting (line 673)
- Logs initialization success/failure (line 674, 676)

**What to Look For:**
- Redis connection failures that are caught but not properly handled
- Kill switch state inconsistencies between Redis and local state
- Connection timeouts or network issues

### Hypothesis B: Position Synchronization Issues
**Issue:** Position synchronization from Tradier broker may fail, be incomplete, or create inconsistent state between broker and local tracking.

**Instrumentation Added:**
- Logs position sync start (line 679)
- Logs Tradier adapter availability (line 687)
- Logs position fetch results (line 688)
- Logs individual position syncs (line 704)
- Logs position updates (line 708)
- Logs sync errors (line 711, 713)

**What to Look For:**
- Positions not being synced from broker
- Position count mismatches
- Errors during position fetch or update
- Race conditions in position tracking

### Hypothesis C: Trade Execution Flow Issues
**Issue:** Trade execution may fail at various stages (pre-trade checks, execution routing, broker communication) without proper error recovery.

**Instrumentation Added:**
- Logs trade execution start (line 1101)
- Logs execution path selection (line 1688)
- Logs middleware errors and fallback (line 1727)
- Logs execution completion (line 1736)
- Logs successful execution (line 1837)
- Logs execution exceptions (line 1839)

**What to Look For:**
- Trades that start but don't complete
- Execution path selection issues
- Middleware failures and fallback behavior
- Exceptions during trade execution
- Execution time anomalies

### Hypothesis D: Error Swallowing
**Issue:** Critical errors may be caught in generic exception handlers and logged but not properly handled, leading to silent failures.

**Instrumentation Added:**
- Logs all exceptions in `execute_trade` with full traceback (line 1839)

**What to Look For:**
- Exceptions that are caught but execution continues
- Missing error context
- Errors that should stop execution but don't

### Hypothesis E: Kill Switch State Inconsistencies
**Issue:** Kill switch state may be inconsistent between Redis and local state, or Redis operations may fail silently.

**Instrumentation Added:**
- Logs kill switch check start (line 848)
- Logs Redis state retrieval (line 852)
- Logs Redis failures and fallback (line 856)

**What to Look For:**
- Kill switch state mismatches
- Redis get() failures
- Fallback to local state when Redis should be used

---

## Instrumentation Details

All instrumentation logs are written in NDJSON format to:
`/Users/melissabishop/Downloads/Neural Agency Engine/.cursor/debug.log`

**Log Format:**
```json
{
  "id": "unique_log_id",
  "timestamp": 1234567890000,
  "location": "file.py:line",
  "message": "Description",
  "data": {
    "hypothesisId": "A|B|C|D|E",
    "key": "value"
  },
  "sessionId": "debug-session",
  "runId": "run1"
}
```

**Hypothesis IDs:**
- **A**: Redis Connection Issues
- **B**: Position Synchronization
- **C**: Trade Execution Flow
- **D**: Error Handling
- **E**: Kill Switch State

---

## Next Steps: Reproduction Instructions

<reproduction_steps>
1. Ensure the NAE system is ready to run (check that required environment variables are set, especially TRADIER_API_KEY and TRADIER_ACCOUNT_ID if using Tradier)
2. Start the Optimus agent by running: `python "NAE Ready/agents/optimus.py"` or use your normal startup method
3. Let the system run for at least 5-10 minutes to capture various operations (initialization, position syncs, potential trades, etc.)
4. If possible, trigger a trade execution (either manually or wait for automatic trading if enabled)
5. Stop the system (Ctrl+C or normal shutdown)
6. Click "Proceed" in the debug UI to confirm completion
</reproduction_steps>

**Important Notes:**
- The log file will be created automatically when the first log entry is written
- If the system fails to start, the logs will still capture initialization errors
- All logs are wrapped in try/except to prevent instrumentation from breaking the system
- Logs are written in append mode, so multiple runs will accumulate (we'll clear before analysis)

---

## After Reproduction: Analysis Plan

Once you've run the system and confirmed completion:

1. **I will read the log file** and analyze each hypothesis:
   - **CONFIRMED**: Log evidence shows the issue occurs
   - **REJECTED**: Log evidence shows the issue does not occur
   - **INCONCLUSIVE**: Logs don't provide enough evidence

2. **For confirmed issues**, I will:
   - Identify the root cause from log evidence
   - Implement targeted fixes
   - Add additional instrumentation if needed
   - Verify fixes with another run

3. **For rejected hypotheses**, I will:
   - Generate new hypotheses based on actual log data
   - Add instrumentation for new areas of concern

4. **After fixes are verified**, I will:
   - Remove instrumentation (as requested)
   - Provide a summary of issues found and fixed

---

## Expected Log Patterns

### Healthy System:
- Redis initialization succeeds
- Position syncs complete with position counts
- Trade executions complete with status "submitted" or "filled"
- Kill switch checks return consistent state
- No exceptions in critical paths

### Problematic Patterns to Watch For:
- Redis connection errors followed by fallback to local state
- Position sync errors or zero positions when positions exist
- Trade executions that start but don't complete
- Exceptions in execute_trade that are caught but not handled
- Kill switch state mismatches between Redis and local

---

## Questions to Answer from Logs

1. **Redis (Hypothesis A):**
   - Does Redis initialize successfully?
   - Are there connection failures?
   - Does kill switch state sync properly?

2. **Position Sync (Hypothesis B):**
   - Are positions being fetched from Tradier?
   - Do position counts match expectations?
   - Are there errors during sync?

3. **Trade Execution (Hypothesis C):**
   - Do trades complete successfully?
   - Are there execution path selection issues?
   - Do middleware fallbacks work correctly?

4. **Error Handling (Hypothesis D):**
   - Are exceptions being caught and logged?
   - Do errors have sufficient context?
   - Are critical errors being ignored?

5. **Kill Switch (Hypothesis E):**
   - Is kill switch state consistent?
   - Are Redis failures handled gracefully?
   - Does fallback to local state work?

---

**Ready for reproduction!** Please follow the reproduction steps above.

