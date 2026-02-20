# NAE Test Suite

This directory contains all test files for the NAE system.

## Test Files

### Integration Tests
- `test_nae_system.py` - Comprehensive system integration tests
- `test_all_api_integrations.py` - API integration tests
- `test_api_keys.py` - API key validation tests

### Agent Tests
- `test_agent_automation.py` - Agent automation tests
- `test_phisher_enhanced.py` - Phisher security tests
- `test_security_alerting.py` - Security alerting tests

### Feature Tests
- `test_pnl_tracking.py` - P&L tracking tests
- `test_legal_compliance_integration.py` - Compliance tests
- `test_optimus_etrade_now.py` - E*Trade integration tests

### Other Tests
- `test_cloud_casey.py` - Cloud Casey deployment tests
- `test_genny_integration.py` - Genny integration tests
- And more...

## Running Tests

Run all tests:
```bash
python3 -m pytest tests/
```

Run specific test:
```bash
python3 tests/test_api_keys.py
```
