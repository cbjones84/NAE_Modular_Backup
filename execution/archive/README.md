# Execution Layer Archive

This folder contains deprecated or archived files from the execution layer.

## Files

### `lean_adapter_deprecated.py`
- **Status**: Deprecated
- **Reason**: Replaced by `lean_self_hosted.py`
- **Date**: 2024
- **Description**: Original QuantConnect Cloud adapter. This file used the QuantConnect Cloud API (`quantconnect.algorithm.QCAlgorithm`) and is no longer used. The current implementation uses self-hosted LEAN (`lean_self_hosted.py`) which provides full control over execution and broker adapters.

**Note**: This file is kept for reference only. Do not use in production.

