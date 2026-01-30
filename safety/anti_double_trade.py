#!/usr/bin/env python3
"""Anti-double-trade system"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class AntiDoubleTrade:
    """Prevents double trades across devices"""
    
    def __init__(self):
        self.safety_dir = Path(__file__).parent.parent / "safety"
        self.trade_log_file = self.safety_dir / "logs" / "trade_log.json"
        self.device_id = os.getenv("NAE_DEVICE_ID", "")
        self.time_window = int(os.getenv("NAE_TRADE_TIME_WINDOW", "60"))  # seconds
        
        # Ensure log file exists
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w') as f:
                json.dump({"trades": []}, f)
    
    def check_trade_allowed(self, trade_nonce: str) -> tuple[bool, str]:
        """Check if trade is allowed"""
        # Load trade log
        with open(self.trade_log_file, 'r') as f:
            log_data = json.load(f)
        
        trades = log_data.get("trades", [])
        current_time = time.time()
        
        # Check for recent trades
        recent_trades = [
            t for t in trades
            if current_time - t.get("timestamp", 0) < self.time_window
        ]
        
        # Check for duplicate nonce
        for trade in trades:
            if trade.get("nonce") == trade_nonce:
                return False, f"Duplicate trade nonce: {trade_nonce}"
        
        # Check for trades from different device
        for trade in recent_trades:
            if trade.get("device_id") != self.device_id:
                return False, f"Recent trade from different device: {trade.get('device_id')}"
        
        return True, "Trade allowed"
    
    def log_trade(self, trade_nonce: str, trade_details: dict):
        """Log a trade"""
        with open(self.trade_log_file, 'r') as f:
            log_data = json.load(f)
        
        trade_entry = {
            "nonce": trade_nonce,
            "device_id": self.device_id,
            "timestamp": time.time(),
            "details": trade_details
        }
        
        log_data["trades"].append(trade_entry)
        
        # Keep only last 1000 trades
        log_data["trades"] = log_data["trades"][-1000:]
        
        with open(self.trade_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def generate_nonce(self) -> str:
        """Generate a unique trade nonce"""
        import hashlib
        import secrets
        
        nonce_data = f"{self.device_id}:{time.time()}:{secrets.token_hex(16)}"
        return hashlib.sha256(nonce_data.encode()).hexdigest()[:32]

if __name__ == '__main__':
    anti_double = AntiDoubleTrade()
    nonce = anti_double.generate_nonce()
    allowed, message = anti_double.check_trade_allowed(nonce)
    print(f"Trade allowed: {allowed}")
    print(f"Message: {message}")
    print(f"Nonce: {nonce}")
