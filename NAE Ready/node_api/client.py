#!/usr/bin/env python3
"""Node API Client"""

import os
import requests
import json

class NodeAPIClient:
    """Client for communicating with Master"""
    
    def __init__(self):
        self.master_url = os.getenv('NAE_MASTER_URL', 'http://localhost:8080')
        self.api_key = os.getenv('NAE_NODE_API_KEY', '')
        self.headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def report_status(self, status: dict):
        """Report node status to master"""
        try:
            response = requests.post(
                f"{self.master_url}/api/node/status",
                json=status,
                headers=self.headers,
                timeout=5
            )
            return response.json()
        except Exception as e:
            print(f"Error reporting status: {e}")
            return None
    
    def report_trade(self, trade: dict):
        """Report trade to master"""
        try:
            response = requests.post(
                f"{self.master_url}/api/trade/confirmation",
                json=trade,
                headers=self.headers,
                timeout=5
            )
            return response.json()
        except Exception as e:
            print(f"Error reporting trade: {e}")
            return None

if __name__ == '__main__':
    client = NodeAPIClient()
    status = client.report_status({"status": "online"})
    print(f"Status report: {status}")
