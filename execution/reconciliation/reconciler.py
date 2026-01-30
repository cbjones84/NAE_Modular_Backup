"""
Reconciliation System

Reconciles NAE ledger with broker positions and PnL.
"""

import os
import json
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ReconciliationEngine:
    """Reconciles positions and PnL between NAE and broker"""
    
    def __init__(self, redis_client: redis.Redis, postgres_conn):
        self.redis = redis_client
        self.postgres = postgres_conn
        self.discrepancy_threshold = float(os.getenv("RECONCILIATION_THRESHOLD", "0.01"))
    
    def reconcile_positions(self, broker: str = "schwab") -> Dict[str, Any]:
        """
        Reconcile positions between NAE ledger and broker
        
        Returns reconciliation report
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "broker": broker,
            "discrepancies": [],
            "matched": [],
            "status": "SUCCESS"
        }
        
        try:
            # Get NAE ledger positions
            nae_positions = self._get_nae_positions()
            
            # Get broker positions
            broker_positions = self._get_broker_positions(broker)
            
            # Compare positions
            for symbol, nae_pos in nae_positions.items():
                broker_pos = broker_positions.get(symbol, {"quantity": 0, "avg_price": 0})
                
                nae_qty = nae_pos.get("quantity", 0)
                broker_qty = broker_pos.get("quantity", 0)
                
                if abs(nae_qty - broker_qty) > self.discrepancy_threshold:
                    discrepancy = {
                        "symbol": symbol,
                        "nae_quantity": nae_qty,
                        "broker_quantity": broker_qty,
                        "difference": broker_qty - nae_qty,
                        "type": "quantity_mismatch"
                    }
                    result["discrepancies"].append(discrepancy)
                    logger.warning(f"Position discrepancy for {symbol}: NAE={nae_qty}, Broker={broker_qty}")
                else:
                    result["matched"].append({
                        "symbol": symbol,
                        "quantity": nae_qty
                    })
            
            # Check for broker positions not in NAE
            for symbol, broker_pos in broker_positions.items():
                if symbol not in nae_positions:
                    discrepancy = {
                        "symbol": symbol,
                        "nae_quantity": 0,
                        "broker_quantity": broker_pos.get("quantity", 0),
                        "difference": broker_pos.get("quantity", 0),
                        "type": "orphan_position"
                    }
                    result["discrepancies"].append(discrepancy)
                    logger.warning(f"Orphan position in broker: {symbol}")
            
            # Save reconciliation result
            self._save_reconciliation_result(result)
            
            if result["discrepancies"]:
                result["status"] = "DISCREPANCIES_FOUND"
            
            return result
        
        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")
            result["status"] = "ERROR"
            result["error"] = str(e)
            return result
    
    def reconcile_pnl(self, broker: str = "schwab", period: str = "daily") -> Dict[str, Any]:
        """
        Reconcile PnL between NAE and broker
        
        Args:
            broker: Broker name
            period: Reconciliation period (daily, weekly, monthly)
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "broker": broker,
            "period": period,
            "nae_pnl": 0.0,
            "broker_pnl": 0.0,
            "difference": 0.0,
            "status": "SUCCESS"
        }
        
        try:
            # Get NAE PnL
            nae_pnl = self._get_nae_pnl(period)
            result["nae_pnl"] = nae_pnl
            
            # Get broker PnL
            broker_pnl = self._get_broker_pnl(broker, period)
            result["broker_pnl"] = broker_pnl
            
            # Calculate difference
            difference = abs(nae_pnl - broker_pnl)
            result["difference"] = difference
            
            # Check threshold
            if difference > self.discrepancy_threshold * abs(nae_pnl) if nae_pnl != 0 else self.discrepancy_threshold:
                result["status"] = "DISCREPANCY"
                logger.warning(f"PnL discrepancy: NAE={nae_pnl}, Broker={broker_pnl}, Diff={difference}")
            else:
                result["status"] = "MATCHED"
            
            # Save reconciliation result
            self._save_reconciliation_result(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error reconciling PnL: {e}")
            result["status"] = "ERROR"
            result["error"] = str(e)
            return result
    
    def _get_nae_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get positions from NAE ledger"""
        cursor = self.postgres.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                SELECT symbol, SUM(quantity) as quantity, AVG(price) as avg_price
                FROM execution_ledger
                WHERE status = 'FILLED'
                GROUP BY symbol
            """)
            rows = cursor.fetchall()
            
            positions = {}
            for row in rows:
                positions[row['symbol']] = {
                    "quantity": float(row['quantity']),
                    "avg_price": float(row['avg_price'])
                }
            
            return positions
        finally:
            cursor.close()
    
    def _get_broker_positions(self, broker: str) -> Dict[str, Dict[str, Any]]:
        """Get positions from broker API"""
        # Would call broker API through adapter
        # For now, return empty dict (would be implemented with actual broker API)
        return {}
    
    def _get_nae_pnl(self, period: str) -> float:
        """Get PnL from NAE ledger"""
        cursor = self.postgres.cursor(cursor_factory=RealDictCursor)
        try:
            if period == "daily":
                start_date = datetime.now().date()
            elif period == "weekly":
                start_date = datetime.now().date() - timedelta(days=7)
            elif period == "monthly":
                start_date = datetime.now().date() - timedelta(days=30)
            else:
                start_date = datetime.now().date()
            
            cursor.execute("""
                SELECT SUM(realized_pnl) as total_pnl
                FROM execution_ledger
                WHERE DATE(executed_at) >= %s
            """, (start_date,))
            
            row = cursor.fetchone()
            return float(row['total_pnl']) if row and row['total_pnl'] else 0.0
        finally:
            cursor.close()
    
    def _get_broker_pnl(self, broker: str, period: str) -> float:
        """Get PnL from broker API"""
        # Would call broker API through adapter
        # For now, return 0.0 (would be implemented with actual broker API)
        return 0.0
    
    def _save_reconciliation_result(self, result: Dict[str, Any]):
        """Save reconciliation result to database"""
        cursor = self.postgres.cursor()
        try:
            cursor.execute("""
                INSERT INTO reconciliation_results (
                    timestamp, broker, type, status, result_data, created_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                result.get("timestamp"),
                result.get("broker"),
                result.get("period", "position"),
                result.get("status"),
                json.dumps(result)
            ))
            self.postgres.commit()
        except Exception as e:
            self.postgres.rollback()
            logger.error(f"Error saving reconciliation result: {e}")
        finally:
            cursor.close()


def get_reconciler(redis_client: redis.Redis, postgres_conn) -> ReconciliationEngine:
    """Get reconciliation engine instance"""
    return ReconciliationEngine(redis_client, postgres_conn)

