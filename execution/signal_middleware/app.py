"""
NAE Signal Middleware - FastAPI Service

Receives trade signals from NAE, validates them, and queues for execution.
"""

import os
import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NAE Signal Middleware", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "nae_execution")
POSTGRES_USER = os.getenv("POSTGRES_USER", "nae")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
HMAC_SECRET = os.getenv("HMAC_SECRET", "").encode()  # Must be set in production
JWT_SECRET = os.getenv("JWT_SECRET", "")

# Initialize connections
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
postgres_pool = None


def get_postgres_conn():
    """Get PostgreSQL connection"""
    global postgres_pool
    if postgres_pool is None:
        postgres_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
    return postgres_pool.getconn()


def return_postgres_conn(conn):
    """Return PostgreSQL connection to pool"""
    postgres_pool.putconn(conn)


# Signal Schema
class RiskMeta(BaseModel):
    """Risk metadata"""
    max_slippage: float = Field(..., ge=0, le=1)
    max_exposure: float = Field(..., ge=0)
    max_position_pct: Optional[float] = Field(None, ge=0, le=1)


class Signal(BaseModel):
    """NAE trade signal"""
    strategy_id: str = Field(..., min_length=1)
    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    action: str = Field(..., regex="^(BUY|SELL|BUY_TO_OPEN|SELL_TO_OPEN|BUY_TO_CLOSE|SELL_TO_CLOSE)$")
    quantity: Optional[int] = Field(None, gt=0)
    notional: Optional[float] = Field(None, gt=0)
    order_type: str = Field(..., regex="^(MARKET|LIMIT|STOP|STOP_LIMIT)$")
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    risk_meta: RiskMeta
    correlation_group: Optional[str] = None
    request_id: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    expected_pnl: Optional[float] = None
    
    @validator('quantity', 'notional')
    def validate_quantity_or_notional(cls, v, values):
        """Ensure either quantity or notional is provided"""
        if 'quantity' in values and 'notional' in values:
            if values['quantity'] is None and values['notional'] is None:
                raise ValueError("Either quantity or notional must be provided")
        return v


class SignalResponse(BaseModel):
    """Signal submission response"""
    status: str
    request_id: str
    signal_id: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def verify_hmac_signature(payload: bytes, signature: str) -> bool:
    """Verify HMAC signature"""
    if not HMAC_SECRET:
        logger.warning("HMAC_SECRET not set, skipping signature verification")
        return True
    
    expected_signature = hmac.new(
        HMAC_SECRET,
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)


def save_signal_to_db(conn, signal: Signal, raw_payload: Dict[str, Any]) -> str:
    """Save signal to PostgreSQL for audit"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            INSERT INTO signals_raw (
                request_id, strategy_id, timestamp, symbol, action,
                quantity, notional, order_type, limit_price, stop_price,
                risk_meta, correlation_group, model_id, confidence,
                expected_pnl, raw_payload, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            ) RETURNING id
        """, (
            signal.request_id,
            signal.strategy_id,
            signal.timestamp,
            signal.symbol,
            signal.action,
            signal.quantity,
            signal.notional,
            signal.order_type,
            signal.limit_price,
            signal.stop_price,
            json.dumps(signal.risk_meta.dict()),
            signal.correlation_group,
            signal.model_id,
            signal.confidence,
            signal.expected_pnl,
            json.dumps(raw_payload),
        ))
        
        signal_id = cursor.fetchone()['id']
        conn.commit()
        return str(signal_id)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving signal to DB: {e}")
        raise
    finally:
        cursor.close()


def validate_signal(signal: Signal) -> Dict[str, Any]:
    """Run pre-trade validation checks"""
    validation_result = {
        "passed": True,
        "checks": [],
        "errors": []
    }
    
    # Get target broker from signal metadata or environment
    target_broker = os.getenv("PRIMARY_BROKER", "schwab")
    
    # Tradier-specific validation
    if target_broker == "tradier":
        try:
            from execution.pre_trade_validator.tradier_validator import TradierPreTradeValidator
            from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
            
            tradier_adapter = TradierBrokerAdapter(
                sandbox=os.getenv("TRADIER_SANDBOX", "true").lower() == "true"
            )
            tradier_validator = TradierPreTradeValidator(tradier_adapter)
            
            signal_dict = signal.dict()
            tradier_validation = tradier_validator.validate_order(signal_dict)
            
            validation_result["checks"].extend(tradier_validation.get("checks", []))
            validation_result["errors"].extend(tradier_validation.get("errors", []))
            validation_result["warnings"] = tradier_validation.get("warnings", [])
            validation_result["passed"] = tradier_validation.get("passed", True)
            
            if not validation_result["passed"]:
                return validation_result
        
        except Exception as e:
            logger.warning(f"Tradier validation failed, using standard validation: {e}")
    
    # Standard validation checks
    # Check trading hours (simplified - would use actual market calendar)
    current_hour = datetime.now().hour
    if current_hour < 9 or current_hour >= 16:
        validation_result["checks"].append({
            "check": "trading_hours",
            "passed": False,
            "message": "Outside trading hours"
        })
        validation_result["errors"].append("Outside trading hours")
        validation_result["passed"] = False
    
    # Check symbol validity (simplified - would validate against broker)
    if len(signal.symbol) < 1 or len(signal.symbol) > 10:
        validation_result["checks"].append({
            "check": "symbol_format",
            "passed": False,
            "message": "Invalid symbol format"
        })
        validation_result["errors"].append("Invalid symbol format")
        validation_result["passed"] = False
    
    # Check position sizing
    if signal.quantity and signal.quantity > 10000:
        validation_result["checks"].append({
            "check": "position_size",
            "passed": False,
            "message": "Position size exceeds limit"
        })
        validation_result["errors"].append("Position size exceeds limit")
        validation_result["passed"] = False
    
    if validation_result["passed"]:
        validation_result["checks"].append({
            "check": "all_checks",
            "passed": True,
            "message": "All validation checks passed"
        })
    
    return validation_result


def queue_signal(signal: Signal, signal_id: str) -> bool:
    """Queue validated signal for execution"""
    try:
        signal_data = {
            "signal_id": signal_id,
            "request_id": signal.request_id,
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "action": signal.action,
            "quantity": signal.quantity,
            "notional": signal.notional,
            "order_type": signal.order_type,
            "limit_price": signal.limit_price,
            "stop_price": signal.stop_price,
            "risk_meta": signal.risk_meta.dict(),
            "correlation_group": signal.correlation_group,
            "model_id": signal.model_id,
            "confidence": signal.confidence,
            "expected_pnl": signal.expected_pnl,
            "timestamp": signal.timestamp.isoformat(),
            "queued_at": datetime.now().isoformat()
        }
        
        # Push to Redis queue
        redis_client.lpush("execution.signals", json.dumps(signal_data))
        logger.info(f"Signal {signal_id} queued for execution")
        return True
    except Exception as e:
        logger.error(f"Error queueing signal: {e}")
        return False


@app.post("/v1/signals", response_model=SignalResponse)
async def receive_signal(
    signal: Signal,
    request: Request,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """
    Receive trade signal from NAE
    
    Validates signal, saves to audit DB, and queues for execution
    """
    try:
        # Verify HMAC signature
        body = await request.body()
        if x_signature and not verify_hmac_signature(body, x_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Get raw payload for audit
        raw_payload = await request.json()
        
        # Save to audit DB
        conn = get_postgres_conn()
        try:
            signal_id = save_signal_to_db(conn, signal, raw_payload)
        finally:
            return_postgres_conn(conn)
        
        # Validate signal
        validation_result = validate_signal(signal)
        
        if not validation_result["passed"]:
            return SignalResponse(
                status="REJECTED",
                request_id=signal.request_id,
                signal_id=signal_id,
                validation_result=validation_result,
                error="; ".join(validation_result["errors"])
            )
        
        # Queue for execution
        if queue_signal(signal, signal_id):
            return SignalResponse(
                status="ACCEPTED",
                request_id=signal.request_id,
                signal_id=signal_id,
                validation_result=validation_result
            )
        else:
            return SignalResponse(
                status="ERROR",
                request_id=signal.request_id,
                signal_id=signal_id,
                error="Failed to queue signal"
            )
    
    except Exception as e:
        logger.error(f"Error processing signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis
        redis_client.ping()
        redis_status = "ok"
    except:
        redis_status = "error"
    
    try:
        # Check PostgreSQL
        conn = get_postgres_conn()
        conn.close()
        return_postgres_conn(conn)
        postgres_status = "ok"
    except:
        postgres_status = "error"
    
    return {
        "status": "healthy" if redis_status == "ok" and postgres_status == "ok" else "degraded",
        "redis": redis_status,
        "postgres": postgres_status,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

