-- NAE Execution Database Schema

-- Signals table (audit trail)
CREATE TABLE IF NOT EXISTS signals_raw (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    quantity INTEGER,
    notional DECIMAL(20, 2),
    order_type VARCHAR(50) NOT NULL,
    limit_price DECIMAL(20, 2),
    stop_price DECIMAL(20, 2),
    risk_meta JSONB,
    correlation_group VARCHAR(255),
    model_id VARCHAR(255),
    confidence DECIMAL(5, 4),
    expected_pnl DECIMAL(20, 2),
    raw_payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_strategy_id (strategy_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_request_id (request_id)
);

-- Execution ledger
CREATE TABLE IF NOT EXISTS execution_ledger (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES signals_raw(id),
    request_id VARCHAR(255) NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    order_id VARCHAR(255),
    broker VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(20, 2) NOT NULL,
    fees DECIMAL(20, 2) DEFAULT 0,
    status VARCHAR(50) NOT NULL,
    realized_pnl DECIMAL(20, 2),
    submitted_at TIMESTAMP,
    executed_at TIMESTAMP,
    broker_balance_snapshot JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_signal_id (signal_id),
    INDEX idx_order_id (order_id),
    INDEX idx_strategy_id (strategy_id),
    INDEX idx_executed_at (executed_at)
);

-- Reconciliation results
CREATE TABLE IF NOT EXISTS reconciliation_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    broker VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_timestamp (timestamp),
    INDEX idx_broker (broker),
    INDEX idx_status (status)
);

-- Circuit breaker state
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    state VARCHAR(50) NOT NULL,
    failures INTEGER DEFAULT 0,
    last_failure_time TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW(),
    INDEX idx_name (name)
);

-- Strategy state
CREATE TABLE IF NOT EXISTS strategy_state (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(255) UNIQUE NOT NULL,
    paused BOOLEAN DEFAULT FALSE,
    exposure_limit DECIMAL(20, 2),
    current_exposure DECIMAL(20, 2) DEFAULT 0,
    risk_budget DECIMAL(20, 2),
    last_updated TIMESTAMP DEFAULT NOW(),
    INDEX idx_strategy_id (strategy_id)
);

-- OAuth tokens (encrypted)
CREATE TABLE IF NOT EXISTS oauth_tokens (
    id SERIAL PRIMARY KEY,
    broker VARCHAR(50) NOT NULL,
    account_id VARCHAR(255) NOT NULL,
    token_type VARCHAR(50) NOT NULL,
    access_token_encrypted TEXT NOT NULL,
    refresh_token_encrypted TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_refreshed TIMESTAMP,
    INDEX idx_broker_account (broker, account_id),
    INDEX idx_expires_at (expires_at)
);

-- Monitoring events
CREATE TABLE IF NOT EXISTS monitoring_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_created_at (created_at)
);

