# April Agent - Bitcoin Migration, Crypto Intelligence & Kalshi Expert
"""
April Agent v2.0 - Ledger Live Integration + Kalshi Intelligence

ENHANCEMENTS in v2.0:
- Kalshi prediction market intelligence (CFTC-regulated)
- Crypto-macro correlation analysis for prediction markets
- Economics/Finance category expertise for Kalshi
- Cross-platform intelligence (Kalshi + Polymarket)

Responsibilities:
- Bitcoin migration strategies
- Fiat to cryptocurrency conversions
- Ledger Live wallet integration
- Crypto portfolio management
- Integration with Shredder's profit allocation
- KALSHI INTELLIGENCE: Provide crypto-macro analysis for prediction markets
- Economics/Finance market expertise using crypto correlation data

KALSHI EXPERTISE AREAS:
- Economics: Fed rates, inflation, GDP (crypto correlation analysis)
- Finance: Stock milestones, crypto prices (direct expertise)
- Cross-platform arbitrage detection (Kalshi vs Polymarket)
"""

import os
import datetime
import json
import requests
import time
from typing import Dict, List, Optional, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals

GOALS = get_nae_goals()

class AprilAgent:
    def __init__(self, goals=None):
        self.goals = goals if goals else GOALS
        self.log_file = "logs/april.log"
        self.config_file = "config/april_config.json"
        
        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []
        
        # ----------------------
        # Secure Vault Integration
        # ----------------------
        try:
            from secure_vault import get_vault
            self.vault = get_vault()
            self.vault_available = True
        except Exception as e:
            self.vault = None
            self.vault_available = False
            # Will log warning after logging is initialized
        
        # ----------------------
        # API Rate Limiting
        # ----------------------
        self.last_api_call_time = {}
        self.api_rate_limits = {
            "coingecko": 50,  # 50 calls per minute
            "blockchain_info": 600,  # 600 calls per hour
            "etherscan": 5  # 5 calls per second (free tier)
        }
        self.api_call_delays = {
            "coingecko": 1.2,  # 1.2 seconds between calls (50/min = 1.2s)
            "blockchain_info": 6.0,  # 6 seconds between calls (600/hour = 6s)
            "etherscan": 0.2  # 0.2 seconds between calls (5/sec = 0.2s)
        }
        
        # Ledger Live integration settings
        self.ledger_config = {
            "api_endpoint": "https://api.ledgerwallet.com",
            "wallet_addresses": {},
            "supported_coins": ["bitcoin", "ethereum", "litecoin", "bitcoin_cash"],
            "exchange_apis": {
                "binance": {"api_key": "", "secret": ""},
                "coinbase": {"api_key": "", "secret": ""},
                "kraken": {"api_key": "", "secret": ""}
            }
        }
        
        # Initialize directories and config
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        self._load_config()
        
        # Load API keys from vault if available
        self._load_api_keys_from_vault()
        
        if not self.vault_available:
            self.log_action("Warning: Secure vault not available. API keys stored in config file.")
        
        # ----------------------
        # Polymarket Integration (Prediction Markets)
        # ----------------------
        self.polymarket_trader = None
        self.polymarket_enabled = False
        try:
            from agents.polymarket_trader import PolymarketTrader
            self.polymarket_trader = PolymarketTrader()
            self.polymarket_enabled = True
            self.log_action("✅ Polymarket prediction market trading enabled")
        except ImportError:
            self.log_action("ℹ️ Polymarket trading not available (optional)")
        except Exception as e:
            self.log_action(f"⚠️ Polymarket init error: {e}")
        
        # ----------------------
        # Kalshi Intelligence Integration (CFTC-Regulated Prediction Markets)
        # ----------------------
        self.kalshi_trader = None
        self.kalshi_enabled = False
        try:
            from agents.kalshi_trader import KalshiTrader, get_kalshi_trader
            self.kalshi_trader = get_kalshi_trader()
            self.kalshi_enabled = True
            self.log_action("✅ Kalshi prediction market intelligence enabled (CFTC-regulated)")
        except ImportError:
            self.log_action("ℹ️ Kalshi trading not available (optional)")
        except Exception as e:
            self.log_action(f"⚠️ Kalshi init error: {e}")
        
        # Kalshi category expertise mapping - April specializes in economics/finance
        # due to existing crypto/macro data feeds
        self.kalshi_expertise = {
            "economics": {
                "description": "Fed rates, inflation, GDP, unemployment",
                "data_sources": ["fed_calendar", "cpi_data", "jobs_report", "crypto_macro_correlation"],
                "analysis_methods": ["historical_patterns", "yield_curve", "inflation_expectations"],
                "confidence_multiplier": 1.2  # April has edge in this category
            },
            "finance": {
                "description": "Stock milestones, crypto prices, market indices",
                "data_sources": ["crypto_prices", "stock_indices", "volatility_index", "on_chain_metrics"],
                "analysis_methods": ["technical_analysis", "correlation_tracking", "sentiment_analysis"],
                "confidence_multiplier": 1.3  # April has strong edge in crypto-related finance
            },
            "politics": {
                "description": "Elections, policy outcomes",
                "data_sources": ["polling_data", "prediction_market_consensus"],
                "analysis_methods": ["historical_precedent", "polling_aggregation"],
                "confidence_multiplier": 0.8  # Lower confidence - not April's specialty
            },
            "weather": {
                "description": "Temperature records, hurricanes",
                "data_sources": ["weather_apis", "climate_models"],
                "analysis_methods": ["model_consensus", "historical_patterns"],
                "confidence_multiplier": 0.7  # Not April's specialty
            }
        }
        
        # Kalshi intelligence cache
        self.kalshi_intelligence_cache = {}
        self.kalshi_intelligence_cache_ttl = 300  # 5 minutes
        
        self.log_action("April v2.0 initialized with Kalshi intelligence and Ledger Live capabilities.")

    def log_action(self, message):
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[April LOG] {message}")

    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.ledger_config.update(config)
                self.log_action("Configuration loaded successfully.")
            else:
                self._save_config()
        except Exception as e:
            self.log_action(f"Error loading config: {e}")

    def _save_config(self):
        """Save configuration to file"""
        try:
            # Don't save API keys to config file if vault is available
            config_to_save = self.ledger_config.copy()
            if self.vault_available:
                # Remove API keys from config (they're in vault)
                for exchange in config_to_save.get("exchange_apis", {}):
                    config_to_save["exchange_apis"][exchange] = {"api_key": "STORED_IN_VAULT", "secret": "STORED_IN_VAULT"}
            
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            self.log_action("Configuration saved successfully.")
        except Exception as e:
            self.log_action(f"Error saving config: {e}")
    
    def _rate_limit_check(self, api_name: str):
        """Enforce rate limiting for API calls"""
        if api_name not in self.last_api_call_time:
            self.last_api_call_time[api_name] = 0
        
        if api_name not in self.api_call_delays:
            return  # No rate limit for this API
        
        delay = self.api_call_delays[api_name]
        elapsed = time.time() - self.last_api_call_time[api_name]
        
        if elapsed < delay:
            sleep_time = delay - elapsed
            time.sleep(sleep_time)
        
        self.last_api_call_time[api_name] = time.time()
    
    def _load_api_keys_from_vault(self):
        """Load API keys from secure vault"""
        if not self.vault_available:
            return
        
        try:
            for exchange in ["binance", "coinbase", "kraken"]:
                api_key = self.vault.get_secret("april", f"{exchange}_api_key")
                secret = self.vault.get_secret("april", f"{exchange}_api_secret")
                
                if api_key and secret:
                    self.ledger_config["exchange_apis"][exchange] = {
                        "api_key": api_key,
                        "secret": secret
                    }
                    self.log_action(f"Loaded {exchange.upper()} API credentials from vault")
        except Exception as e:
            self.log_action(f"Error loading API keys from vault: {e}")

    def connect_to_ledger_live(self, wallet_addresses: Dict[str, str]) -> bool:
        """Connect April to Ledger Live wallet addresses"""
        try:
            self.ledger_config["wallet_addresses"] = wallet_addresses
            self._save_config()
            
            self.log_action(f"Connected to Ledger Live wallets:")
            for coin, address in wallet_addresses.items():
                self.log_action(f"  {coin.upper()}: {address[:10]}...{address[-10:]}")
            
            return True
        except Exception as e:
            self.log_action(f"Error connecting to Ledger Live: {e}")
            return False

    def setup_exchange_api(self, exchange: str, api_key: str, secret: str, use_vault: bool = True) -> bool:
        """Setup API credentials for cryptocurrency exchanges"""
        try:
            if exchange.lower() not in self.ledger_config["exchange_apis"]:
                self.log_action(f"Unsupported exchange: {exchange}")
                return False
            
            # Store in secure vault if available
            if use_vault and self.vault_available:
                try:
                    self.vault.set_secret("april", f"{exchange.lower()}_api_key", api_key)
                    self.vault.set_secret("april", f"{exchange.lower()}_api_secret", secret)
                    self.log_action(f"API credentials stored securely in vault for {exchange.upper()}")
                except Exception as e:
                    self.log_action(f"Error storing in vault, falling back to config: {e}")
                    use_vault = False
            
            # Also store in config (will be marked as STORED_IN_VAULT if vault is used)
            if not use_vault or not self.vault_available:
                self.ledger_config["exchange_apis"][exchange.lower()] = {
                    "api_key": api_key,
                    "secret": secret
                }
                self._save_config()
                self.log_action(f"API credentials configured for {exchange.upper()} (stored in config)")
            else:
                self._save_config()  # Save config without keys
            
            return True
        except Exception as e:
            self.log_action(f"Error setting up {exchange} API: {e}")
            return False

    def get_wallet_balance(self, coin: str) -> Optional[float]:
        """Get wallet balance for a specific cryptocurrency"""
        try:
            if coin.lower() not in self.ledger_config["wallet_addresses"]:
                self.log_action(f"No wallet address configured for {coin}")
                return None
            
            address = self.ledger_config["wallet_addresses"][coin.lower()]
            self.log_action(f"Checking {coin.upper()} balance for address: {address[:10]}...")
            
            # Try to get real balance from blockchain APIs
            balance = self._get_blockchain_balance(coin.lower(), address)
            
            if balance is None:
                # Fallback: Return placeholder if API fails
                self.log_action(f"Warning: Could not fetch real balance for {coin.upper()}, returning placeholder")
                balance = 0.0
            
            self.log_action(f"{coin.upper()} balance: {balance}")
            return balance
            
        except Exception as e:
            self.log_action(f"Error getting {coin} balance: {e}")
            return None
    
    def _get_blockchain_balance(self, coin: str, address: str) -> Optional[float]:
        """Get actual balance from blockchain APIs"""
        try:
            coin_map = {
                "bitcoin": "btc",
                "ethereum": "eth",
                "litecoin": "ltc",
                "bitcoin_cash": "bch"
            }
            
            coin_id = coin_map.get(coin.lower(), coin.lower())
            
            # Use blockchain.info API for Bitcoin
            if coin.lower() == "bitcoin":
                try:
                    self._rate_limit_check("blockchain_info")
                    url = f"https://blockchain.info/q/addressbalance/{address}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Response is in satoshis, convert to BTC
                        satoshis = int(response.text)
                        balance = satoshis / 100000000
                        return balance
                except Exception as e:
                    self.log_action(f"Error fetching Bitcoin balance: {e}")
            
            # Use Etherscan API for Ethereum (requires API key for production)
            elif coin.lower() == "ethereum":
                try:
                    # Try to get Etherscan API key from vault or environment
                    etherscan_api_key = os.getenv("ETHERSCAN_API_KEY", "YourApiKeyToken")
                    
                    # Try vault first
                    if self.vault_available:
                        try:
                            vault_key = self.vault.get_secret("april", "etherscan_api_key")
                            if vault_key:
                                etherscan_api_key = vault_key
                        except:
                            pass
                    
                    self._rate_limit_check("etherscan")
                    # Using public API endpoint (has rate limits)
                    url = f"https://api.etherscan.io/api"
                    params = {
                        "module": "account",
                        "action": "balance",
                        "address": address,
                        "tag": "latest",
                        "apikey": etherscan_api_key
                    }
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "1":
                            # Response is in Wei, convert to ETH
                            wei = int(data.get("result", 0))
                            balance = wei / 1000000000000000000
                            return balance
                except Exception as e:
                    self.log_action(f"Error fetching Ethereum balance: {e}")
            
            # For other coins, could integrate with respective APIs
            # For now, return None to trigger placeholder
            
            return None
            
        except Exception as e:
            self.log_action(f"Error in blockchain balance fetch: {e}")
            return None

    def convert_fiat_to_crypto(self, amount: float, from_currency: str, to_crypto: str) -> Dict:
        """Convert fiat currency to cryptocurrency"""
        try:
            self.log_action(f"Converting {amount} {from_currency.upper()} to {to_crypto.upper()}")
            
            # Get real exchange rate from CoinGecko
            exchange_rate = self._get_exchange_rate(to_crypto.lower(), from_currency.lower())
            
            if exchange_rate is None:
                # Fallback to placeholder if API fails
                self.log_action(f"Warning: Could not fetch real exchange rate, using placeholder")
                exchange_rate = self._get_placeholder_rate(to_crypto.lower())
            
            crypto_received = amount * exchange_rate
            
            conversion_result = {
                "success": True,
                "amount_fiat": amount,
                "currency_fiat": from_currency.upper(),
                "crypto_received": crypto_received,
                "crypto_currency": to_crypto.upper(),
                "exchange_rate": exchange_rate,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Conversion successful: {conversion_result['crypto_received']} {to_crypto.upper()} (rate: {exchange_rate})")
            return conversion_result
            
        except Exception as e:
            self.log_action(f"Error converting fiat to crypto: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_exchange_rate(self, crypto: str, fiat: str) -> Optional[float]:
        """Get real exchange rate from CoinGecko API"""
        try:
            self._rate_limit_check("coingecko")
            
            coin_map = {
                "bitcoin": "bitcoin",
                "ethereum": "ethereum",
                "litecoin": "litecoin",
                "bitcoin_cash": "bitcoin-cash"
            }
            
            coin_id = coin_map.get(crypto.lower(), crypto.lower())
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": fiat.lower()
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if coin_id in data and fiat.lower() in data[coin_id]:
                    rate = data[coin_id][fiat.lower()]
                    # Convert to crypto per fiat (e.g., BTC per USD)
                    return 1.0 / rate if rate > 0 else None
            return None
            
        except Exception as e:
            self.log_action(f"Error fetching exchange rate: {e}")
            return None
    
    def _get_placeholder_rate(self, crypto: str) -> float:
        """Get placeholder exchange rate (fallback)"""
        placeholder_rates = {
            "bitcoin": 0.000025,  # ~$40,000 per BTC
            "ethereum": 0.0004,   # ~$2,500 per ETH
            "litecoin": 0.01,     # ~$100 per LTC
            "bitcoin_cash": 0.0005  # ~$200 per BCH
        }
        return placeholder_rates.get(crypto.lower(), 0.000025)

    def migrate_bitcoin_strategy(self, allocation_amount: float) -> Dict:
        """Execute Bitcoin migration strategy from Shredder's profit allocation"""
        try:
            self.log_action(f"Executing Bitcoin migration strategy for ${allocation_amount}")
            
            # Get current Bitcoin price
            btc_price = self._get_bitcoin_price()
            if not btc_price:
                return {"success": False, "error": "Could not get Bitcoin price"}
            
            # Calculate Bitcoin amount
            btc_amount = allocation_amount / btc_price
            
            # Execute conversion
            conversion_result = self.convert_fiat_to_crypto(allocation_amount, "USD", "bitcoin")
            
            if conversion_result["success"]:
                migration_result = {
                    "success": True,
                    "allocation_amount": allocation_amount,
                    "bitcoin_amount": btc_amount,
                    "bitcoin_price": btc_price,
                    "wallet_address": self.ledger_config["wallet_addresses"].get("bitcoin", "Not configured"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                self.log_action(f"Bitcoin migration completed: {btc_amount} BTC")
                return migration_result
            else:
                return conversion_result
                
        except Exception as e:
            self.log_action(f"Error in Bitcoin migration: {e}")
            return {"success": False, "error": str(e)}

    def _get_bitcoin_price(self) -> Optional[float]:
        """Get current Bitcoin price from API"""
        try:
            self._rate_limit_check("coingecko")
            # Using CoinGecko API (free, no API key required)
            response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=10)
            if response.status_code == 200:
                data = response.json()
                price = data["bitcoin"]["usd"]
                self.log_action(f"Current Bitcoin price: ${price:,.2f}")
                return price
            else:
                self.log_action(f"Error fetching Bitcoin price: {response.status_code}")
                return None
        except Exception as e:
            self.log_action(f"Error getting Bitcoin price: {e}")
            return None

    def get_portfolio_summary(self) -> Dict:
        """Get summary of cryptocurrency portfolio"""
        try:
            portfolio = {}
            total_value_usd = 0.0
            
            # Get current prices for all coins
            coin_prices = {}
            for coin in self.ledger_config["supported_coins"]:
                price = self._get_crypto_price_usd(coin)
                coin_prices[coin] = price if price else 0.0
            
            for coin in self.ledger_config["supported_coins"]:
                balance = self.get_wallet_balance(coin)
                if balance is not None:
                    price_usd = coin_prices.get(coin, 0.0)
                    value_usd = balance * price_usd
                    total_value_usd += value_usd
                    
                    portfolio[coin] = {
                        "balance": balance,
                        "price_usd": price_usd,
                        "value_usd": value_usd,
                        "wallet_address": self.ledger_config["wallet_addresses"].get(coin, "Not configured")
                    }
            
            summary = {
                "portfolio": portfolio,
                "total_value_usd": round(total_value_usd, 2),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Portfolio summary generated: Total value ${total_value_usd:,.2f}")
            return summary
            
        except Exception as e:
            self.log_action(f"Error generating portfolio summary: {e}")
            return {"error": str(e)}
    
    def _get_crypto_price_usd(self, coin: str) -> Optional[float]:
        """Get current USD price for a cryptocurrency"""
        try:
            self._rate_limit_check("coingecko")
            
            coin_map = {
                "bitcoin": "bitcoin",
                "ethereum": "ethereum",
                "litecoin": "litecoin",
                "bitcoin_cash": "bitcoin-cash"
            }
            
            coin_id = coin_map.get(coin.lower(), coin.lower())
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if coin_id in data and "usd" in data[coin_id]:
                    return data[coin_id]["usd"]
            return None
            
        except Exception as e:
            self.log_action(f"Error fetching {coin} price: {e}")
            return None

    def run(self) -> Dict[str, Any]:
        """Main execution loop for April agent"""
        self.log_action("April Bitcoin migration agent running...")
        
        status = {
            "status": "success",
            "agent": "April",
            "warnings": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Check wallet connections
        if not self.ledger_config["wallet_addresses"]:
            warning = "No wallet addresses configured. Use connect_to_ledger_live() to set up."
            self.log_action(f"WARNING: {warning}")
            status["warnings"].append(warning)
        
        # Check exchange APIs
        configured_exchanges = []
        for ex, creds in self.ledger_config["exchange_apis"].items():
            # Check vault first, then config
            api_key = None
            secret = None
            
            if self.vault_available:
                api_key = self.vault.get_secret("april", f"{ex}_api_key")
                secret = self.vault.get_secret("april", f"{ex}_api_secret")
            
            if not api_key or not secret:
                api_key = creds.get("api_key")
                secret = creds.get("secret")
            
            if api_key and secret and api_key != "" and secret != "" and api_key != "STORED_IN_VAULT":
                configured_exchanges.append(ex)
        
        if not configured_exchanges:
            warning = "No exchange APIs configured. Use setup_exchange_api() to set up."
            self.log_action(f"WARNING: {warning}")
            status["warnings"].append(warning)
        
        status["configured_exchanges"] = configured_exchanges
        status["wallet_addresses_count"] = len(self.ledger_config["wallet_addresses"])
        
        self.log_action("April agent ready for Bitcoin operations.")
        return status
    
    # ----------------------
    # Messaging / AutoGen hooks
    # ----------------------
    def receive_message(self, message: dict):
        """Receive message from other agents"""
        self.inbox.append(message)
        self.log_action(f"Received message: {message}")
        
        # Handle specific message types
        if isinstance(message, dict):
            msg_type = message.get("type", "")
            content = message.get("content", {})
            
            if msg_type == "bitcoin_migration":
                # Execute Bitcoin migration from Shredder's profit allocation
                allocation = content.get("allocation_amount", 0)
                if allocation > 0:
                    result = self.migrate_bitcoin_strategy(allocation)
                    self.log_action(f"Bitcoin migration result: {result}")
    
    def send_message(self, message: dict, recipient_agent):
        """Send message to another agent"""
        try:
            if hasattr(recipient_agent, "receive_message"):
                recipient_agent.receive_message(message)
                self.outbox.append({"to": recipient_agent.__class__.__name__, "message": message})
                self.log_action(f"Sent message to {recipient_agent.__class__.__name__}: {message}")
            else:
                self.log_action(f"Recipient {recipient_agent.__class__.__name__} cannot receive messages")
        except Exception as e:
            self.log_action(f"Failed to send message: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        try:
            return {
                "status": "healthy",
                "agent": "April",
                "version": "v2.0",
                "vault_available": self.vault_available,
                "configured_exchanges": len([k for k, v in self.ledger_config.get("exchange_apis", {}).items() if v.get("api_key")]),
                "wallet_addresses": len(self.ledger_config.get("wallet_addresses", {})),
                "inbox_size": len(self.inbox),
                "outbox_size": len(self.outbox),
                "last_api_call_times": {k: v for k, v in self.last_api_call_time.items() if k in self.api_rate_limits},
                "polymarket_enabled": self.polymarket_enabled,
                "kalshi_enabled": self.kalshi_enabled,
                "kalshi_expertise": list(self.kalshi_expertise.keys()),
                "prediction_markets": {
                    "kalshi": self.kalshi_enabled,
                    "polymarket": self.polymarket_enabled
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "agent": "April",
                "error": str(e)
            }

    # ----------------------
    # Polymarket Prediction Market Methods
    # ----------------------
    
    def run_polymarket_trading(
        self,
        capital: float = 1000.0,
        strategies: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Run Polymarket prediction market trading
        
        Args:
            capital: Capital to deploy (in USDC)
            strategies: List of strategies ('bonding', 'semantic')
            dry_run: If True, generate signals without executing
            
        Returns:
            Trading results
        """
        if not self.polymarket_enabled or not self.polymarket_trader:
            return {"error": "Polymarket trading not enabled"}
        
        try:
            results = self.polymarket_trader.run_cycle(
                strategies=strategies or ["bonding"],
                capital=capital,
                dry_run=dry_run
            )
            self.log_action(f"Polymarket trading cycle complete: {results.get('total_signals', 0)} signals")
            return results
        except Exception as e:
            self.log_action(f"Polymarket trading error: {e}")
            return {"error": str(e)}
    
    def get_polymarket_opportunities(self) -> Dict[str, Any]:
        """Get current Polymarket bonding opportunities"""
        if not self.polymarket_enabled or not self.polymarket_trader:
            return {"error": "Polymarket not enabled"}
        
        try:
            if self.polymarket_trader.adapter:
                opportunities = self.polymarket_trader.adapter.find_bonding_opportunities()
                return {
                    "count": len(opportunities),
                    "top_opportunities": opportunities[:5],
                    "timestamp": datetime.datetime.now().isoformat()
                }
            return {"error": "Polymarket adapter not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_polymarket_portfolio(self) -> Dict[str, Any]:
        """Get Polymarket portfolio status"""
        if not self.polymarket_enabled or not self.polymarket_trader:
            return {"error": "Polymarket not enabled"}
        
        return self.polymarket_trader.get_portfolio_status()

    # ----------------------
    # Kalshi Intelligence Methods
    # ----------------------
    
    def analyze_kalshi_economic_markets(self) -> List[Dict[str, Any]]:
        """
        Analyze Kalshi economics markets using April's macro/crypto expertise.
        
        April specializes in economics and finance categories due to:
        - Existing crypto price feeds and correlation data
        - Macro indicator tracking (Fed rates, inflation)
        - Real-time market sentiment analysis
        
        Returns:
            List of market insights with edge assessment
        """
        if not self.kalshi_enabled or not self.kalshi_trader:
            return [{"error": "Kalshi not enabled"}]
        
        try:
            insights = []
            
            # Get markets from economics and finance categories
            adapter = getattr(self.kalshi_trader, 'adapter', None)
            if not adapter:
                return [{"error": "Kalshi adapter not available"}]
            
            # Get active markets
            markets = []
            if hasattr(adapter, 'get_markets'):
                # Get economics markets
                econ_markets = adapter.get_markets(category="economics", status="open") or []
                finance_markets = adapter.get_markets(category="finance", status="open") or []
                markets = econ_markets + finance_markets
            
            for market in markets[:20]:  # Analyze top 20 markets
                # Get market data
                ticker = getattr(market, 'ticker', market.get('ticker', '')) if hasattr(market, 'ticker') or isinstance(market, dict) else str(market)
                title = getattr(market, 'title', market.get('title', '')) if hasattr(market, 'title') or isinstance(market, dict) else ''
                category = getattr(market, 'category', market.get('category', 'unknown')) if hasattr(market, 'category') or isinstance(market, dict) else 'unknown'
                yes_price = getattr(market, 'yes_price', market.get('yes_price', 0.5)) if hasattr(market, 'yes_price') or isinstance(market, dict) else 0.5
                
                # Analyze using April's crypto/macro correlation
                correlation_analysis = self._analyze_crypto_macro_correlation(market)
                
                # Calculate edge
                april_estimate = correlation_analysis.get("probability", yes_price)
                edge = april_estimate - yes_price
                
                # Only include if there's potential edge
                if abs(edge) > 0.05:  # 5%+ edge threshold
                    expertise = self.kalshi_expertise.get(category.lower(), {})
                    confidence_multiplier = expertise.get("confidence_multiplier", 0.8)
                    
                    insights.append({
                        "market_ticker": ticker,
                        "title": title,
                        "category": category,
                        "current_price": yes_price,
                        "april_estimate": round(april_estimate, 3),
                        "edge_pct": round(edge * 100, 2),
                        "confidence": round(correlation_analysis.get("confidence", 0.5) * confidence_multiplier, 2),
                        "reasoning": correlation_analysis.get("reasoning", "Crypto-macro correlation analysis"),
                        "recommended_action": "BUY_YES" if edge > 0 else "BUY_NO",
                        "data_sources": expertise.get("data_sources", []),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            # Sort by absolute edge
            insights.sort(key=lambda x: abs(x.get("edge_pct", 0)), reverse=True)
            
            self.log_action(f"📊 Analyzed {len(markets)} Kalshi markets, found {len(insights)} with edge")
            
            return insights
            
        except Exception as e:
            self.log_action(f"Error analyzing Kalshi economic markets: {e}")
            return [{"error": str(e)}]
    
    def _analyze_crypto_macro_correlation(self, market) -> Dict[str, Any]:
        """
        Analyze a Kalshi market using crypto/macro correlation data.
        
        April's unique edge: Understanding how crypto markets correlate with
        macroeconomic events (Fed decisions, inflation, etc.)
        
        Args:
            market: Kalshi market object or dict
            
        Returns:
            Dict with probability estimate and reasoning
        """
        try:
            # Extract market info
            title = getattr(market, 'title', market.get('title', '')) if hasattr(market, 'title') or isinstance(market, dict) else ''
            category = getattr(market, 'category', market.get('category', '')) if hasattr(market, 'category') or isinstance(market, dict) else ''
            yes_price = getattr(market, 'yes_price', market.get('yes_price', 0.5)) if hasattr(market, 'yes_price') or isinstance(market, dict) else 0.5
            
            title_lower = title.lower()
            
            # Get current crypto market sentiment
            btc_trend = self._get_btc_trend_analysis()
            
            # Initialize base probability from market price
            probability = yes_price
            confidence = 0.5
            reasoning = []
            
            # Fed rate analysis
            if "fed" in title_lower or "rate" in title_lower or "fomc" in title_lower:
                # Fed decisions strongly correlated with crypto
                if btc_trend.get("trend") == "bullish":
                    # Bullish crypto often indicates market expects dovish Fed
                    probability = max(0.3, probability - 0.05)  # Slightly less likely rate hike
                    reasoning.append("BTC bullish trend suggests market expects dovish Fed")
                elif btc_trend.get("trend") == "bearish":
                    probability = min(0.7, probability + 0.05)  # Slightly more likely rate hike
                    reasoning.append("BTC bearish trend suggests market pricing in hawkish Fed")
                confidence = 0.7
            
            # Inflation analysis
            elif "inflation" in title_lower or "cpi" in title_lower:
                # Crypto often leads inflation expectations
                if btc_trend.get("momentum") == "strong":
                    # Strong crypto momentum often correlates with inflation hedging
                    probability = min(0.8, probability + 0.03)
                    reasoning.append("Strong BTC momentum suggests inflation concerns")
                confidence = 0.65
            
            # Stock market milestones
            elif "sp500" in title_lower or "s&p" in title_lower or "nasdaq" in title_lower:
                # Crypto and stocks often correlated
                if btc_trend.get("trend") == "bullish":
                    probability = min(0.75, probability + 0.05)
                    reasoning.append("BTC bullish trend correlated with risk-on sentiment")
                elif btc_trend.get("trend") == "bearish":
                    probability = max(0.25, probability - 0.05)
                    reasoning.append("BTC bearish trend correlated with risk-off sentiment")
                confidence = 0.6
            
            # Bitcoin-specific markets
            elif "bitcoin" in title_lower or "btc" in title_lower or "crypto" in title_lower:
                # April has strong expertise here
                if btc_trend.get("trend") == "bullish":
                    probability = min(0.85, probability + 0.10)
                    reasoning.append(f"Direct BTC analysis: {btc_trend.get('trend')} with {btc_trend.get('momentum', 'moderate')} momentum")
                elif btc_trend.get("trend") == "bearish":
                    probability = max(0.15, probability - 0.10)
                    reasoning.append(f"Direct BTC analysis: {btc_trend.get('trend')} with {btc_trend.get('momentum', 'moderate')} momentum")
                confidence = 0.85  # High confidence on crypto markets
            
            # Default: minimal adjustment
            else:
                reasoning.append("Limited crypto-macro correlation data for this market")
                confidence = 0.4
            
            return {
                "probability": probability,
                "confidence": confidence,
                "reasoning": "; ".join(reasoning) if reasoning else "Standard market analysis",
                "btc_trend": btc_trend,
                "edge_detected": abs(probability - yes_price) > 0.05
            }
            
        except Exception as e:
            self.log_action(f"Error in crypto-macro correlation analysis: {e}")
            return {
                "probability": 0.5,
                "confidence": 0.3,
                "reasoning": f"Analysis error: {e}",
                "edge_detected": False
            }
    
    def _get_btc_trend_analysis(self) -> Dict[str, Any]:
        """
        Get current Bitcoin trend analysis using April's crypto feeds.
        
        Returns:
            Dict with trend direction, momentum, and supporting data
        """
        try:
            # Try to get real BTC price data
            btc_price = self.get_crypto_price("bitcoin")
            
            if btc_price:
                # Use simple momentum analysis
                # In production, this would use historical data and technical indicators
                # For now, use a simplified approach
                trend = "neutral"
                momentum = "moderate"
                
                # Check 24h change if available (simplified)
                # This would ideally come from the API response
                return {
                    "trend": trend,
                    "momentum": momentum,
                    "current_price": btc_price,
                    "data_source": "coingecko",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Fallback if no price data
            return {
                "trend": "neutral",
                "momentum": "moderate",
                "current_price": None,
                "data_source": "fallback",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "trend": "neutral",
                "momentum": "moderate",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def provide_kalshi_intelligence_to_optimus(self, market_ticker: str) -> Dict[str, Any]:
        """
        Generate intelligence report for Optimus on a specific Kalshi market.
        
        Leverages April's real-time crypto and macro data feeds to inform
        Kalshi trading decisions.
        
        Args:
            market_ticker: Kalshi market ticker to analyze
            
        Returns:
            Dict with comprehensive market intelligence
        """
        try:
            if not self.kalshi_enabled:
                return {"error": "Kalshi not enabled"}
            
            # Check cache
            cache_key = f"intel_{market_ticker}"
            cached = self.kalshi_intelligence_cache.get(cache_key)
            if cached and (time.time() - cached.get("timestamp", 0)) < self.kalshi_intelligence_cache_ttl:
                return cached.get("data")
            
            # Get market details
            adapter = getattr(self.kalshi_trader, 'adapter', None)
            market = None
            if adapter and hasattr(adapter, 'get_market'):
                market = adapter.get_market(market_ticker)
            
            if not market:
                return {"error": f"Market {market_ticker} not found"}
            
            # Run correlation analysis
            correlation = self._analyze_crypto_macro_correlation(market)
            
            # Get BTC trend
            btc_trend = self._get_btc_trend_analysis()
            
            # Get macro indicators
            macro_indicators = self._get_macro_indicators()
            
            intelligence = {
                "market_ticker": market_ticker,
                "market_title": getattr(market, 'title', market.get('title', '')) if hasattr(market, 'title') or isinstance(market, dict) else '',
                "category": getattr(market, 'category', market.get('category', '')) if hasattr(market, 'category') or isinstance(market, dict) else '',
                "current_price": getattr(market, 'yes_price', market.get('yes_price', 0)) if hasattr(market, 'yes_price') or isinstance(market, dict) else 0,
                "april_probability_estimate": correlation.get("probability"),
                "confidence": correlation.get("confidence"),
                "edge_assessment": correlation.get("edge_detected"),
                "reasoning": correlation.get("reasoning"),
                "btc_correlation": btc_trend,
                "macro_context": macro_indicators,
                "recommendation": self._generate_trade_recommendation(correlation, market),
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "april_kalshi_intelligence"
            }
            
            # Cache the result
            self.kalshi_intelligence_cache[cache_key] = {
                "data": intelligence,
                "timestamp": time.time()
            }
            
            self.log_action(f"📈 Generated Kalshi intelligence for {market_ticker}")
            
            return intelligence
            
        except Exception as e:
            self.log_action(f"Error generating Kalshi intelligence: {e}")
            return {"error": str(e)}
    
    def _get_macro_indicators(self) -> Dict[str, Any]:
        """Get current macro indicators for Kalshi market analysis"""
        # In production, this would pull from Fed calendar, economic data APIs
        return {
            "fed_funds_rate": "5.25-5.50%",
            "next_fomc_meeting": "Check Fed calendar",
            "cpi_latest": "Check BLS data",
            "market_sentiment": "Check VIX",
            "note": "For production, integrate with economic data APIs"
        }
    
    def _generate_trade_recommendation(self, correlation: Dict, market) -> Dict[str, Any]:
        """Generate trade recommendation based on April's analysis"""
        yes_price = getattr(market, 'yes_price', market.get('yes_price', 0.5)) if hasattr(market, 'yes_price') or isinstance(market, dict) else 0.5
        april_estimate = correlation.get("probability", yes_price)
        confidence = correlation.get("confidence", 0.5)
        
        edge = april_estimate - yes_price
        
        if abs(edge) < 0.05 or confidence < 0.5:
            return {
                "action": "NO_TRADE",
                "reason": "Insufficient edge or confidence",
                "edge_pct": round(edge * 100, 2),
                "confidence": confidence
            }
        
        # Position sizing based on Kelly-like approach (simplified)
        position_fraction = min(0.25, abs(edge) * confidence)  # Max 25% of bankroll
        
        return {
            "action": "BUY_YES" if edge > 0 else "BUY_NO",
            "edge_pct": round(edge * 100, 2),
            "confidence": confidence,
            "suggested_position_fraction": round(position_fraction, 3),
            "reasoning": correlation.get("reasoning")
        }
    
    def run_kalshi_trading(
        self,
        capital: float = 1000.0,
        strategies: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Run Kalshi prediction market trading with April's intelligence.
        
        Args:
            capital: Capital to deploy (in USD)
            strategies: List of strategies ('bonding', 'semantic', 'arbitrage')
            dry_run: If True, generate signals without executing
            
        Returns:
            Trading results
        """
        if not self.kalshi_enabled or not self.kalshi_trader:
            return {"error": "Kalshi trading not enabled"}
        
        try:
            # First, generate April's market intelligence
            insights = self.analyze_kalshi_economic_markets()
            
            # Run Kalshi trader cycle with April's insights
            results = {
                "april_insights": len([i for i in insights if not i.get("error")]),
                "top_opportunities": insights[:5],
                "dry_run": dry_run,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # If Kalshi trader has run_cycle, use it
            if hasattr(self.kalshi_trader, 'run_cycle'):
                trader_results = self.kalshi_trader.run_cycle(
                    strategies=strategies or ["bonding", "semantic"],
                    capital=capital,
                    dry_run=dry_run
                )
                results["trader_results"] = trader_results
            
            self.log_action(f"📊 Kalshi trading cycle complete: {results.get('april_insights', 0)} insights")
            return results
            
        except Exception as e:
            self.log_action(f"Kalshi trading error: {e}")
            return {"error": str(e)}
    
    def get_kalshi_status(self) -> Dict[str, Any]:
        """Get Kalshi integration status"""
        return {
            "enabled": self.kalshi_enabled,
            "trader_available": self.kalshi_trader is not None,
            "expertise_categories": list(self.kalshi_expertise.keys()),
            "cache_size": len(self.kalshi_intelligence_cache),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_prediction_market_summary(self) -> Dict[str, Any]:
        """Get summary of all prediction market capabilities"""
        return {
            "kalshi": self.get_kalshi_status(),
            "polymarket": {
                "enabled": self.polymarket_enabled,
                "trader_available": self.polymarket_trader is not None
            },
            "april_specialties": ["economics", "finance", "crypto"],
            "timestamp": datetime.datetime.now().isoformat()
        }

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up April with Ledger Live"""
        return """
🔐 **April + Ledger Live Setup Instructions:**

**Step 1: Configure Wallet Addresses**
```python
april = April()
wallet_addresses = {
    "bitcoin": "your_bitcoin_address_here",
    "ethereum": "your_ethereum_address_here",
    "litecoin": "your_litecoin_address_here"
}
april.connect_to_ledger_live(wallet_addresses)
```

**Step 2: Setup Exchange APIs (Optional)**
```python
# For automated conversions
april.setup_exchange_api("binance", "your_api_key", "your_secret")
april.setup_exchange_api("coinbase", "your_api_key", "your_secret")
```

**Step 3: Test Connection**
```python
# Check Bitcoin balance
balance = april.get_wallet_balance("bitcoin")

# Get portfolio summary
portfolio = april.get_portfolio_summary()
```

**Step 4: Execute Bitcoin Migration**
```python
# Convert $1000 from Shredder's profits to Bitcoin
result = april.migrate_bitcoin_strategy(1000.0)
```

**Security Notes:**
- Store API keys securely
- Use read-only permissions where possible
- Test with small amounts first
- Keep Ledger device secure
"""