"""
Auto-implemented improvement from GitHub
Source: sohammirajkar/AI-Semiconductor-Alpha-Momentum-/asam_strategy.py
Implemented: 2025-12-09T11:02:27.842173
Usefulness Score: 100
Keywords: def , class , strategy, calculate, simulate, fit, loss, risk, sharpe, volatility, var, position, size, stop, loss
"""

# Original source: sohammirajkar/AI-Semiconductor-Alpha-Momentum-
# Path: asam_strategy.py


# Function: __init__
def __init__(self, finviz_api_key: str, quiver_api_key: Optional[str] = None):
        self.finviz_api_key = finviz_api_key
        self.quiver_api_key = quiver_api_key

        # Core semiconductor universe
        self.semiconductor_universe = [
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL', 
            'KLAC', 'LRCX', 'AMAT', 'MU', 'NXPI', 'MCHP', 'SWKS'
        ]

        # Supporting tech ecosystem
        self.tech_ecosystem = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'CRM', 'ORCL'
        ]

        # Industrial gas (LINDE thesis)
        self.industrial_gases = ['LIN', 'APD', 'ECL']

        # Risk management parameters
        self.max_position_size = 0.02  # 2% of portfolio per trade
        self.stop_loss_threshold = 0.025  # 2.5% stop loss
        self.take_profit_threshold = 0.06  # 6% take profit (2.4:1 R/R)
        self.max_holding_days = 5

        # Signal weights (sum to 1.0)
        self.momentum_weight = 0.40
        self.mean_reversion_weight = 0.30
        self.alternative_data_weight = 0.20
        self.macro_sentiment_weight = 0.10

        print("ASAM Strategy Initialized")
        print(f"Universe: {len(self.semiconductor_universe)} semiconductors + ecosystem")
        print(f"Risk Parameters: {self.max_position_size*100}% max position, {self.stop_loss_threshold*100}% stop")

    def fetch_market_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV daily bars from Polygon.io.
        period: "1y" (use 'from'/'to' for fine control, e.g. "2023-10-01" to "2024-10-01")
        """
        # Load API key from environment variable
        POLYGON_KEY = os.getenv("POLYGON_API_KEY")
        if not POLYGON_KEY:
            print("WARNING: POLYGON_API_KEY not found in environment variables. Falling back to yfinance.")
            return self._fetch_market_data_yfinance(symbols, period)
            
        try:
            client = RESTClient(api_key=POLYGON_KEY)
        except Exception as e:
            print(f"WARNING: Failed to initialize Polygon client: {e}. Falling back to yfinance.")
            return self._fetch_market_data_yfinance(symbols, period)
            
        data = {}

        end_date = datetime.now().date()
        if period.endswith("y"):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        else:
            # fallback default 1 year
            start_date = end_date - timedelta(days=365)

        for symbol in symbols:
            try:
                # Add 'X:' prefix for NASDAQ symbols if needed
                polygon_symbol = symbol if symbol.startswith('X:') or symbol.startswith('C:') else f"X:{symbol}"
                
                bars = []
                resp = client.list_aggs(
                    polygon_symbol,
                    1, "day",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=5000
                )
                
                # Convert response to list if it's a generator
                if hasattr(resp, '__iter__') and not isinstance(resp, (list, tuple)):
                    resp_list = list(resp)
                else:
                    resp_list = resp
                
                for bar in resp_list:
                    # Handle different response formats with proper type checking
                    try:
                        # Get attributes with fallbacks
                        timestamp = getattr(bar, 'timestamp', getattr(bar, 't', None))
                        open_price = getattr(bar, 'open', getattr(bar, 'o', None))
                        high_price = getattr(bar, 'high', getattr(bar, 'h', None))
                        low_price = getattr(bar, 'low', getattr(bar, 'l', None))
                        close_price = getattr(bar, 'close', getattr(bar, 'c', None))
                        volume = getattr(bar, 'volume', getattr(bar, 'v', None))
                        
                        # Validate all values are not None and are of correct types
                        if (timestamp is not None and 
                            open_price is not None and 
                            high_price is not None and 
                            low_price is not None and 
                            close_price is not None and 
                            volume is not None):
                            
                            # Convert to proper types
                            timestamp_int = int(timestamp)
                            open_float = float(open_price)
                            high_float = float(high_price)
                            low_float = float(low_price)
                            close_float = float(close_price)
                            volume_int = int(volume)
                            
                            bars.append({
                                "Date": pd.to_datetime(timestamp_int, unit="ms"),
                                "Open": open_float,
                                "High": high_float,
                                "Low": low_float,
                                "Close": close_float,
                                "Volume": volume_int,
                            })
                    except (ValueError, TypeError) as e:
                        # Skip malformed bars
                        continue
                    
                df = pd.DataFrame(bars)
                if not df.empty:
                    df.set_index("Date", inplace=True)
                    df["Symbol"] = symbol
                    data[symbol] = df
                    print(f"Fetched {len(df)} bars for {symbol}")
                else:
                    print(f"No data returned for {symbol}")
                    
            except Exception as e:
                print(f"Polygon error fetching {symbol}: {e}")
                # Try without prefix
                try:
                    bars = []
                    resp = client.list_aggs(
                        symbol,
                        1, "day",
                        from_=start_date.strftime("%Y-%m-%d"),
                        to=end_date.strftime("%Y-%m-%d"),
                        limit=5000
                    )
                    
                    # Convert response to list if it's a generator
                    if hasattr(resp, '__iter__') and not isinstance(resp, (list, tuple)):
                        resp_list = list(resp)
                    else:
                        resp_list = resp
                    
                    for bar in resp_list:
                        # Handle different response formats with proper type checking
                        try:
                            # Get attributes with fallbacks
                            timestamp = getattr(bar, 'timestamp', getattr(bar, 't', None))
                            open_price = getattr(bar, 'open', getattr(bar, 'o', None))
                            high_price = getattr(bar, 'high', getattr(bar, 'h', None))
                            low_price = getattr(bar, 'low', getattr(bar, 'l', None))
                            close_price = getattr(bar, 'close', getattr(bar, 'c', None))
                            volume = getattr(bar, 'volume', getattr(bar, 'v', None))
                            
                            # Validate all values are not None and are of correct types
                            if (timestamp is not None and 
                                open_price is not None and 
                                high_price is not None and 
                                low_price is not None and 
                                close_price is not None and 
                                volume is not None):
                                
                                # Convert to proper types
                                timestamp_int = int(timestamp)
                                open_float = float(open_price)
                                high_float = float(high_price)
                                low_float = float(low_price)
                                close_float = float(close_price)
                                volume_int = int(volume)
                                
                                bars.append({
                                    "Date": pd.to_datetime(timestamp_int, unit="ms"),
                                    "Open": open_float,
                                    "High": high_float,
                                    "Low": low_float,
                                    "Close": close_float,
                                    "Volume": volume_int,
                                })
                        except (ValueError, TypeError) as e:
                            # Skip malformed bars
                            continue
                            
                    df = pd.DataFrame(bars)
                    if not df.empty:
                        df.set_index("Date", inplace=True)
                        df["Symbol"] = symbol
                        data[symbol] = df
                        print(f"Fetched {len(df)} bars for {symbol} (no prefix)")
                    else:
                        print(f"No data returned for {symbol} (no prefix)")
                        
                except Exception as e2:
                    print(f"Polygon error fetching {symbol} (no prefix): {e2}")
                
        return data

    def _fetch_market_data_yfinance(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fallback method to fetch market data using yfinance.
        """
        print("Using yfinance as fallback...")
        data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    df['Symbol'] = symbol
                    data[symbol] = df
                    print(f"Fetched {len(df)} days for {symbol}")
                else:
                    print(f"No data returned for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return data

    def calculate_momentum_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate momentum factors for each symbol"""
        momentum_scores = []

        for symbol, df in data.items():
            if len(df) < 252:  # Need at least 1 year of data
                continue

            # Price momentum calculations
            returns_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1)
            returns_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) 
            returns_6m = (df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1)
            returns_12m = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1)

            # Relative strength vs NQ (NASDAQ-100)
            nq_data = yf.Ticker("^NDX").history(period="1y")
            nq_returns_3m = (nq_data['Close'].iloc[-1] / nq_data['Close'].iloc[-63] - 1)
            relative_strength = returns_3m - nq_returns_3m

            # Volatility-adjusted momentum
            volatility = df['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
            vol_adj_momentum = returns_3m / volatility if volatility > 0 else 0

            # Risk-adjusted momentum (Sharpe-like)
            daily_returns = df['Close'].pct_change().dropna()
            if len(daily_returns) > 60:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Fix for linter errors - use direct numpy array conversion
            volume_series = df['Volume'].rolling(20).mean()
            # Convert to numpy array and check for valid values
            volume_values = np.asarray(volume_series)
            # Filter out NaN values
            valid_volume_values = volume_values[~np.isnan(volume_values)]
            volume_avg_20d = valid_volume_values[-1] if len(valid_volume_values) > 0 else 0

            momentum_scores.append({
                'Symbol': symbol,
                'Returns_1M': returns_1m,
                'Returns_3M': returns_3m,
                'Returns_6M': returns_6m,
                'Returns_12M': returns_12m,
                'Relative_Strength': relative_strength,
                'Vol_Adj_Momentum': vol_adj_momentum,
                'Sharpe_Ratio': sharpe_ratio,
                'Current_Price': df['Close'].iloc[-1],
                'Volume_Avg_20D': volume_avg_20d
            })

        momentum_df = pd.DataFrame(momentum_scores)

        # Create composite momentum score
        momentum_df['Momentum_Score'] = (
            0.3 * momentum_df['Returns_3M'].rank(pct=True) +
            0.2 * momentum_df['Returns_6M'].rank(pct=True) +
            0.2 * momentum_df['Relative_Strength'].rank(pct=True) +
            0.2 * momentum_df['Vol_Adj_Momentum'].rank(pct=True) +
            0.1 * momentum_df['Sharpe_Ratio'].rank(pct=True)
        )

        return momentum_df.sort_values('Momentum_Score', ascending=False)

    def calculate_mean_reversion_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate mean reversion signals"""
        reversion_scores = []

        for symbol, df in data.items():
            if len(df) < 50:
                continue

            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Fix for linter errors - use direct numpy array conversion
            rsi_values = np.asarray(rsi)
            # Filter out NaN values
            valid_rsi_values = rsi_values[~np.isnan(rsi_values)]
            current_rsi = valid_rsi_values[-1] if len(valid_rsi_values) > 0 else 50

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_Middle'] = df['Close'].rolling(bb_period).mean()
            df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(bb_period).std() * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(bb_period).std() * bb_std)

            # Fix for linter errors - use direct numpy array conversion
            bb_lower_values = np.asarray(df['BB_Lower'])
            bb_upper_values = np.asarray(df['BB_Upper'])
            close_values = np.asarray(df['Close'])
            
            # Filter out NaN values
            valid_bb_lower = bb_lower_values[~np.isnan(bb_lower_values)]
            valid_bb_upper = bb_upper_values[~np.isnan(bb_upper_values)]
            valid_close = close_values[~np.isnan(close_values)]
            
            if (len(valid_bb_lower) > 0 and 
                len(valid_bb_upper) > 0 and 
                len(valid_close) > 0):
                bb_lower_val = valid_bb_lower[-1]
                bb_upper_val = valid_bb_upper[-1]
                close_val = valid_close[-1]
                if (bb_upper_val - bb_lower_val) != 0:
                    bb_position = (close_val - bb_lower_val) / (bb_upper_val - bb_lower_val)
                else:
                    bb_position = 0.5
            else:
                bb_position = 0.5

            # Volume-Price Analysis
            volume_sma = df['Volume'].rolling(20).mean()
            # Fix for linter errors - use direct numpy array conversion
            volume_values = np.asarray(df['Volume'])
            volume_sma_values = np.asarray(volume_sma)
            
            # Filter out NaN values
            valid_volume = volume_values[~np.isnan(volume_values)]
            valid_volume_sma = volume_sma_values[~np.isnan(volume_sma_values)]
            
            if (len(valid_volume_sma) > 0 and 
                len(valid_volume) > 0):
                volume_val = valid_volume[-1]
                volume_sma_val = valid_volume_sma[-1]
                volume_ratio = volume_val / volume_sma_val if volume_sma_val != 0 else 1
            else:
                volume_ratio = 1

            # Volatility expansion/contraction
            volatility_20d = df['Close'].pct_change().rolling(20).std()
            volatility_sma = volatility_20d.rolling(60).mean()
            # Fix for linter errors - use direct numpy array conversion
            volatility_values = np.asarray(volatility_20d)
            volatility_sma_values = np.asarray(volatility_sma)
            
            # Filter out NaN values
            valid_volatility = volatility_values[~np.isnan(volatility_values)]
            valid_volatility_sma = volatility_sma_values[~np.isnan(volatility_sma_values)]
            
            if (len(valid_volatility) > 0 and 
                len(valid_volatility_sma) > 0):
                vol_val = valid_volatility[-1]
                vol_sma_val = valid_volatility_sma[-1]
                volatility_ratio = vol_val / vol_sma_val if vol_sma_val != 0 else 1
            else:
                volatility_ratio = 1

            reversion_scores.append({
                'Symbol': symbol,
                'RSI': current_rsi,
                'BB_Position': bb_position,
                'Volume_Ratio': volume_ratio,
                'Volatility_Ratio': volatility_ratio,
                'Oversold_Signal': 1 if current_rsi < 30 else 0,
                'Overbought_Signal': 1 if current_rsi > 70 else 0
            })

        reversion_df = pd.DataFrame(reversion_scores)

        # Mean reversion composite score (higher = more oversold = buy signal)
        reversion_df['Mean_Reversion_Score'] = (
            0.4 * (100 - reversion_df['RSI']) / 100 +  # Inverted RSI
            0.3 * (1 - reversion_df['BB_Position']) +  # Lower BB position
            0.2 * np.minimum(reversion_df['Volume_Ratio'], 3) / 3 +  # Volume spike (capped)
            0.1 * np.minimum(reversion_df['Volatility_Ratio'], 2) / 2  # Volatility expansion
        )

        return reversion_df.sort_values('Mean_Reversion_Score', ascending=False)

    def fetch_alternative_data_signals(self) -> Dict[str, float]:
        """Fetch alternative data signals (simulated for demo)"""
        # In production, this would integrate with Quiver Quantitative API
        alt_signals = {}

        # Simulated congressional trading sentiment
        alt_signals['congressional_sentiment'] = np.random.normal(0.1, 0.3)  # Slight bullish bias

        # Simulated insider trading momentum  
        alt_signals['insider_momentum'] = np.random.normal(0.05, 0.2)

        # Simulated corporate jet activity (proxy for business confidence)
        alt_signals['corporate_activity'] = np.random.normal(0.02, 0.15)

        # Patent filing momentum (tech innovation proxy)
        alt_signals['patent_momentum'] = np.random.normal(0.08, 0.25)

        print("Alternative data signals fetched (simulated)")
        return alt_signals

    def calculate_macro_sentiment_score(self) -> float:
        """Calculate macro and sentiment factors"""
        try:
            # VIX for market fear
            vix = yf.Ticker("^VIX").history(period="1mo")
            current_vix = vix['Close'].iloc[-1]
            vix_score = max(0, (30 - current_vix) / 30)  # Lower VIX = higher score

            # Dollar strength (DXY) - negative for tech/semiconductors
            dxy = yf.Ticker("DX-Y.NYB").history(period="1mo")
            dxy_change = (dxy['Close'].iloc[-1] / dxy['Close'].iloc[-20] - 1)
            dollar_score = max(0, -dxy_change)  # Dollar weakness = positive for tech

            # Yield curve (10Y-2Y spread)
            ten_year = yf.Ticker("^TNX").history(period="1mo")['Close'].iloc[-1]
            two_year = yf.Ticker("^IRX").history(period="1mo")['Close'].iloc[-1] 
            yield_spread = ten_year - two_year
            curve_score = max(0, yield_spread / 2)  # Steeper curve = better growth outlook

            macro_score = (0.4 * vix_score + 0.4 * dollar_score + 0.2 * curve_score)

            print(f"Macro Score: {macro_score:.3f} (VIX: {current_vix:.1f}, DXY chg: {dxy_change:.2%})")
            return macro_score

        except Exception as e:
            print(f"Error calculating macro score: {e}")
            return 0.5  # Neutral score on error

    def generate_trading_signal(self) -> TradingSignal:
        """Generate final trading signal combining all factors"""
        print("\n" + "="*50)
        print("GENERATING TRADING SIGNAL")
        print("="*50)

        # Fetch market data
        all_symbols = self.semiconductor_universe + self.tech_ecosystem
        market_data = self.fetch_market_data(all_symbols, period="1y")

        # Check if we have any data
        if not market_data or all(len(df) == 0 for df in market_data.values()):
            print("WARNING: No market data available. Returning neutral signal.")
            return TradingSignal(
                timestamp=datetime.now(),
                signal=SignalStrength.NEUTRAL,
                confidence=0.0,
                expected_return=0.0,
                max_holding_period=0,
                stop_loss=0.0,
                take_profit=0.0
            )

        # Filter out empty dataframes
        market_data = {symbol: df for symbol, df in market_data.items() if len(df) > 0}

        if not market_data:
            print("WARNING: No valid market data after filtering. Returning neutral signal.")
            return TradingSignal(
                timestamp=datetime.now(),
                signal=SignalStrength.NEUTRAL,
                confidence=0.0,
                expected_return=0.0,
                max_holding_period=0,
                stop_loss=0.0,
                take_profit=0.0
            )

        # Calculate factor scores
        try:
            momentum_df = self.calculate_momentum_factors(market_data)
        except Exception as e:
            print(f"Warning: Error calculating momentum factors: {e}")
            momentum_df = pd.DataFrame()

        try:
            reversion_df = self.calculate_mean_reversion_factors(market_data)
        except Exception as e:
            print(f"Warning: Error calculating mean reversion factors: {e}")
            reversion_df = pd.DataFrame()

        try:
            alt_signals = self.fetch_alternative_data_signals()
        except Exception as e:
            print(f"Warning: Error fetching alternative data signals: {e}")
            alt_signals = {}

        try:
            macro_sentiment = self.calculate_macro_sentiment_score()
        except Exception as e:
            print(f"Warning: Error calculating macro sentiment score: {e}")
            macro_sentiment = 0.5

        # Handle case where we have no valid factor data
        if momentum_df.empty and reversion_df.empty:
            print("WARNING: No valid factor data. Returning neutral signal.")
            return TradingSignal(
                timestamp=datetime.now(),
                signal=SignalStrength.NEUTRAL,
                confidence=0.0,
                expected_return=0.0,
                max_holding_period=0,
                stop_loss=0.0,
                take_profit=0.0
            )

        # Combine signals for composite score
        # Use default values if dataframes are empty
        momentum_signal = momentum_df['Momentum_Score'].mean() if not momentum_df.empty and 'Momentum_Score' in momentum_df.columns else 0
        reversion_signal = reversion_df['Mean_Reversion_Score'].mean() if not reversion_df.empty and 'Mean_Reversion_Score' in reversion_df.columns else 0
        alt_data_signal = sum(alt_signals.values()) / len(alt_signals) if alt_signals else 0

        # Final composite signal
        composite_score = (
            self.momentum_weight * momentum_signal +
            self.mean_reversion_weight * reversion_signal +
            self.alternative_data_weight * alt_data_signal +
            self.macro_sentiment_weight * macro_sentiment
        )

        # Normalize to [-1, 1] range
        normalized_score = np.tanh(composite_score - 0.5)

        # Determine signal strength and confidence
        abs_score = abs(normalized_score)

        if abs_score > 0.6:
            signal_strength = SignalStrength.STRONG_BUY if normalized_score > 0 else SignalStrength.STRONG_SELL
            confidence = min(0.95, abs_score * 1.2)
        elif abs_score > 0.3:
            signal_strength = SignalStrength.BUY if normalized_score > 0 else SignalStrength.SELL
            confidence = abs_score * 0.8
        else:
            signal_strength = SignalStrength.NEUTRAL
            confidence = abs_score * 0.5

        # Expected return based on historical analysis and confidence
        expected_return = normalized_score * 0.15 * confidence  # Max 15% expected move

        # Risk management parameters
        if signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            stop_loss = -self.stop_loss_threshold
            take_profit = self.take_profit_threshold
            holding_period = min(self.max_holding_days, int(5 * confidence))
        elif signal_strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            stop_loss = self.stop_loss_threshold  # Reversed for short
            take_profit = -self.take_profit_threshold
            holding_period = min(self.max_holding_days, int(5 * confidence))
        else:
            stop_loss = 0.0
            take_profit = 0.0
            holding_period = 0

        # Generate final signal
        trading_signal = TradingSignal(
            timestamp=datetime.now(),
            signal=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            max_holding_period=holding_period,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Print signal summary
        print(f"\nFINAL SIGNAL SUMMARY:")
        print(f"Signal: {signal_strength.name}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Expected Return: {expected_return:.2%}")
        print(f"Max Holding: {holding_period} days")
        print(f"Stop Loss: {stop_loss:.2%}")
        print(f"Take Profit: {take_profit:.2%}")

        print(f"\nFACTOR BREAKDOWN:")
        print(f"Momentum: {momentum_signal:.3f} (weight: {self.momentum_weight})")
        print(f"Mean Reversion: {reversion_signal:.3f} (weight: {self.mean_reversion_weight})")  
        print(f"Alt Data: {alt_data_signal:.3f} (weight: {self.alternative_data_weight})")
        print(f"Macro/Sentiment: {macro_sentiment:.3f} (weight: {self.macro_sentiment_weight})")
        print(f"Composite Score: {composite_score:.3f}")
        print(f"Normalized Score: {normalized_score:.3f}")

        return trading_signal


# Example usage

