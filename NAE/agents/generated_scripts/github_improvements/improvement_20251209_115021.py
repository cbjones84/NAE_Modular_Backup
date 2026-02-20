"""
Auto-implemented improvement from GitHub
Source: roboticslover/Risk_Management-Drawdown-Limiter-System-/main.py
Implemented: 2025-12-09T11:50:21.934965
Usefulness Score: 100
Keywords: def , strategy, backtest, calculate, simulate, model, loss, risk, sharpe, drawdown, volatility, position, size, stop, loss
"""

# Original source: roboticslover/Risk_Management-Drawdown-Limiter-System-
# Path: main.py


# Function: calculate_drawdowns
def calculate_drawdowns(data):
    """
    Calculate drawdowns from price data
    
    Args:
        data (pandas.DataFrame): Stock price data with 'Close' column
        
    Returns:
        pandas.DataFrame: DataFrame with drawdown analysis
    """
    if data is None or len(data) == 0:
        return None
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Calculate cumulative returns
    df['Cumulative Return'] = (1 + df['Return'].fillna(0)).cumprod()
    
    # Calculate running maximum
    df['Running Max'] = df['Cumulative Return'].cummax()
    
    # Calculate drawdown
    df['Drawdown'] = (df['Cumulative Return'] / df['Running Max'] - 1) * 100
    
    # Calculate drawdown duration
    df['Is Drawdown'] = df['Drawdown'] < 0
    df['Drawdown Group'] = (df['Is Drawdown'] != df['Is Drawdown'].shift(1)).cumsum()
    df['Drawdown Duration'] = df.groupby('Drawdown Group').cumcount()
    df.loc[~df['Is Drawdown'], 'Drawdown Duration'] = 0
    
    return df



# Function: identify_major_drawdowns
def identify_major_drawdowns(drawdown_data, threshold=-5):
    """
    Identify major drawdown periods
    
    Args:
        drawdown_data (pandas.DataFrame): Drawdown analysis data
        threshold (float): Threshold for considering a drawdown major (percentage)
        
    Returns:
        list: List of drawdown periods with start date, end date, magnitude
    """
    if drawdown_data is None or len(drawdown_data) == 0:
        return []
    
    # Find where drawdowns cross the threshold
    df = drawdown_data.copy()
    major_drawdowns = []
    
    # If no drawdowns exceed threshold, return empty list
    if not any(df['Drawdown'] < threshold):
        return major_drawdowns
    
    # Identify periods where drawdown is below threshold
    is_major_drawdown = df['Drawdown'] < threshold
    starts_drawdown = is_major_drawdown & ~is_major_drawdown.shift(1).fillna(False)
    ends_drawdown = ~is_major_drawdown & is_major_drawdown.shift(1).fillna(False)
    
    # Get the start and end dates
    drawdown_starts = df.index[starts_drawdown]
    drawdown_ends = df.index[ends_drawdown]
    
    # If we end in a drawdown period, add the last date as the end
    if len(drawdown_starts) > len(drawdown_ends):
        drawdown_ends = pd.Index([df.index[-1]]).append(drawdown_ends)
    
    # Create list of drawdown periods
    for i in range(min(len(drawdown_starts), len(drawdown_ends))):
        start_date = drawdown_starts[i]
        end_date = drawdown_ends[i]
        
        # Get period data
        period_data = df.loc[start_date:end_date]
        
        if len(period_data) == 0:
            continue
            
        # Find maximum drawdown in this period
        max_drawdown = period_data['Drawdown'].min()
        
        # Find the date of maximum drawdown (use idxmin for more robust approach)
        try:
            max_drawdown_date = period_data['Drawdown'].idxmin()
        except Exception:
            # Fallback to first date if there's an issue
            max_drawdown_date = start_date
        
        # Find recovery date
        recovery_date = None
        try:
            recovery_data = df.loc[max_drawdown_date:end_date]
            recovery_points = recovery_data[recovery_data['Drawdown'] >= 0]
            if not recovery_points.empty:
                recovery_date = recovery_points.index[0]
        except Exception:
            # If recovery calculation fails, set to None
            recovery_date = None
        
        # Calculate duration
        try:
            duration_days = (end_date - start_date).days
        except Exception:
            duration_days = 0
        
        major_drawdowns.append({
            'start_date': start_date,
            'max_drawdown_date': max_drawdown_date,
            'end_date': end_date,
            'recovery_date': recovery_date,
            'max_drawdown': max_drawdown,
            'duration_days': duration_days
        })
    
    return major_drawdowns



# Function: get_ai_insights
def get_ai_insights(ticker, drawdown_data, api_key, major_drawdowns):
    """
    Get AI insights about the drawdowns using OpenAI API via direct HTTP requests
    
    Args:
        ticker (str): Stock ticker symbol
        drawdown_data (pandas.DataFrame): Drawdown analysis data
        api_key (str): OpenAI API key
        major_drawdowns (list): List of major drawdown periods
        
    Returns:
        dict: AI insights and recommendations
    """
    if not api_key:
        return get_fallback_insights(ticker, drawdown_data, major_drawdowns)
    
    try:
        # Prepare data for AI analysis - with error handling
        recent_drawdown = drawdown_data['Drawdown'].iloc[-1] if len(drawdown_data) > 0 else 0
        max_historical_drawdown = drawdown_data['Drawdown'].min() if len(drawdown_data) > 0 else 0
        avg_drawdown = drawdown_data['Drawdown'].mean() if len(drawdown_data) > 0 else 0
        
        # Count how many drawdowns exceeded -10%
        severe_drawdowns = len([d for d in major_drawdowns if d['max_drawdown'] < -10])
        
        # Calculate average recovery time (in days)
        recovery_times = []
        for d in major_drawdowns:
            if d['recovery_date'] is not None and d['max_drawdown_date'] is not None:
                try:
                    recovery_time = (d['recovery_date'] - d['max_drawdown_date']).days
                    recovery_times.append(recovery_time)
                except Exception:
                    continue
        
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Prepare summary statistics for the prompt
        stats_summary = {
            'ticker': ticker,
            'current_drawdown': f"{recent_drawdown:.2f}%",
            'max_historical_drawdown': f"{max_historical_drawdown:.2f}%",
            'average_drawdown': f"{avg_drawdown:.2f}%",
            'number_of_major_drawdowns': len(major_drawdowns),
            'number_of_severe_drawdowns': severe_drawdowns,
            'average_recovery_time_days': int(avg_recovery_time)
        }
        
        # Prepare the top 3 worst drawdowns for the prompt
        worst_drawdowns = sorted(major_drawdowns, key=lambda x: x['max_drawdown'])[:3]
        worst_drawdowns_summary = []
        
        for i, d in enumerate(worst_drawdowns):
            try:
                worst_drawdowns_summary.append({
                    'rank': i+1,
                    'start_date': d['start_date'].strftime('%Y-%m-%d') if d['start_date'] else "Unknown",
                    'max_drawdown_date': d['max_drawdown_date'].strftime('%Y-%m-%d') if d['max_drawdown_date'] else "Unknown",
                    'end_date': d['end_date'].strftime('%Y-%m-%d') if d['end_date'] else "Unknown",
                    'recovery_date': d['recovery_date'].strftime('%Y-%m-%d') if d['recovery_date'] else "Not recovered",
                    'max_drawdown': f"{d['max_drawdown']:.2f}%",
                    'duration_days': d['duration_days']
                })
            except Exception:
                # Skip problematic entries
                continue
    
    except Exception as e:
        st.warning(f"Error preparing data for AI analysis: {str(e)}. Using fallback insights.")
        return get_fallback_insights(ticker, drawdown_data, major_drawdowns)
    
    # Create the prompt for OpenAI
    prompt = f"""
    Analyze the drawdown information for {ticker} and provide insights and recommendations for limiting future drawdowns:
    
    STATISTICS:
    {json.dumps(stats_summary, indent=2)}
    
    TOP 3 WORST DRAWDOWNS:
    {json.dumps(worst_drawdowns_summary, indent=2)}
    
    Please provide:
    1. Analysis of the historical drawdown patterns
    2. Current risk assessment based on recent data
    3. Three specific strategies to limit drawdowns for this stock
    4. Expected effectiveness of each strategy (high/medium/low)
    
    Format your response as JSON with the following structure:
    {{
        "analysis": "Your analysis of historical patterns...",
        "risk_assessment": "Current risk level and explanation...",
        "strategies": [
            {{
                "name": "Strategy name",
                "description": "Detailed description",
                "implementation": "How to implement it",
                "effectiveness": "high/medium/low"
            }},
            // more strategies...
        ]
    }}
    """
    
    # Prepare the direct HTTP request to OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {"role": "system", "content": "You are a financial analyst specializing in risk management and drawdown analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                insights = json.loads(content)
                return insights
            except json.JSONDecodeError:
                st.warning("AI response was not in valid JSON format. Using fallback insights.")
                return get_fallback_insights(ticker, drawdown_data, major_drawdowns)
        else:
            st.warning(f"OpenAI API request failed with status code {response.status_code}. Using fallback insights.")
            return get_fallback_insights(ticker, drawdown_data, major_drawdowns)
            
    except Exception as e:
        st.warning(f"Error communicating with OpenAI API: {str(e)}. Using fallback insights.")
        return get_fallback_insights(ticker, drawdown_data, major_drawdowns)



# Function: get_fallback_insights
def get_fallback_insights(ticker, drawdown_data, major_drawdowns):
    """
    Provide fallback insights when AI is unavailable
    
    Args:
        ticker (str): Stock ticker symbol
        drawdown_data (pandas.DataFrame): Drawdown analysis data
        major_drawdowns (list): List of major drawdown periods
        
    Returns:
        dict: Fallback insights and recommendations
    """
    try:
        # Safe data access with fallbacks
        recent_drawdown = drawdown_data['Drawdown'].iloc[-1] if len(drawdown_data) > 0 else 0
        max_historical_drawdown = drawdown_data['Drawdown'].min() if len(drawdown_data) > 0 else 0
        
        # Determine risk level based on recent and historical drawdowns
        if recent_drawdown < -15:
            risk_level = "High"
            risk_explanation = f"Current drawdown of {recent_drawdown:.2f}% represents significant risk."
        elif recent_drawdown < -8:
            risk_level = "Medium"
            risk_explanation = f"Current drawdown of {recent_drawdown:.2f}% represents moderate risk."
        else:
            risk_level = "Low"
            risk_explanation = f"Current drawdown of {recent_drawdown:.2f}% represents relatively low risk."
        
        # Additional context based on historical patterns
        if max_historical_drawdown < -30:
            historical_context = f"{ticker} has experienced severe drawdowns (up to {max_historical_drawdown:.2f}%) in the past, suggesting potential for significant volatility."
        elif max_historical_drawdown < -20:
            historical_context = f"{ticker} has experienced notable drawdowns (up to {max_historical_drawdown:.2f}%) in the past, indicating moderate historical volatility."
        else:
            historical_context = f"{ticker} has historically shown relatively contained drawdowns (max: {max_historical_drawdown:.2f}%), suggesting lower volatility compared to many stocks."
        
        # Create fallback strategies
        strategies = [
            {
                "name": "Stop-Loss Orders",
                "description": "Implement stop-loss orders to automatically sell the stock when it drops to a pre-determined price level.",
                "implementation": f"Consider setting stop-loss orders at 5-10% below your purchase price for {ticker}.",
                "effectiveness": "Medium" if risk_level == "Medium" else "High" if risk_level == "High" else "Low"
            },
            {
                "name": "Position Sizing",
                "description": "Limit the percentage of your portfolio allocated to this stock to control potential drawdowns.",
                "implementation": f"Based on historical volatility, consider limiting {ticker} to no more than {max(5, 20 - abs(int(max_historical_drawdown)))}% of your portfolio.",
                "effectiveness": "High"
            },
            {
                "name": "Hedging with Options",
                "description": "Use protective puts to limit downside while maintaining upside potential.",
                "implementation": f"Purchase put options for {ticker} with strikes 10-15% below current price and 3-6 month expirations.",
                "effectiveness": "High" if risk_level == "High" else "Medium"
            }
        ]
        
        return {
            "analysis": f"Analysis based on historical data: {historical_context} The stock has experienced {len(major_drawdowns)} major drawdown periods exceeding 5% in the analyzed timeframe.",
            "risk_assessment": f"Current Risk Level: {risk_level}. {risk_explanation}",
            "strategies": strategies
        }
        
    except Exception as e:
        # Ultimate fallback if everything fails
        return {
            "analysis": f"Basic analysis for {ticker}: Unable to perform detailed analysis due to data issues.",
            "risk_assessment": "Risk level: Unable to determine. Please check data quality and try again.",
            "strategies": [
                {
                    "name": "General Risk Management",
                    "description": "Use standard risk management practices for stock investments.",
                    "implementation": "Diversify your portfolio, set position limits, and use stop-losses.",
                    "effectiveness": "Medium"
                }
            ]
        }



# Function: simulate_drawdown_limiter
def simulate_drawdown_limiter(stock_data, threshold=-8, recovery_threshold=-5):
    """
    Simulate a drawdown limiter strategy
    
    Args:
        stock_data (pandas.DataFrame): Stock price data
        threshold (float): Drawdown threshold to trigger exit (percentage)
        recovery_threshold (float): Drawdown threshold to trigger re-entry (percentage)
        
    Returns:
        pandas.DataFrame: Simulation results
    """
    if stock_data is None or len(stock_data) == 0:
        return None
    
    # Calculate daily returns
    stock_data['Return'] = stock_data['Close'].pct_change()
    
    # Initialize simulation DataFrame
    sim = pd.DataFrame(index=stock_data.index)
    sim['Stock Price'] = stock_data['Close']
    sim['Stock Return'] = stock_data['Return']
    
    # Calculate cumulative returns for stock
    sim['Stock Cumulative'] = (1 + sim['Stock Return'].fillna(0)).cumprod()
    
    # Calculate running maximum
    sim['Running Max'] = sim['Stock Cumulative'].cummax()
    
    # Calculate drawdown
    sim['Drawdown'] = (sim['Stock Cumulative'] / sim['Running Max'] - 1) * 100
    
    # Initialize strategy signals
    sim['Position'] = 1  # Start invested
    sim['Signal'] = 0    # No initial signal
    
    # Generate signals based on drawdown threshold
    for i in range(1, len(sim)):
        if sim['Position'].iloc[i-1] == 1 and sim['Drawdown'].iloc[i] <= threshold:
            # Exit when drawdown exceeds threshold
            sim.loc[sim.index[i], 'Signal'] = -1
            sim.loc[sim.index[i], 'Position'] = 0
        elif sim['Position'].iloc[i-1] == 0 and sim['Drawdown'].iloc[i] >= recovery_threshold:
            # Re-enter when drawdown recovers to recovery threshold
            sim.loc[sim.index[i], 'Signal'] = 1
            sim.loc[sim.index[i], 'Position'] = 1
        else:
            # Maintain position
            sim.loc[sim.index[i], 'Position'] = sim['Position'].iloc[i-1]
    
    # Calculate strategy returns
    sim['Strategy Return'] = sim['Stock Return'] * sim['Position'].shift(1).fillna(0)
    
    # Calculate cumulative returns for strategy
    sim['Strategy Cumulative'] = (1 + sim['Strategy Return'].fillna(0)).cumprod()
    
    # Calculate maximum drawdown for strategy
    sim['Strategy Running Max'] = sim['Strategy Cumulative'].cummax()
    sim['Strategy Drawdown'] = (sim['Strategy Cumulative'] / sim['Strategy Running Max'] - 1) * 100
    
    return sim



# Function: create_performance_metrics
def create_performance_metrics(stock_data, simulation_data):
    """
    Calculate and format performance metrics for both strategies
    
    Args:
        stock_data (pandas.DataFrame): Stock price data
        simulation_data (pandas.DataFrame): Simulation results
        
    Returns:
        dict: Performance metrics
    """
    if stock_data is None or simulation_data is None:
        return None
    
    # Time period in years
    years = (stock_data.index[-1] - stock_data.index[0]).days / 365.25
    
    # Stock metrics
    stock_total_return = (simulation_data['Stock Cumulative'].iloc[-1] - 1) * 100
    stock_annualized_return = ((1 + stock_total_return/100) ** (1/years) - 1) * 100
    stock_volatility = simulation_data['Stock Return'].std() * (252 ** 0.5) * 100  # Annualized
    stock_max_drawdown = simulation_data['Drawdown'].min()
    stock_sharpe = stock_annualized_return / stock_volatility if stock_volatility > 0 else 0
    
    # Strategy metrics
    strategy_total_return = (simulation_data['Strategy Cumulative'].iloc[-1] - 1) * 100
    strategy_annualized_return = ((1 + strategy_total_return/100) ** (1/years) - 1) * 100
    strategy_volatility = simulation_data['Strategy Return'].std() * (252 ** 0.5) * 100  # Annualized
    strategy_max_drawdown = simulation_data['Strategy Drawdown'].min()
    strategy_sharpe = strategy_annualized_return / strategy_volatility if strategy_volatility > 0 else 0
    
    # Count trades
    num_trades = (simulation_data['Signal'] != 0).sum()
    
    return {
        'stock_metrics': {
            'total_return': f"{stock_total_return:.2f}%",
            'annualized_return': f"{stock_annualized_return:.2f}%",
            'volatility': f"{stock_volatility:.2f}%",
            'max_drawdown': f"{stock_max_drawdown:.2f}%",
            'sharpe_ratio': f"{stock_sharpe:.2f}"
        },
        'strategy_metrics': {
            'total_return': f"{strategy_total_return:.2f}%",
            'annualized_return': f"{strategy_annualized_return:.2f}%",
            'volatility': f"{strategy_volatility:.2f}%",
            'max_drawdown': f"{strategy_max_drawdown:.2f}%",
            'sharpe_ratio': f"{strategy_sharpe:.2f}",
            'num_trades': num_trades
        }
    }


