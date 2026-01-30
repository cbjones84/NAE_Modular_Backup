#!/usr/bin/env python3
"""
API Integrations for Trading Strategy Ingestion
Integrates with Twitter, Reddit, Discord, and other trading platforms
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from ratelimit import limits, sleep_and_retry
from cachetools import TTLCache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APISource:
    """Configuration for API sources"""
    name: str
    api_key: str
    api_secret: str = None
    base_url: str = None
    rate_limit: int = 100
    enabled: bool = True

class TwitterAPI:
    """Twitter API integration for trading insights"""
    
    def __init__(self, config: APISource):
        self.config = config
        self.cache = TTLCache(maxsize=500, ttl=1800)  # 30 minute cache
        
    def initialize_client(self):
        """Initialize Twitter API client"""
        try:
            import tweepy
            
            if not self.config.api_key:
                logger.warning("Twitter API key not configured")
                return None
                
            client = tweepy.Client(
                bearer_token=self.config.api_key,
                wait_on_rate_limit=True
            )
            return client
        except ImportError:
            logger.error("tweepy not installed - pip install tweepy")
            return None
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            return None
    
    @sleep_and_retry
    @limits(calls=10, period=60)  # 10 calls per minute
    def get_trading_tweets(self, hashtags: List[str] = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get trading-related tweets with enhanced analysis"""
        client = self.initialize_client()
        if not client:
            return []
        
        hashtags = hashtags or ['#OptionsTrading', '#TradingStrategy', '#OptionsStrategy', '#ThetaGang', '#SPY', '#QQQ']
        tweets = []
        
        try:
            for hashtag in hashtags:
                # Enhanced query for better trading content
                enhanced_query = f"{hashtag} -is:retweet lang:en"
                
                response = client.search_recent_tweets(
                    query=enhanced_query,
                    max_results=min(max_results, 20),
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'entities'],
                    expansions=['author_id'],
                    user_fields=['username', 'verified', 'public_metrics']
                )
                
                if response.data:
                    for tweet in response.data:
                        # Extract trading strategy mentions
                        strategy_indicators = [
                            'strategy', 'trade', 'position', 'call', 'put', 'option',
                            'bullish', 'bearish', 'long', 'short', 'spread', 'straddle',
                            'earnings', 'volatility', 'theta', 'delta', 'gamma', 'vega'
                        ]
                        
                        text_lower = tweet.text.lower()
                        if any(indicator in text_lower for indicator in strategy_indicators):
                            tweet_data = {
                                'id': tweet.id,
                                'text': tweet.text,
                                'author_id': tweet.author_id,
                                'created_at': tweet.created_at.isoformat(),
                                'like_count': tweet.public_metrics['like_count'],
                                'retweet_count': tweet.public_metrics['retweet_count'],
                                'reply_count': tweet.public_metrics['reply_count'],
                                'quote_count': tweet.public_metrics['quote_count'],
                                'hashtag': hashtag,
                                'source': 'twitter',
                                'url': f"https://twitter.com/user/status/{tweet.id}",
                                'engagement_score': self._calculate_twitter_engagement(tweet.public_metrics),
                                'strategy_type': self._classify_twitter_strategy(tweet.text),
                                'sentiment': self._analyze_twitter_sentiment(tweet.text),
                                'name': f"Twitter Strategy: {tweet.text[:50]}...",
                                'details': tweet.text
                            }
                            tweets.append(tweet_data)
            
            logger.info(f"Retrieved {len(tweets)} trading strategies from Twitter")
            return tweets
            
        except Exception as e:
            logger.error(f"Error fetching Twitter tweets: {e}")
            return []
    
    def _calculate_twitter_engagement(self, metrics: Dict[str, int]) -> float:
        """Calculate engagement score for Twitter content"""
        if not metrics:
            return 0.0
        
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        
        # Weighted engagement score
        engagement = (likes * 1.0) + (retweets * 2.0) + (replies * 3.0)
        return min(engagement / 100.0, 10.0)  # Normalize to 0-10 scale
    
    def _classify_twitter_strategy(self, text: str) -> str:
        """Classify Twitter strategy type"""
        text_lower = text.lower()
        
        if 'call' in text_lower and 'put' in text_lower:
            return 'spread'
        elif 'call' in text_lower:
            return 'call'
        elif 'put' in text_lower:
            return 'put'
        elif 'straddle' in text_lower or 'strangle' in text_lower:
            return 'straddle'
        elif 'earnings' in text_lower:
            return 'earnings'
        elif 'volatility' in text_lower:
            return 'volatility'
        else:
            return 'general'
    
    def _analyze_twitter_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for Twitter content"""
        positive_words = ['bullish', 'long', 'buy', 'profit', 'gain', 'up', 'rise', 'strong']
        negative_words = ['bearish', 'short', 'sell', 'loss', 'down', 'fall', 'weak', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_user_timeline(self, username: str, max_tweets: int = 20) -> List[Dict[str, Any]]:
        """Get tweets from a specific user's timeline"""
        client = self.initialize_client()
        if not client:
            return []
        
        try:
            user = client.get_user(username=username)
            if not user.data:
                return []
            
            tweets = client.get_users_tweets(
                id=user.data.id,
                max_results=min(max_tweets, 20),
                tweet_fields=['created_at', 'public_metrics']
            )
            
            tweet_data = []
            if tweets.data:
                for tweet in tweets.data:
                    tweet_data.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'author': username,
                        'created_at': tweet.created_at.isoformat(),
                        'like_count': tweet.public_metrics['like_count'],
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'source': 'twitter'
                    })
            
            logger.info(f"Retrieved {len(tweet_data)} tweets from @{username}")
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error fetching user timeline for @{username}: {e}")
            return []

class RedditAPI:
    """Reddit API integration for trading discussions"""
    
    def __init__(self, config: APISource):
        self.config = config
        self.cache = TTLCache(maxsize=300, ttl=1800)  # 30 minute cache
        
    def initialize_client(self):
        """Initialize Reddit API client"""
        try:
            import praw
            
            if not self.config.api_key or not self.config.api_secret:
                logger.warning("Reddit API credentials not configured")
                return None
                
            reddit = praw.Reddit(
                client_id=self.config.api_key,
                client_secret=self.config.api_secret,
                user_agent="NAE Trading Bot 1.0"
            )
            return reddit
        except ImportError:
            logger.error("praw not installed - pip install praw")
            return None
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            return None
    
    @sleep_and_retry
    @limits(calls=5, period=60)  # 5 calls per minute
    def get_trading_posts(self, subreddits: List[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Get trading-related posts from Reddit with enhanced analysis"""
        reddit = self.initialize_client()
        if not reddit:
            return []
        
        subreddits = subreddits or ['options', 'wallstreetbets', 'investing', 'thetagang', 'SecurityAnalysis']
        posts = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                
                for post in subreddit.hot(limit=limit):
                    if self._is_trading_related(post):
                        post_data = {
                            'id': post.id,
                            'title': post.title,
                            'content': post.selftext,
                            'author': str(post.author),
                            'subreddit': subreddit_name,
                            'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio,
                            'url': f"https://reddit.com{post.permalink}",
                            'source': 'reddit',
                            'engagement_score': self._calculate_reddit_engagement(post),
                            'strategy_type': self._classify_reddit_strategy(post.title + ' ' + post.selftext),
                            'sentiment': self._analyze_reddit_sentiment(post.title + ' ' + post.selftext),
                            'name': f"Reddit Strategy: {post.title[:50]}...",
                            'details': f"{post.title}\n\n{post.selftext}",
                            'upvotes': post.score,
                            'comments': post.num_comments
                        }
                        posts.append(post_data)
            
            logger.info(f"Retrieved {len(posts)} trading strategies from Reddit")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    def _calculate_reddit_engagement(self, post) -> float:
        """Calculate engagement score for Reddit posts"""
        score = post.score
        comments = post.num_comments
        upvote_ratio = post.upvote_ratio
        
        # Weighted engagement score
        engagement = (score * 0.5) + (comments * 2.0) + (upvote_ratio * 50.0)
        return min(engagement / 100.0, 10.0)  # Normalize to 0-10 scale
    
    def _classify_reddit_strategy(self, text: str) -> str:
        """Classify Reddit strategy type"""
        text_lower = text.lower()
        
        if 'call' in text_lower and 'put' in text_lower:
            return 'spread'
        elif 'call' in text_lower:
            return 'call'
        elif 'put' in text_lower:
            return 'put'
        elif 'straddle' in text_lower or 'strangle' in text_lower:
            return 'straddle'
        elif 'earnings' in text_lower:
            return 'earnings'
        elif 'volatility' in text_lower:
            return 'volatility'
        elif 'theta' in text_lower:
            return 'theta'
        elif 'yolo' in text_lower or 'all in' in text_lower:
            return 'yolo'
        else:
            return 'general'
    
    def _analyze_reddit_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for Reddit content"""
        positive_words = ['bullish', 'long', 'buy', 'profit', 'gain', 'up', 'rise', 'strong', 'moon', 'rocket']
        negative_words = ['bearish', 'short', 'sell', 'loss', 'down', 'fall', 'weak', 'crash', 'dump', 'bear']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _is_trading_related(self, post) -> bool:
        """Check if a Reddit post is trading-related"""
        text = (post.title + ' ' + post.selftext).lower()
        
        trading_keywords = [
            'strategy', 'trade', 'option', 'call', 'put', 'spread', 'iron condor',
            'butterfly', 'straddle', 'strangle', 'covered call', 'cash secured put',
            'wheel', 'theta', 'gamma', 'delta', 'vega', 'profit', 'loss', 'stop',
            'earnings', 'volatility', 'dividend', 'stock', 'equity', 'portfolio'
        ]
        
        return any(keyword in text for keyword in trading_keywords)

class DiscordAPI:
    """Discord API integration for trading channels"""
    
    def __init__(self, config: APISource):
        self.config = config
        self.cache = TTLCache(maxsize=200, ttl=900)  # 15 minute cache
        
    def initialize_client(self):
        """Initialize Discord bot client"""
        try:
            import discord
            from discord.ext import commands
            
            if not self.config.api_key:
                logger.warning("Discord bot token not configured")
                return None
                
            intents = discord.Intents.default()
            intents.message_content = True
            
            bot = commands.Bot(command_prefix='!', intents=intents)
            return bot
        except ImportError:
            logger.error("discord.py not installed - pip install discord.py")
            return None
        except Exception as e:
            logger.error(f"Error initializing Discord client: {e}")
            return None
    
    async def get_trading_messages(self, channel_ids: List[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trading-related messages from Discord channels"""
        bot = self.initialize_client()
        if not bot:
            return []
        
        messages = []
        
        try:
            @bot.event
            async def on_ready():
                logger.info(f"Discord bot logged in as {bot.user}")
                
                channel_ids = channel_ids or []  # Add specific channel IDs here
                
                for channel_id in channel_ids:
                    channel = bot.get_channel(channel_id)
                    if channel:
                        async for message in channel.history(limit=limit):
                            if self._is_trading_related(message.content):
                                message_data = {
                                    'id': message.id,
                                    'content': message.content,
                                    'author': str(message.author),
                                    'channel': channel.name,
                                    'created_at': message.created_at.isoformat(),
                                    'reactions': len(message.reactions),
                                    'source': 'discord'
                                }
                                messages.append(message_data)
            
            # This would need to be run in an async context
            # await bot.start(self.config.api_key)
            
        except Exception as e:
            logger.error(f"Error fetching Discord messages: {e}")
        
        return messages
    
    def _is_trading_related(self, content: str) -> bool:
        """Check if a Discord message is trading-related"""
        text = content.lower()
        
        trading_keywords = [
            'strategy', 'trade', 'option', 'call', 'put', 'spread', 'iron condor',
            'butterfly', 'straddle', 'strangle', 'covered call', 'cash secured put',
            'wheel', 'theta', 'gamma', 'delta', 'vega', 'profit', 'loss', 'stop'
        ]
        
        return any(keyword in text for keyword in trading_keywords)

class NewsAPI:
    """News API integration for financial news and market insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('news_api_key')
        self.base_url = 'https://newsapi.org/v2'
        self.cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour cache
        
    def get_financial_news(self, symbols: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get financial news articles with trading analysis"""
        if not self.api_key:
            logger.warning("News API key not configured")
            return []
        
        symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
        articles = []
        
        try:
            for symbol in symbols:
                # Check cache first
                cache_key = f"news_{symbol}_{limit}"
                if cache_key in self.cache:
                    articles.extend(self.cache[cache_key])
                    continue
                
                # Fetch news for symbol
                url = f"{self.base_url}/everything"
                params = {
                    'q': f"{symbol} OR {symbol} stock OR {symbol} options",
                    'domains': 'bloomberg.com,reuters.com,marketwatch.com,cnbc.com,ft.com,wsj.com',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(limit, 100),
                    'apiKey': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('status') == 'ok':
                    symbol_articles = []
                    for article in data.get('articles', []):
                        if self._is_trading_relevant(article):
                            article_data = {
                                'id': article.get('url', '').split('/')[-1],
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'content': article.get('content', ''),
                                'url': article.get('url', ''),
                                'published_at': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', ''),
                                'symbol': symbol,
                                'source': 'news_api',
                                'sentiment': self._analyze_news_sentiment(article.get('title', '') + ' ' + article.get('description', '')),
                                'strategy_type': self._classify_news_strategy(article.get('title', '') + ' ' + article.get('description', '')),
                                'name': f"News Analysis: {article.get('title', '')[:50]}...",
                                'details': f"{article.get('title', '')}\n\n{article.get('description', '')}"
                            }
                            symbol_articles.append(article_data)
                    
                    # Cache results
                    self.cache[cache_key] = symbol_articles
                    articles.extend(symbol_articles)
                    
        except Exception as e:
            logger.error(f"Error fetching financial news: {e}")
            
        logger.info(f"Retrieved {len(articles)} financial news articles")
        return articles
    
    def _is_trading_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if news article is trading-relevant"""
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        trading_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance',
            'options', 'volatility', 'trading', 'market', 'stock',
            'bullish', 'bearish', 'upgrade', 'downgrade', 'target',
            'analyst', 'forecast', 'outlook', 'dividend', 'split'
        ]
        
        return any(keyword in text for keyword in trading_keywords)
    
    def _analyze_news_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for news content"""
        positive_words = ['bullish', 'upgrade', 'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'rise']
        negative_words = ['bearish', 'downgrade', 'miss', 'weak', 'decline', 'loss', 'fall', 'drop', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_news_strategy(self, text: str) -> str:
        """Classify news-based strategy type"""
        text_lower = text.lower()
        
        if 'earnings' in text_lower:
            return 'earnings'
        elif 'volatility' in text_lower:
            return 'volatility'
        elif 'options' in text_lower:
            return 'options'
        elif 'dividend' in text_lower:
            return 'dividend'
        elif 'merger' in text_lower or 'acquisition' in text_lower:
            return 'merger'
        else:
            return 'general'

class AlphaVantageAPI:
    """Alpha Vantage API integration for market data and sentiment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('alpha_vantage_key')
        self.base_url = 'https://www.alphavantage.co/query'
        self.cache = TTLCache(maxsize=100, ttl=1800)  # 30 minute cache
        
    def get_market_sentiment(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Get market sentiment and technical data for symbols"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT']
        market_data = []
        
        try:
            for symbol in symbols:
                # Check cache first
                cache_key = f"av_{symbol}"
                if cache_key in self.cache:
                    market_data.append(self.cache[cache_key])
                    continue
                
                # Get technical indicators
                technical_data = self._get_technical_indicators(symbol)
                
                # Get news sentiment
                news_sentiment = self._get_news_sentiment(symbol)
                
                # Combine data
                symbol_data = {
                    'symbol': symbol,
                    'source': 'alpha_vantage',
                    'technical_indicators': technical_data,
                    'news_sentiment': news_sentiment,
                    'overall_sentiment': self._calculate_overall_sentiment(technical_data, news_sentiment),
                    'name': f"Market Analysis: {symbol}",
                    'details': f"Technical and sentiment analysis for {symbol}"
                }
                
                # Cache results
                self.cache[cache_key] = symbol_data
                market_data.append(symbol_data)
                
                # Rate limiting
                time.sleep(12)  # 5 calls per minute limit
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            
        logger.info(f"Retrieved market data for {len(market_data)} symbols")
        return market_data
    
    def _get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators for a symbol"""
        try:
            # RSI
            rsi_params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': 14,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            rsi_response = requests.get(self.base_url, params=rsi_params, timeout=10)
            rsi_data = rsi_response.json()
            
            # MACD
            macd_params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': 'daily',
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            macd_response = requests.get(self.base_url, params=macd_params, timeout=10)
            macd_data = macd_response.json()
            
            # Bollinger Bands
            bb_params = {
                'function': 'BBANDS',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': 20,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            bb_response = requests.get(self.base_url, params=bb_params, timeout=10)
            bb_data = bb_response.json()
            
            return {
                'rsi': self._extract_latest_value(rsi_data, 'RSI'),
                'macd': self._extract_latest_value(macd_data, 'MACD'),
                'bollinger_bands': self._extract_latest_value(bb_data, 'Real Upper Band'),
                'volatility': self._calculate_volatility(bb_data)
            }
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            return {}
    
    def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for a symbol"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                sentiments = []
                for article in data['feed'][:10]:  # Last 10 articles
                    if 'ticker_sentiment' in article:
                        for ticker_sentiment in article['ticker_sentiment']:
                            if ticker_sentiment['ticker'] == symbol:
                                sentiments.append(float(ticker_sentiment['ticker_sentiment_score']))
                
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    return {
                        'average_sentiment': avg_sentiment,
                        'sentiment_count': len(sentiments),
                        'sentiment_classification': self._classify_sentiment_score(avg_sentiment)
                    }
            
            return {'average_sentiment': 0.0, 'sentiment_count': 0, 'sentiment_classification': 'neutral'}
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return {'average_sentiment': 0.0, 'sentiment_count': 0, 'sentiment_classification': 'neutral'}
    
    def _extract_latest_value(self, data: Dict[str, Any], key: str) -> float:
        """Extract the latest value from Alpha Vantage response"""
        try:
            if 'Technical Analysis' in data:
                technical_data = data['Technical Analysis']
                latest_date = max(technical_data.keys())
                return float(technical_data[latest_date][key])
            return 0.0
        except:
            return 0.0
    
    def _calculate_volatility(self, bb_data: Dict[str, Any]) -> float:
        """Calculate volatility from Bollinger Bands"""
        try:
            if 'Technical Analysis' in bb_data:
                technical_data = bb_data['Technical Analysis']
                latest_date = max(technical_data.keys())
                upper = float(technical_data[latest_date]['Real Upper Band'])
                lower = float(technical_data[latest_date]['Real Lower Band'])
                middle = float(technical_data[latest_date]['Real Middle Band'])
                return (upper - lower) / middle
            return 0.0
        except:
            return 0.0
    
    def _classify_sentiment_score(self, score: float) -> str:
        """Classify sentiment score into category"""
        if score > 0.35:
            return 'very_bullish'
        elif score > 0.15:
            return 'bullish'
        elif score > -0.15:
            return 'neutral'
        elif score > -0.35:
            return 'bearish'
        else:
            return 'very_bearish'
    
    def _calculate_overall_sentiment(self, technical_data: Dict[str, Any], news_sentiment: Dict[str, Any]) -> str:
        """Calculate overall sentiment from technical and news data"""
        technical_score = 0
        news_score = news_sentiment.get('average_sentiment', 0)
        
        # Technical analysis scoring
        rsi = technical_data.get('rsi', 50)
        if rsi > 70:
            technical_score -= 0.5  # Overbought
        elif rsi < 30:
            technical_score += 0.5  # Oversold
        
        volatility = technical_data.get('volatility', 0)
        if volatility > 0.1:
            technical_score += 0.2  # High volatility can be bullish for options
        
        # Combine scores
        overall_score = (technical_score + news_score) / 2
        
        if overall_score > 0.2:
            return 'bullish'
        elif overall_score < -0.2:
            return 'bearish'
        else:
            return 'neutral'

class MarketauxAPI:
    """Marketaux API integration for financial news and market sentiment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('marketaux_api_key')
        self.base_url = 'https://api.marketaux.com/v1'
        self.cache = TTLCache(maxsize=200, ttl=1800)  # 30 minute cache
        
    def get_financial_news(self, symbols: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get financial news articles with trading analysis"""
        if not self.api_key:
            logger.warning("Marketaux API key not configured")
            return []
        
        symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
        articles = []
        
        try:
            for symbol in symbols:
                # Check cache first
                cache_key = f"marketaux_{symbol}_{limit}"
                if cache_key in self.cache:
                    articles.extend(self.cache[cache_key])
                    continue
                
                # Fetch news for symbol
                url = f"{self.base_url}/news/all"
                params = {
                    'symbols': symbol,
                    'language': 'en',
                    'limit': min(limit, 100),
                    'api_token': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('data'):  # Marketaux returns data directly, not status
                    symbol_articles = []
                    for article in data.get('data', []):
                        if self._is_trading_relevant(article):
                            article_data = {
                                'id': article.get('uuid', ''),
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'content': article.get('content', ''),
                                'url': article.get('url', ''),
                                'published_at': article.get('published_at', ''),
                                'source': article.get('source', ''),
                                'symbol': symbol,
                                'source': 'marketaux',
                                'sentiment': self._analyze_news_sentiment(article.get('title', '') + ' ' + article.get('description', '')),
                                'strategy_type': self._classify_news_strategy(article.get('title', '') + ' ' + article.get('description', '')),
                                'name': f"Marketaux News: {article.get('title', '')[:50]}...",
                                'details': f"{article.get('title', '')}\n\n{article.get('description', '')}"
                            }
                            symbol_articles.append(article_data)
                    
                    # Cache results
                    self.cache[cache_key] = symbol_articles
                    articles.extend(symbol_articles)
                    
        except Exception as e:
            logger.error(f"Error fetching Marketaux news: {e}")
            
        logger.info(f"Retrieved {len(articles)} financial news articles from Marketaux")
        return articles
    
    def _is_trading_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if news article is trading-relevant"""
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        trading_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance',
            'options', 'volatility', 'trading', 'market', 'stock',
            'bullish', 'bearish', 'upgrade', 'downgrade', 'target',
            'analyst', 'forecast', 'outlook', 'dividend', 'split',
            'merger', 'acquisition', 'ipo', 'bankruptcy', 'rally'
        ]
        
        return any(keyword in text for keyword in trading_keywords)
    
    def _analyze_news_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for news content"""
        positive_words = ['bullish', 'upgrade', 'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'rise', 'rally']
        negative_words = ['bearish', 'downgrade', 'miss', 'weak', 'decline', 'loss', 'fall', 'drop', 'crash', 'bankruptcy']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_news_strategy(self, text: str) -> str:
        """Classify news-based strategy type"""
        text_lower = text.lower()
        
        if 'earnings' in text_lower:
            return 'earnings'
        elif 'volatility' in text_lower:
            return 'volatility'
        elif 'options' in text_lower:
            return 'options'
        elif 'dividend' in text_lower:
            return 'dividend'
        elif 'merger' in text_lower or 'acquisition' in text_lower:
            return 'merger'
        elif 'ipo' in text_lower:
            return 'ipo'
        else:
            return 'general'

class TiingoAPI:
    """Tiingo API integration for market data, news, and fundamentals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('tiingo_api_key')
        self.base_url = 'https://api.tiingo.com'
        self.cache = TTLCache(maxsize=200, ttl=1800)  # 30 minute cache
        
    def get_market_data(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Get market data and analysis for symbols"""
        if not self.api_key:
            logger.warning("Tiingo API key not configured")
            return []
        
        symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
        strategies = []
        
        try:
            for symbol in symbols:
                # Check cache first
                cache_key = f"tiingo_{symbol}"
                if cache_key in self.cache:
                    strategies.extend(self.cache[cache_key])
                    continue
                
                # Get latest price data
                price_data = self._get_latest_price(symbol)
                if price_data:
                    # Get news for the symbol
                    news_data = self._get_news(symbol)
                    
                    # Create market analysis strategy
                    strategy = {
                        'id': f"tiingo_{symbol}_{int(time.time())}",
                        'name': f"Tiingo Analysis: {symbol}",
                        'details': f"Market analysis for {symbol} based on Tiingo data",
                        'source': 'tiingo',
                        'symbol': symbol,
                        'price_data': price_data,
                        'news_count': len(news_data),
                        'sentiment': self._analyze_tiingo_sentiment(news_data),
                        'strategy_type': self._classify_tiingo_strategy(price_data, news_data),
                        'confidence_score': self._calculate_tiingo_confidence(price_data, news_data),
                        'risk_score': self._calculate_tiingo_risk(price_data),
                        'quality_score': self._calculate_tiingo_quality(price_data, news_data)
                    }
                    
                    strategies.append(strategy)
                    
                    # Cache results
                    self.cache[cache_key] = [strategy]
                    
        except Exception as e:
            logger.error(f"Error fetching Tiingo market data: {e}")
            
        logger.info(f"Retrieved {len(strategies)} market analyses from Tiingo")
        return strategies
    
    def _get_latest_price(self, symbol: str) -> Dict[str, Any]:
        """Get latest price data for a symbol"""
        try:
            url = f"{self.base_url}/tiingo/daily/{symbol}/prices"
            params = {
                'token': self.api_key,
                'startDate': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                'endDate': datetime.now().strftime('%Y-%m-%d'),
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                latest = data[-1]  # Most recent data
                return {
                    'date': latest.get('date'),
                    'open': latest.get('open'),
                    'high': latest.get('high'),
                    'low': latest.get('low'),
                    'close': latest.get('close'),
                    'volume': latest.get('volume'),
                    'adjOpen': latest.get('adjOpen'),
                    'adjHigh': latest.get('adjHigh'),
                    'adjLow': latest.get('adjLow'),
                    'adjClose': latest.get('adjClose'),
                    'adjVolume': latest.get('adjVolume'),
                    'divCash': latest.get('divCash'),
                    'splitFactor': latest.get('splitFactor')
                }
        except Exception as e:
            logger.error(f"Error fetching Tiingo price data for {symbol}: {e}")
        
        return {}
    
    def _get_news(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent news for a symbol"""
        try:
            url = f"{self.base_url}/tiingo/news"
            params = {
                'token': self.api_key,
                'tickers': symbol,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data if isinstance(data, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching Tiingo news for {symbol}: {e}")
        
        return []
    
    def _analyze_tiingo_sentiment(self, news_data: List[Dict[str, Any]]) -> str:
        """Analyze sentiment from Tiingo news data"""
        if not news_data:
            return 'neutral'
        
        positive_words = ['bullish', 'upgrade', 'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'rise', 'rally']
        negative_words = ['bearish', 'downgrade', 'miss', 'weak', 'decline', 'loss', 'fall', 'drop', 'crash', 'bankruptcy']
        
        positive_count = 0
        negative_count = 0
        
        for article in news_data:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            positive_count += sum(1 for word in positive_words if word in text)
            negative_count += sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_tiingo_strategy(self, price_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> str:
        """Classify strategy type based on Tiingo data"""
        if not price_data:
            return 'general'
        
        # Analyze price movement
        close = price_data.get('close', 0)
        open_price = price_data.get('open', 0)
        
        if close > open_price:
            price_trend = 'bullish'
        elif close < open_price:
            price_trend = 'bearish'
        else:
            price_trend = 'neutral'
        
        # Check for earnings or major news
        news_text = ' '.join([article.get('title', '') + ' ' + article.get('description', '') for article in news_data]).lower()
        
        if 'earnings' in news_text:
            return 'earnings'
        elif 'volatility' in news_text:
            return 'volatility'
        elif 'dividend' in news_text:
            return 'dividend'
        elif price_trend == 'bullish':
            return 'momentum'
        elif price_trend == 'bearish':
            return 'contrarian'
        else:
            return 'general'
    
    def _calculate_tiingo_confidence(self, price_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for Tiingo data"""
        score = 50.0  # Base score
        
        # Boost confidence based on data completeness
        if price_data and len(price_data) > 5:
            score += 20.0
        
        if news_data and len(news_data) > 0:
            score += 15.0
        
        # Boost confidence based on recent data
        if price_data.get('date'):
            try:
                data_date = datetime.fromisoformat(price_data['date'].replace('T', ' ').replace('Z', ''))
                days_old = (datetime.now() - data_date).days
                if days_old <= 1:
                    score += 15.0
                elif days_old <= 3:
                    score += 10.0
            except:
                pass
        
        return min(score, 100.0)
    
    def _calculate_tiingo_risk(self, price_data: Dict[str, Any]) -> float:
        """Calculate risk score based on price volatility"""
        if not price_data:
            return 50.0
        
        high = price_data.get('high', 0)
        low = price_data.get('low', 0)
        close = price_data.get('close', 0)
        
        if close == 0:
            return 50.0
        
        # Calculate daily volatility
        daily_range = (high - low) / close
        volatility_score = min(daily_range * 100, 100.0)
        
        return volatility_score
    
    def _calculate_tiingo_quality(self, price_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> float:
        """Calculate quality score for Tiingo data"""
        score = 60.0  # Base score
        
        # Boost quality based on data completeness
        if price_data and len(price_data) > 8:
            score += 20.0
        
        if news_data and len(news_data) > 2:
            score += 20.0
        
        return min(score, 100.0)

class TradingPlatformAPI:
    """Integration with trading platforms for strategy data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
    
    def get_tradingview_ideas(self, symbol: str = "SPY", limit: int = 20) -> List[Dict[str, Any]]:
        """Get TradingView ideas for a symbol"""
        try:
            # TradingView doesn't have a public API, so we'll use web scraping
            # This is a placeholder for the actual implementation
            logger.info(f"TradingView ideas for {symbol} would be fetched here")
            return []
        except Exception as e:
            logger.error(f"Error fetching TradingView ideas: {e}")
            return []
    
    def get_alpha_vantage_news(self, symbol: str = "SPY") -> List[Dict[str, Any]]:
        """Get news from Alpha Vantage API"""
        try:
            from alpha_vantage.news import News
            
            if not self.config.get('alpha_vantage_key'):
                logger.warning("Alpha Vantage API key not configured")
                return []
            
            news = News(key=self.config['alpha_vantage_key'])
            data, _ = news.get_news_sentiment(symbol=symbol)
            
            news_items = []
            for item in data:
                news_items.append({
                    'title': item['title'],
                    'summary': item['summary'],
                    'sentiment': item['overall_sentiment_label'],
                    'sentiment_score': item['overall_sentiment_score'],
                    'url': item['url'],
                    'source': 'alpha_vantage',
                    'published_at': item['time_published']
                })
            
            logger.info(f"Retrieved {len(news_items)} news items from Alpha Vantage")
            return news_items
            
        except ImportError:
            logger.error("alpha-vantage not installed - pip install alpha-vantage")
            return []
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []

class APIManager:
    """Manages all API integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.apis = {}
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize all configured APIs"""
        # Twitter API
        if self.config.get('twitter'):
            twitter_config = APISource(
                name='twitter',
                api_key=self.config['twitter'].get('bearer_token'),
                api_secret=self.config['twitter'].get('api_secret')
            )
            self.apis['twitter'] = TwitterAPI(twitter_config)
        
        # Reddit API
        if self.config.get('reddit'):
            reddit_config = APISource(
                name='reddit',
                api_key=self.config['reddit'].get('client_id'),
                api_secret=self.config['reddit'].get('client_secret')
            )
            self.apis['reddit'] = RedditAPI(reddit_config)
        
        # Discord API
        if self.config.get('discord'):
            discord_config = APISource(
                name='discord',
                api_key=self.config['discord'].get('bot_token')
            )
            self.apis['discord'] = DiscordAPI(discord_config)
        
        # News API
        if self.config.get('news_api'):
            self.apis['news'] = NewsAPI(self.config)
        
        # Marketaux API
        if self.config.get('marketaux'):
            # Map marketaux.api_key to marketaux_api_key for the API
            marketaux_config = self.config.copy()
            if 'marketaux' in marketaux_config and 'api_key' in marketaux_config['marketaux']:
                marketaux_config['marketaux_api_key'] = marketaux_config['marketaux']['api_key']
            self.apis['marketaux'] = MarketauxAPI(marketaux_config)
        
        # Tiingo API
        if self.config.get('tiingo'):
            # Map tiingo.api_key to tiingo_api_key for the API
            tiingo_config = self.config.copy()
            if 'tiingo' in tiingo_config and 'api_key' in tiingo_config['tiingo']:
                tiingo_config['tiingo_api_key'] = tiingo_config['tiingo']['api_key']
            self.apis['tiingo'] = TiingoAPI(tiingo_config)
        
        # Alpha Vantage API
        if self.config.get('alpha_vantage'):
            # Map alpha_vantage.api_key to alpha_vantage_key for the API
            av_config = self.config.copy()
            if 'alpha_vantage' in av_config and 'api_key' in av_config['alpha_vantage']:
                av_config['alpha_vantage_key'] = av_config['alpha_vantage']['api_key']
            self.apis['alpha_vantage'] = AlphaVantageAPI(av_config)
        
        # Trading Platform APIs
        self.apis['trading_platform'] = TradingPlatformAPI(self.config)
    
    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get strategies from all configured APIs"""
        all_strategies = []
        
        # Twitter strategies
        if 'twitter' in self.apis:
            try:
                twitter_strategies = self.apis['twitter'].get_trading_tweets()
                all_strategies.extend(twitter_strategies)
            except Exception as e:
                logger.error(f"Error fetching Twitter strategies: {e}")
        
        # Reddit strategies
        if 'reddit' in self.apis:
            try:
                reddit_strategies = self.apis['reddit'].get_trading_posts()
                all_strategies.extend(reddit_strategies)
            except Exception as e:
                logger.error(f"Error fetching Reddit strategies: {e}")
        
        # Discord strategies (would need async implementation)
        if 'discord' in self.apis:
            logger.info("Discord strategies would be fetched here (requires async)")
        
        # News strategies
        if 'news' in self.apis:
            try:
                news_strategies = self.apis['news'].get_financial_news()
                all_strategies.extend(news_strategies)
            except Exception as e:
                logger.error(f"Error fetching News strategies: {e}")
        
        # Marketaux strategies
        if 'marketaux' in self.apis:
            try:
                marketaux_strategies = self.apis['marketaux'].get_financial_news()
                all_strategies.extend(marketaux_strategies)
            except Exception as e:
                logger.error(f"Error fetching Marketaux strategies: {e}")
        
        # Tiingo strategies
        if 'tiingo' in self.apis:
            try:
                tiingo_strategies = self.apis['tiingo'].get_market_data()
                all_strategies.extend(tiingo_strategies)
            except Exception as e:
                logger.error(f"Error fetching Tiingo strategies: {e}")
        
        # Alpha Vantage strategies
        if 'alpha_vantage' in self.apis:
            try:
                av_strategies = self.apis['alpha_vantage'].get_market_sentiment()
                all_strategies.extend(av_strategies)
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage strategies: {e}")
        
        # Trading platform strategies
        if 'trading_platform' in self.apis:
            try:
                platform_strategies = self.apis['trading_platform'].get_tradingview_ideas()
                all_strategies.extend(platform_strategies)
            except Exception as e:
                logger.error(f"Error fetching trading platform strategies: {e}")
        
        logger.info(f"Retrieved {len(all_strategies)} total strategies from all APIs")
        return all_strategies
    
    def get_strategies_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get strategies from a specific source"""
        if source not in self.apis:
            logger.warning(f"API source '{source}' not configured")
            return []
        
        api = self.apis[source]
        
        if source == 'twitter':
            return api.get_trading_tweets()
        elif source == 'reddit':
            return api.get_trading_posts()
        elif source == 'discord':
            logger.info("Discord strategies would be fetched here (requires async)")
            return []
        elif source == 'trading_platform':
            return api.get_tradingview_ideas()
        
        return []

def main():
    """Test the API integrations"""
    print("Testing API Integrations...")
    
    # Example configuration
    config = {
        'twitter': {
            'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', ''),
            'api_secret': os.getenv('TWITTER_API_SECRET', '')
        },
        'reddit': {
            'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET', '')
        },
        'discord': {
            'bot_token': os.getenv('DISCORD_BOT_TOKEN', '')
        },
        'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', '')
    }
    
    api_manager = APIManager(config)
    
    print("\n1. Testing Twitter API...")
    twitter_strategies = api_manager.get_strategies_by_source('twitter')
    print(f"Found {len(twitter_strategies)} Twitter strategies")
    
    print("\n2. Testing Reddit API...")
    reddit_strategies = api_manager.get_strategies_by_source('reddit')
    print(f"Found {len(reddit_strategies)} Reddit strategies")
    
    print("\n3. Testing all APIs...")
    all_strategies = api_manager.get_all_strategies()
    print(f"Found {len(all_strategies)} total strategies from all APIs")
    
    print("\nAPI integration test completed!")

if __name__ == "__main__":
    main()
