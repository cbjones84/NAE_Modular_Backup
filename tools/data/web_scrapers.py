#!/usr/bin/env python3
"""
Real Web Scrapers for Trading Strategy Ingestion
Scrapes real trading forums, websites, and social media for strategy data
"""

import os
import time
import json
import hashlib
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ratelimit import limits, sleep_and_retry
from cachetools import TTLCache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedStrategy:
    """Data structure for scraped trading strategies"""
    title: str
    content: str
    source: str
    url: str
    author: str
    timestamp: str
    upvotes: int = 0
    comments: int = 0
    tags: List[str] = None
    strategy_type: str = "unknown"
    confidence_score: float = 0.0

class TradingForumScraper:
    """Scrapes trading forums for strategy discussions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        
    @sleep_and_retry
    @limits(calls=10, period=60)  # 10 calls per minute
    def scrape_reddit_options(self, subreddit: str = "options") -> List[ScrapedStrategy]:
        """Scrape Reddit r/options for trading strategies"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            strategies = []
            
            for post in data['data']['children']:
                post_data = post['data']
                
                # Filter for strategy-related posts
                if self._is_strategy_post(post_data):
                    strategy = ScrapedStrategy(
                        title=post_data.get('title', ''),
                        content=post_data.get('selftext', ''),
                        source=f"reddit_r_{subreddit}",
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        author=post_data.get('author', 'unknown'),
                        timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                        upvotes=post_data.get('ups', 0),
                        comments=post_data.get('num_comments', 0),
                        tags=self._extract_tags(post_data.get('title', '') + ' ' + post_data.get('selftext', '')),
                        strategy_type=self._classify_strategy(post_data.get('title', '') + ' ' + post_data.get('selftext', '')),
                        confidence_score=self._calculate_confidence(post_data)
                    )
                    strategies.append(strategy)
            
            logger.info(f"Scraped {len(strategies)} strategies from Reddit r/{subreddit}")
            return strategies
            
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            return []
    
    def scrape_tradingview_ideas(self, symbol: str = "SPY") -> List[ScrapedStrategy]:
        """Scrape TradingView ideas for specific symbols"""
        try:
            # Use Selenium for dynamic content
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            url = f"https://www.tradingview.com/ideas/{symbol.lower()}/"
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "tv-card"))
            )
            
            strategies = []
            cards = driver.find_elements(By.CLASS_NAME, "tv-card")
            
            for card in cards[:10]:  # Limit to first 10
                try:
                    title_elem = card.find_element(By.CLASS_NAME, "tv-card__title")
                    title = title_elem.text
                    
                    content_elem = card.find_element(By.CLASS_NAME, "tv-card__description")
                    content = content_elem.text
                    
                    author_elem = card.find_element(By.CLASS_NAME, "tv-card__author")
                    author = author_elem.text
                    
                    link_elem = card.find_element(By.TAG_NAME, "a")
                    url = link_elem.get_attribute("href")
                    
                    strategy = ScrapedStrategy(
                        title=title,
                        content=content,
                        source="tradingview",
                        url=url,
                        author=author,
                        timestamp=datetime.now().isoformat(),
                        tags=self._extract_tags(title + ' ' + content),
                        strategy_type=self._classify_strategy(title + ' ' + content),
                        confidence_score=self._calculate_tradingview_confidence(card)
                    )
                    strategies.append(strategy)
                    
                except Exception as e:
                    logger.warning(f"Error parsing TradingView card: {e}")
                    continue
            
            driver.quit()
            logger.info(f"Scraped {len(strategies)} strategies from TradingView")
            return strategies
            
        except Exception as e:
            logger.error(f"Error scraping TradingView: {e}")
            return []
    
    def scrape_seeking_alpha(self, symbol: str = "SPY") -> List[ScrapedStrategy]:
        """Scrape Seeking Alpha articles for trading strategies"""
        try:
            url = f"https://seekingalpha.com/symbol/{symbol}/analysis"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            strategies = []
            
            articles = soup.find_all('div', class_='sa-article')
            
            for article in articles[:10]:  # Limit to first 10
                try:
                    title_elem = article.find('h3', class_='sa-article-title')
                    title = title_elem.text.strip() if title_elem else ""
                    
                    content_elem = article.find('div', class_='sa-article-summary')
                    content = content_elem.text.strip() if content_elem else ""
                    
                    author_elem = article.find('span', class_='sa-article-author')
                    author = author_elem.text.strip() if author_elem else "unknown"
                    
                    link_elem = article.find('a')
                    url = f"https://seekingalpha.com{link_elem.get('href')}" if link_elem else ""
                    
                    strategy = ScrapedStrategy(
                        title=title,
                        content=content,
                        source="seeking_alpha",
                        url=url,
                        author=author,
                        timestamp=datetime.now().isoformat(),
                        tags=self._extract_tags(title + ' ' + content),
                        strategy_type=self._classify_strategy(title + ' ' + content),
                        confidence_score=self._calculate_seeking_alpha_confidence(article)
                    )
                    strategies.append(strategy)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Seeking Alpha article: {e}")
                    continue
            
            logger.info(f"Scraped {len(strategies)} strategies from Seeking Alpha")
            return strategies
            
        except Exception as e:
            logger.error(f"Error scraping Seeking Alpha: {e}")
            return []
    
    def scrape_discord_trading(self, webhook_url: str = None) -> List[ScrapedStrategy]:
        """Scrape Discord trading channels (requires bot token)"""
        # This would require Discord bot implementation
        # For now, return empty list
        logger.info("Discord scraping not implemented - requires bot token")
        return []
    
    def _is_strategy_post(self, post_data: Dict[str, Any]) -> bool:
        """Determine if a Reddit post contains trading strategy information"""
        title = post_data.get('title', '').lower()
        content = post_data.get('selftext', '').lower()
        
        strategy_keywords = [
            'strategy', 'trade', 'option', 'call', 'put', 'spread', 'iron condor',
            'butterfly', 'straddle', 'strangle', 'covered call', 'cash secured put',
            'wheel', 'theta', 'gamma', 'delta', 'vega', 'profit', 'loss', 'stop'
        ]
        
        text = title + ' ' + content
        return any(keyword in text for keyword in strategy_keywords)
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from strategy text"""
        text_lower = text.lower()
        tags = []
        
        tag_mapping = {
            'call': ['call', 'calls'],
            'put': ['put', 'puts'],
            'spread': ['spread', 'spreads'],
            'iron_condor': ['iron condor', 'iron condors'],
            'butterfly': ['butterfly', 'butterflies'],
            'straddle': ['straddle', 'straddles'],
            'strangle': ['strangle', 'strangles'],
            'covered_call': ['covered call', 'covered calls'],
            'cash_secured_put': ['cash secured put', 'csp'],
            'wheel': ['wheel', 'wheel strategy'],
            'theta': ['theta', 'theta gang'],
            'earnings': ['earnings', 'earnings play'],
            'volatility': ['volatility', 'vol', 'iv'],
            'dividend': ['dividend', 'dividend capture']
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _classify_strategy(self, text: str) -> str:
        """Classify the type of trading strategy"""
        text_lower = text.lower()
        
        if 'iron condor' in text_lower:
            return 'iron_condor'
        elif 'butterfly' in text_lower:
            return 'butterfly'
        elif 'straddle' in text_lower:
            return 'straddle'
        elif 'strangle' in text_lower:
            return 'strangle'
        elif 'covered call' in text_lower:
            return 'covered_call'
        elif 'cash secured put' in text_lower or 'csp' in text_lower:
            return 'cash_secured_put'
        elif 'wheel' in text_lower:
            return 'wheel'
        elif 'spread' in text_lower:
            return 'spread'
        elif 'call' in text_lower:
            return 'call'
        elif 'put' in text_lower:
            return 'put'
        else:
            return 'unknown'
    
    def _calculate_confidence(self, post_data: Dict[str, Any]) -> float:
        """Calculate confidence score for Reddit posts"""
        score = 0.0
        
        # Upvote ratio
        upvote_ratio = post_data.get('upvote_ratio', 0.5)
        score += upvote_ratio * 30
        
        # Number of comments (engagement)
        comments = post_data.get('num_comments', 0)
        score += min(comments * 2, 20)
        
        # Post length (more detailed = higher confidence)
        content_length = len(post_data.get('selftext', ''))
        score += min(content_length / 100, 20)
        
        # Account age (older accounts = more trustworthy)
        # This would require additional API calls
        
        return min(score, 100.0)
    
    def _calculate_tradingview_confidence(self, card) -> float:
        """Calculate confidence score for TradingView ideas"""
        try:
            score = 50.0  # Base score
            
            # Look for likes/views indicators
            likes_elem = card.find_element(By.CLASS_NAME, "tv-card__likes")
            if likes_elem:
                likes_text = likes_elem.text
                if 'K' in likes_text:
                    likes = float(likes_text.replace('K', '')) * 1000
                else:
                    likes = float(likes_text)
                score += min(likes / 10, 30)
            
            return min(score, 100.0)
        except:
            return 50.0
    
    def _calculate_seeking_alpha_confidence(self, article) -> float:
        """Calculate confidence score for Seeking Alpha articles"""
        try:
            score = 60.0  # Base score for Seeking Alpha
            
            # Look for author reputation indicators
            author_elem = article.find('span', class_='sa-article-author')
            if author_elem:
                # This would require additional logic to check author reputation
                score += 20
            
            return min(score, 100.0)
        except:
            return 60.0

class SocialMediaScraper:
    """Scrapes social media platforms for trading insights"""
    
    def __init__(self, twitter_config: Dict[str, str] = None):
        self.twitter_config = twitter_config or {}
        self.cache = TTLCache(maxsize=500, ttl=1800)  # 30 minute cache
    
    def scrape_twitter_trading(self, hashtags: List[str] = None) -> List[ScrapedStrategy]:
        """Scrape Twitter for trading-related tweets"""
        if not self.twitter_config.get('bearer_token'):
            logger.warning("Twitter API not configured - skipping Twitter scraping")
            return []
        
        try:
            import tweepy
            
            client = tweepy.Client(bearer_token=self.twitter_config['bearer_token'])
            
            hashtags = hashtags or ['#OptionsTrading', '#TradingStrategy', '#OptionsStrategy']
            strategies = []
            
            for hashtag in hashtags:
                tweets = client.search_recent_tweets(
                    query=hashtag,
                    max_results=20,
                    tweet_fields=['created_at', 'author_id', 'public_metrics']
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        strategy = ScrapedStrategy(
                            title=f"Tweet: {tweet.text[:100]}...",
                            content=tweet.text,
                            source="twitter",
                            url=f"https://twitter.com/i/web/status/{tweet.id}",
                            author=str(tweet.author_id),
                            timestamp=tweet.created_at.isoformat(),
                            upvotes=tweet.public_metrics['like_count'],
                            comments=tweet.public_metrics['reply_count'],
                            tags=self._extract_twitter_tags(tweet.text),
                            strategy_type=self._classify_strategy(tweet.text),
                            confidence_score=self._calculate_twitter_confidence(tweet)
                        )
                        strategies.append(strategy)
            
            logger.info(f"Scraped {len(strategies)} strategies from Twitter")
            return strategies
            
        except Exception as e:
            logger.error(f"Error scraping Twitter: {e}")
            return []
    
    def _extract_twitter_tags(self, text: str) -> List[str]:
        """Extract hashtags and relevant tags from Twitter text"""
        import re
        
        hashtags = re.findall(r'#\w+', text)
        tags = [tag.lower().replace('#', '') for tag in hashtags]
        
        # Add strategy-specific tags
        strategy_tags = self._extract_tags(text)
        tags.extend(strategy_tags)
        
        return list(set(tags))
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        text_lower = text.lower()
        tags = []
        
        tag_mapping = {
            'call': ['call', 'calls'],
            'put': ['put', 'puts'],
            'spread': ['spread', 'spreads'],
            'iron_condor': ['iron condor', 'iron condors'],
            'butterfly': ['butterfly', 'butterflies'],
            'straddle': ['straddle', 'straddles'],
            'strangle': ['strangle', 'strangles'],
            'covered_call': ['covered call', 'covered calls'],
            'cash_secured_put': ['cash secured put', 'csp'],
            'wheel': ['wheel', 'wheel strategy'],
            'theta': ['theta', 'theta gang'],
            'earnings': ['earnings', 'earnings play'],
            'volatility': ['volatility', 'vol', 'iv'],
            'dividend': ['dividend', 'dividend capture']
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _classify_strategy(self, text: str) -> str:
        """Classify the type of trading strategy"""
        text_lower = text.lower()
        
        if 'iron condor' in text_lower:
            return 'iron_condor'
        elif 'butterfly' in text_lower:
            return 'butterfly'
        elif 'straddle' in text_lower:
            return 'straddle'
        elif 'strangle' in text_lower:
            return 'strangle'
        elif 'covered call' in text_lower:
            return 'covered_call'
        elif 'cash secured put' in text_lower or 'csp' in text_lower:
            return 'cash_secured_put'
        elif 'wheel' in text_lower:
            return 'wheel'
        elif 'spread' in text_lower:
            return 'spread'
        elif 'call' in text_lower:
            return 'call'
        elif 'put' in text_lower:
            return 'put'
        else:
            return 'unknown'
    
    def _calculate_twitter_confidence(self, tweet) -> float:
        """Calculate confidence score for Twitter tweets"""
        score = 30.0  # Base score for Twitter
        
        # Engagement metrics
        metrics = tweet.public_metrics
        score += min(metrics['like_count'] * 0.5, 30)
        score += min(metrics['retweet_count'] * 1.0, 20)
        score += min(metrics['reply_count'] * 0.3, 10)
        
        # Tweet length
        score += min(len(tweet.text) / 20, 10)
        
        return min(score, 100.0)

class NewsScraper:
    """Scrapes financial news for market insights"""
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key
        self.cache = TTLCache(maxsize=200, ttl=1800)  # 30 minute cache
    
    def scrape_financial_news(self, symbols: List[str] = None) -> List[ScrapedStrategy]:
        """Scrape financial news for trading insights"""
        if not self.news_api_key:
            logger.warning("News API key not configured - skipping news scraping")
            return []
        
        try:
            from newsapi import NewsApiClient
            
            newsapi = NewsApiClient(api_key=self.news_api_key)
            strategies = []
            
            symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
            
            for symbol in symbols:
                articles = newsapi.get_everything(
                    q=symbol,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )
                
                for article in articles['articles']:
                    strategy = ScrapedStrategy(
                        title=article['title'],
                        content=article['description'] or '',
                        source="financial_news",
                        url=article['url'],
                        author=article['author'] or 'unknown',
                        timestamp=article['publishedAt'],
                        tags=self._extract_news_tags(article['title'] + ' ' + (article['description'] or '')),
                        strategy_type='news_analysis',
                        confidence_score=self._calculate_news_confidence(article)
                    )
                    strategies.append(strategy)
            
            logger.info(f"Scraped {len(strategies)} strategies from financial news")
            return strategies
            
        except Exception as e:
            logger.error(f"Error scraping financial news: {e}")
            return []

def main():
    """Test the web scrapers"""
    print("Testing Real Web Scrapers...")
    
    # Test forum scraper
    forum_scraper = TradingForumScraper()
    
    print("\n1. Testing Reddit scraping...")
    reddit_strategies = forum_scraper.scrape_reddit_options()
    print(f"Found {len(reddit_strategies)} Reddit strategies")
    for strategy in reddit_strategies[:3]:
        print(f"  - {strategy.title[:50]}... (Score: {strategy.confidence_score:.1f})")
    
    print("\n2. Testing TradingView scraping...")
    tv_strategies = forum_scraper.scrape_tradingview_ideas()
    print(f"Found {len(tv_strategies)} TradingView strategies")
    for strategy in tv_strategies[:3]:
        print(f"  - {strategy.title[:50]}... (Score: {strategy.confidence_score:.1f})")
    
    print("\n3. Testing Seeking Alpha scraping...")
    sa_strategies = forum_scraper.scrape_seeking_alpha()
    print(f"Found {len(sa_strategies)} Seeking Alpha strategies")
    for strategy in sa_strategies[:3]:
        print(f"  - {strategy.title[:50]}... (Score: {strategy.confidence_score:.1f})")
    
    print("\nWeb scraping test completed!")

if __name__ == "__main__":
    main()
