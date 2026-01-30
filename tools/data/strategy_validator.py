#!/usr/bin/env python3
"""
Data Validation for Trading Strategies
Validates and scores trading strategies from various sources
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of strategy validation"""
    is_valid: bool
    confidence_score: float
    risk_score: float
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    strategy_type: str
    complexity_score: float

class StrategyValidator:
    """Validates trading strategies for quality and risk"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Strategy patterns for validation
        self.strategy_patterns = {
            'iron_condor': [
                r'iron\s+condor', r'sell\s+call\s+spread.*sell\s+put\s+spread',
                r'long\s+put.*short\s+put.*short\s+call.*long\s+call'
            ],
            'butterfly': [
                r'butterfly', r'long\s+call.*short\s+call.*long\s+call',
                r'long\s+put.*short\s+put.*long\s+put'
            ],
            'straddle': [
                r'straddle', r'buy\s+call.*buy\s+put.*same\s+strike',
                r'long\s+call.*long\s+put'
            ],
            'strangle': [
                r'strangle', r'buy\s+call.*buy\s+put.*different\s+strikes',
                r'out\s+of\s+the\s+money.*call.*put'
            ],
            'covered_call': [
                r'covered\s+call', r'own\s+stock.*sell\s+call',
                r'buy\s+and\s+hold.*sell\s+call'
            ],
            'cash_secured_put': [
                r'cash\s+secured\s+put', r'csp', r'sell\s+put.*cash\s+backing',
                r'put\s+seller.*cash\s+reserve'
            ],
            'wheel': [
                r'wheel\s+strategy', r'csp.*covered\s+call.*cycle',
                r'put\s+seller.*call\s+seller'
            ]
        }
        
        # Risk indicators
        self.risk_indicators = [
            'unlimited risk', 'naked', 'margin', 'leverage', 'all in',
            'yolo', 'gambling', 'lottery', 'pump', 'dump', 'moon',
            'diamond hands', 'paper hands', 'hodl'
        ]
        
        # Quality indicators
        self.quality_indicators = [
            'risk management', 'stop loss', 'position sizing', 'diversification',
            'backtest', 'historical', 'probability', 'expected value',
            'greeks', 'theta', 'delta', 'gamma', 'vega', 'rho'
        ]
    
    def validate_strategy(self, strategy_data: Dict[str, Any]) -> ValidationResult:
        """Validate a trading strategy"""
        try:
            # Extract text content
            text_content = self._extract_text_content(strategy_data)
            
            # Basic validation checks
            is_valid, basic_issues = self._basic_validation(strategy_data, text_content)
            
            # Calculate scores
            confidence_score = self._calculate_confidence_score(strategy_data, text_content)
            risk_score = self._calculate_risk_score(text_content)
            quality_score = self._calculate_quality_score(strategy_data, text_content)
            complexity_score = self._calculate_complexity_score(text_content)
            
            # Identify strategy type
            strategy_type = self._identify_strategy_type(text_content)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                strategy_data, text_content, risk_score, quality_score
            )
            
            # Combine all issues
            all_issues = basic_issues + self._identify_issues(strategy_data, text_content)
            
            # Determine overall validity
            overall_valid = (
                is_valid and 
                confidence_score >= 30 and 
                risk_score <= 70 and 
                quality_score >= 40
            )
            
            return ValidationResult(
                is_valid=overall_valid,
                confidence_score=confidence_score,
                risk_score=risk_score,
                quality_score=quality_score,
                issues=all_issues,
                recommendations=recommendations,
                strategy_type=strategy_type,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            logger.error(f"Error validating strategy: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                risk_score=100.0,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Fix validation error"],
                strategy_type="unknown",
                complexity_score=0.0
            )
    
    def _extract_text_content(self, strategy_data: Dict[str, Any]) -> str:
        """Extract all text content from strategy data"""
        text_parts = []
        
        # Add title
        if 'title' in strategy_data:
            text_parts.append(str(strategy_data['title']))
        
        # Add content
        if 'content' in strategy_data:
            text_parts.append(str(strategy_data['content']))
        
        # Add details
        if 'details' in strategy_data:
            text_parts.append(str(strategy_data['details']))
        
        # Add aggregated details
        if 'aggregated_details' in strategy_data:
            text_parts.append(str(strategy_data['aggregated_details']))
        
        return ' '.join(text_parts).lower()
    
    def _basic_validation(self, strategy_data: Dict[str, Any], text_content: str) -> Tuple[bool, List[str]]:
        """Perform basic validation checks"""
        issues = []
        
        # Check for minimum content length
        if len(text_content) < 50:
            issues.append("Content too short - insufficient detail")
        
        # Check for required fields
        if not strategy_data.get('title') and not strategy_data.get('content'):
            issues.append("Missing title and content")
        
        # Check for spam indicators
        spam_indicators = ['click here', 'free money', 'guaranteed profit', 'no risk']
        if any(indicator in text_content for indicator in spam_indicators):
            issues.append("Contains spam indicators")
        
        # Check for excessive repetition
        words = text_content.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:
                issues.append("Excessive word repetition")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _calculate_confidence_score(self, strategy_data: Dict[str, Any], text_content: str) -> float:
        """Calculate confidence score for the strategy"""
        score = 0.0
        
        # Source reputation
        source = strategy_data.get('source', 'unknown')
        source_scores = {
            'seeking_alpha': 80,
            'tradingview': 75,
            'reddit': 60,
            'twitter': 50,
            'discord': 45,
            'unknown': 30
        }
        score += source_scores.get(source, 30)
        
        # Engagement metrics
        if 'upvotes' in strategy_data:
            upvotes = strategy_data['upvotes']
            score += min(upvotes * 0.5, 20)
        
        if 'comments' in strategy_data:
            comments = strategy_data['comments']
            score += min(comments * 0.3, 15)
        
        # Content quality indicators
        quality_mentions = sum(1 for indicator in self.quality_indicators if indicator in text_content)
        score += min(quality_mentions * 5, 25)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text_content)
        score += (sentiment['compound'] + 1) * 10  # Convert -1 to 1 range to 0-20
        
        return min(score, 100.0)
    
    def _calculate_risk_score(self, text_content: str) -> float:
        """Calculate risk score for the strategy"""
        risk_score = 0.0
        
        # Count risk indicators
        risk_mentions = sum(1 for indicator in self.risk_indicators if indicator in text_content)
        risk_score += risk_mentions * 15
        
        # Check for high-risk language
        high_risk_patterns = [
            r'unlimited\s+risk', r'naked\s+call', r'naked\s+put',
            r'margin\s+call', r'all\s+in', r'yolo'
        ]
        
        for pattern in high_risk_patterns:
            if re.search(pattern, text_content):
                risk_score += 20
        
        # Check for proper risk management
        risk_management_patterns = [
            r'stop\s+loss', r'position\s+sizing', r'risk\s+management',
            r'max\s+loss', r'risk\s+reward'
        ]
        
        risk_management_mentions = sum(1 for pattern in risk_management_patterns 
                                     if re.search(pattern, text_content))
        risk_score -= risk_management_mentions * 10
        
        return max(0.0, min(risk_score, 100.0))
    
    def _calculate_quality_score(self, strategy_data: Dict[str, Any], text_content: str) -> float:
        """Calculate quality score for the strategy"""
        score = 0.0
        
        # Content length (more detailed = higher quality)
        content_length = len(text_content)
        score += min(content_length / 20, 20)
        
        # Technical terms usage
        technical_terms = [
            'delta', 'gamma', 'theta', 'vega', 'rho', 'greeks',
            'implied volatility', 'historical volatility', 'iv',
            'probability', 'expected value', 'backtest', 'historical'
        ]
        
        technical_mentions = sum(1 for term in technical_terms if term in text_content)
        score += min(technical_mentions * 5, 30)
        
        # Strategy-specific quality indicators
        quality_mentions = sum(1 for indicator in self.quality_indicators if indicator in text_content)
        score += min(quality_mentions * 3, 25)
        
        # Author credibility (if available)
        author = strategy_data.get('author', '')
        if author and author != 'unknown':
            score += 10
        
        # URL presence (indicates external validation)
        if strategy_data.get('url'):
            score += 5
        
        return min(score, 100.0)
    
    def _calculate_complexity_score(self, text_content: str) -> float:
        """Calculate complexity score for the strategy"""
        score = 0.0
        
        # Count strategy patterns (more patterns = more complex)
        pattern_count = 0
        for strategy_type, patterns in self.strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_content):
                    pattern_count += 1
                    break
        
        score += pattern_count * 20
        
        # Count technical terms
        technical_terms = [
            'delta', 'gamma', 'theta', 'vega', 'rho', 'greeks',
            'implied volatility', 'historical volatility', 'iv',
            'probability', 'expected value', 'backtest'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in text_content)
        score += min(technical_count * 5, 30)
        
        # Sentence complexity (average sentence length)
        sentences = re.split(r'[.!?]+', text_content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            score += min(avg_sentence_length * 2, 20)
        
        return min(score, 100.0)
    
    def _identify_strategy_type(self, text_content: str) -> str:
        """Identify the type of trading strategy"""
        for strategy_type, patterns in self.strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_content):
                    return strategy_type
        
        return 'unknown'
    
    def _identify_issues(self, strategy_data: Dict[str, Any], text_content: str) -> List[str]:
        """Identify specific issues with the strategy"""
        issues = []
        
        # Check for missing risk management
        if not any(term in text_content for term in ['stop loss', 'risk management', 'position sizing']):
            issues.append("No risk management mentioned")
        
        # Check for unrealistic promises
        unrealistic_patterns = [
            r'guaranteed\s+profit', r'no\s+risk', r'100%\s+win\s+rate',
            r'guaranteed\s+return', r'risk\s+free'
        ]
        
        for pattern in unrealistic_patterns:
            if re.search(pattern, text_content):
                issues.append("Contains unrealistic promises")
                break
        
        # Check for excessive complexity
        complexity_score = self._calculate_complexity_score(text_content)
        if complexity_score > 80:
            issues.append("Strategy may be too complex for beginners")
        
        # Check for missing technical details
        if not any(term in text_content for term in ['delta', 'theta', 'gamma', 'vega']):
            issues.append("Missing technical analysis details")
        
        return issues
    
    def _generate_recommendations(self, strategy_data: Dict[str, Any], text_content: str, 
                                risk_score: float, quality_score: float) -> List[str]:
        """Generate recommendations for improving the strategy"""
        recommendations = []
        
        # Risk management recommendations
        if risk_score > 60:
            recommendations.append("Consider adding stop-loss orders")
            recommendations.append("Implement position sizing rules")
            recommendations.append("Diversify across multiple strategies")
        
        # Quality improvement recommendations
        if quality_score < 50:
            recommendations.append("Add more technical analysis")
            recommendations.append("Include historical performance data")
            recommendations.append("Provide detailed entry/exit criteria")
        
        # Strategy-specific recommendations
        strategy_type = self._identify_strategy_type(text_content)
        
        if strategy_type == 'iron_condor':
            recommendations.append("Monitor implied volatility levels")
            recommendations.append("Consider early assignment risk")
        
        elif strategy_type == 'covered_call':
            recommendations.append("Monitor dividend dates")
            recommendations.append("Consider tax implications")
        
        elif strategy_type == 'cash_secured_put':
            recommendations.append("Ensure adequate cash reserves")
            recommendations.append("Monitor assignment risk")
        
        return recommendations

class StrategyScorer:
    """Scores and ranks trading strategies"""
    
    def __init__(self):
        self.validator = StrategyValidator()
    
    def score_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank a list of strategies"""
        scored_strategies = []
        
        for strategy in strategies:
            validation_result = self.validator.validate_strategy(strategy)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(validation_result)
            
            scored_strategy = {
                **strategy,
                'validation_result': validation_result,
                'overall_score': overall_score,
                'is_valid': validation_result.is_valid,
                'confidence_score': validation_result.confidence_score,
                'risk_score': validation_result.risk_score,
                'quality_score': validation_result.quality_score,
                'complexity_score': validation_result.complexity_score,
                'strategy_type': validation_result.strategy_type,
                'issues': validation_result.issues,
                'recommendations': validation_result.recommendations
            }
            
            scored_strategies.append(scored_strategy)
        
        # Sort by overall score (descending)
        scored_strategies.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return scored_strategies
    
    def _calculate_overall_score(self, validation_result: ValidationResult) -> float:
        """Calculate overall score from validation result"""
        # Weighted combination of scores
        overall_score = (
            validation_result.confidence_score * 0.3 +
            validation_result.quality_score * 0.3 +
            (100 - validation_result.risk_score) * 0.2 +
            validation_result.complexity_score * 0.1 +
            (100 if validation_result.is_valid else 0) * 0.1
        )
        
        return min(overall_score, 100.0)

def main():
    """Test the data validation system"""
    print("Testing Data Validation System...")
    
    # Test strategy data
    test_strategies = [
        {
            'title': 'Iron Condor Strategy on SPY',
            'content': 'Sell call spread and put spread on SPY. Use 30 DTE options. Set stop loss at 50% of max profit. Monitor delta and theta.',
            'source': 'seeking_alpha',
            'author': 'trading_expert',
            'upvotes': 25,
            'comments': 8
        },
        {
            'title': 'YOLO AAPL Calls',
            'content': 'Buy calls on AAPL. No risk management. All in!',
            'source': 'reddit',
            'author': 'unknown',
            'upvotes': 5,
            'comments': 2
        },
        {
            'title': 'Covered Call Strategy',
            'content': 'Buy 100 shares of stock and sell call options. Use delta 0.3 calls. Set stop loss. Monitor earnings dates.',
            'source': 'tradingview',
            'author': 'options_trader',
            'upvotes': 15,
            'comments': 5
        }
    ]
    
    scorer = StrategyScorer()
    scored_strategies = scorer.score_strategies(test_strategies)
    
    print(f"\nScored {len(scored_strategies)} strategies:")
    for i, strategy in enumerate(scored_strategies, 1):
        print(f"\n{i}. {strategy['title']}")
        print(f"   Overall Score: {strategy['overall_score']:.1f}")
        print(f"   Confidence: {strategy['confidence_score']:.1f}")
        print(f"   Risk Score: {strategy['risk_score']:.1f}")
        print(f"   Quality: {strategy['quality_score']:.1f}")
        print(f"   Strategy Type: {strategy['strategy_type']}")
        print(f"   Valid: {strategy['is_valid']}")
        if strategy['issues']:
            print(f"   Issues: {', '.join(strategy['issues'])}")
        if strategy['recommendations']:
            print(f"   Recommendations: {', '.join(strategy['recommendations'][:2])}")
    
    print("\nData validation test completed!")

if __name__ == "__main__":
    main()
