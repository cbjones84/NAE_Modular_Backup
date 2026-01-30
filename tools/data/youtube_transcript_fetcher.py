#!/usr/bin/env python3
"""
YouTube Transcript Fetcher for NAE
Fetches transcripts from YouTube videos using yt-dlp (legal for transcripts)
Processes transcripts to extract trading knowledge
"""

import os
import json
import re
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class YouTubeVideoInfo:
    """Information about a YouTube video"""
    video_id: str
    url: str
    title: str
    channel: str
    duration: int  # in seconds
    transcript: str
    transcript_language: str
    upload_date: str
    view_count: int = 0
    like_count: int = 0

@dataclass
class ExtractedKnowledge:
    """Extracted knowledge from a video transcript"""
    video_id: str
    video_title: str
    strategies: List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]
    rules: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    risk_warnings: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]  # Video chunks for processing
    extraction_timestamp: str
    quality_score: float = 0.0

class YouTubeTranscriptFetcher:
    """Fetches and processes YouTube transcripts"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.temp_dir = tempfile.mkdtemp(prefix="nae_youtube_")
        
    def fetch_transcript(self, video_url: str) -> Optional[YouTubeVideoInfo]:
        """
        Fetch transcript from a YouTube video using yt-dlp
        
        Args:
            video_url: YouTube video URL or video ID
            
        Returns:
            YouTubeVideoInfo object with transcript, or None if failed
        """
        try:
            # Extract video ID if URL provided
            video_id = self._extract_video_id(video_url)
            if not video_id:
                logger.error(f"Could not extract video ID from URL: {video_url}")
                return None
            
            # Use yt-dlp to get video info and transcript
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--convert-subs", "srt",
                "--output", os.path.join(self.temp_dir, "%(id)s.%(ext)s"),
                "--print", "json",
                video_url
            ]
            
            logger.info(f"Fetching transcript for video: {video_id}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"yt-dlp failed: {result.stderr}")
                return None
            
            # Parse JSON output
            try:
                video_info_json = json.loads(result.stdout.split('\n')[0])
            except json.JSONDecodeError:
                logger.error("Failed to parse yt-dlp JSON output")
                return None
            
            # Read transcript file
            transcript_file = os.path.join(self.temp_dir, f"{video_id}.en.srt")
            if not os.path.exists(transcript_file):
                # Try auto-generated subtitle
                transcript_file = os.path.join(self.temp_dir, f"{video_id}.en.vtt")
                if not os.path.exists(transcript_file):
                    logger.warning(f"No transcript found for video {video_id}")
                    return None
            
            transcript_text = self._read_transcript_file(transcript_file)
            
            if not transcript_text:
                logger.warning(f"Empty transcript for video {video_id}")
                return None
            
            # Create YouTubeVideoInfo object
            video_info = YouTubeVideoInfo(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=video_info_json.get("title", "Unknown"),
                channel=video_info_json.get("channel", "Unknown"),
                duration=video_info_json.get("duration", 0),
                transcript=transcript_text,
                transcript_language="en",
                upload_date=video_info_json.get("upload_date", ""),
                view_count=video_info_json.get("view_count", 0),
                like_count=video_info_json.get("like_count", 0)
            )
            
            logger.info(f"Successfully fetched transcript for: {video_info.title}")
            return video_info
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout fetching transcript for {video_url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None
    
    def fetch_multiple_transcripts(self, video_urls: List[str]) -> List[YouTubeVideoInfo]:
        """Fetch transcripts from multiple YouTube videos"""
        results = []
        for url in video_urls:
            video_info = self.fetch_transcript(url)
            if video_info:
                results.append(video_info)
            # Small delay to avoid rate limiting
            import time
            time.sleep(1)
        return results
    
    def _extract_video_id(self, url_or_id: str) -> Optional[str]:
        """Extract video ID from YouTube URL or return if already an ID"""
        # If it's already a video ID (11 characters, alphanumeric)
        if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url_or_id):
            return url_or_id
        
        # Try to extract from various YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def _read_transcript_file(self, file_path: str) -> str:
        """Read and clean transcript from SRT or VTT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove SRT/VTT formatting and timestamps
            # SRT format: number, timestamp, text
            # VTT format: timestamp, text
            lines = content.split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines, timestamps, and sequence numbers
                if not line:
                    continue
                if re.match(r'^\d+$', line):  # Sequence number
                    continue
                if re.match(r'^\d{2}:\d{2}:\d{2}', line):  # Timestamp
                    continue
                if line.startswith('WEBVTT') or line.startswith('Kind:'):
                    continue
                if '-->' in line:  # Timestamp line
                    continue
                # Remove HTML tags if present
                line = re.sub(r'<[^>]+>', '', line)
                if line:
                    text_lines.append(line)
            
            return ' '.join(text_lines)
            
        except Exception as e:
            logger.error(f"Error reading transcript file: {e}")
            return ""
    
    def chunk_transcript(self, transcript: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Break transcript into chunks for processing
        
        Args:
            transcript: Full transcript text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0
        
        for i, word in enumerate(words):
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "start_word": i - len(current_chunk),
                    "end_word": i - 1,
                    "char_count": len(chunk_text)
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start_word": len(words) - len(current_chunk),
                "end_word": len(words) - 1,
                "char_count": len(chunk_text)
            })
        
        return chunks
    
    def extract_knowledge(self, video_info: YouTubeVideoInfo) -> ExtractedKnowledge:
        """
        Extract trading knowledge from video transcript
        
        Extracts:
        - Strategies
        - Definitions
        - Rules
        - Examples
        - Risk warnings
        """
        transcript = video_info.transcript.lower()
        chunks = self.chunk_transcript(video_info.transcript)
        
        # Extract different types of knowledge
        strategies = self._extract_strategies(transcript, chunks, video_info.video_id)
        definitions = self._extract_definitions(transcript, chunks, video_info.video_id)
        rules = self._extract_rules(transcript, chunks, video_info.video_id)
        examples = self._extract_examples(transcript, chunks, video_info.video_id)
        risk_warnings = self._extract_risk_warnings(transcript, chunks, video_info.video_id)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            strategies, definitions, rules, examples, risk_warnings
        )
        
        return ExtractedKnowledge(
            video_id=video_info.video_id,
            video_title=video_info.title,
            strategies=strategies,
            definitions=definitions,
            rules=rules,
            examples=examples,
            risk_warnings=risk_warnings,
            chunks=chunks,
            extraction_timestamp=datetime.now().isoformat(),
            quality_score=quality_score
        )
    
    def _extract_strategies(self, transcript: str, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Extract trading strategies from transcript"""
        strategies = []
        
        # Strategy keywords and patterns
        strategy_patterns = [
            r'(?:strategy|approach|method|technique|system|setup|play)\s+(?:is|for|to|called|known as|involves)\s+([^\.]+)',
            r'(?:i|we|you|traders?)\s+(?:use|employ|implement|apply|execute)\s+([^\.]+?)(?:strategy|approach|method)',
            r'(?:the|a|an)\s+([^\.]+?)\s+(?:strategy|approach|method|technique)',
        ]
        
        strategy_keywords = [
            'covered call', 'cash secured put', 'csp', 'wheel', 'iron condor',
            'butterfly', 'straddle', 'strangle', 'credit spread', 'debit spread',
            'calendar spread', 'diagonal spread', 'protective put', 'collar',
            'covered strangle', 'ratio spread', 'backspread'
        ]
        
        # Find strategy mentions
        found_strategies = set()
        for pattern in strategy_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                strategy_text = match.group(1).strip()
                if len(strategy_text) > 10 and len(strategy_text) < 200:
                    found_strategies.add(strategy_text)
        
        # Also look for keyword-based strategies
        for keyword in strategy_keywords:
            if keyword in transcript:
                # Find context around keyword
                pattern = rf'.{{0,100}}{re.escape(keyword)}.{{0,100}}'
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    context = match.group(0)
                    found_strategies.add(context)
        
        # Convert to structured format
        for i, strategy_text in enumerate(found_strategies):
            strategies.append({
                "strategy_id": f"{video_id}_strategy_{i}",
                "text": strategy_text,
                "type": self._classify_strategy_type(strategy_text),
                "confidence": 0.7  # Default confidence
            })
        
        return strategies
    
    def _extract_definitions(self, transcript: str, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Extract definitions from transcript"""
        definitions = []
        
        # Definition patterns
        definition_patterns = [
            r'(?:is|means|refers to|defined as|called)\s+([^\.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|means|refers to|defined as)\s+([^\.]+)',
        ]
        
        definition_keywords = ['delta', 'gamma', 'theta', 'vega', 'rho', 'iv', 'iv crush',
                              'time decay', 'intrinsic value', 'extrinsic value', 'itm', 'otm', 'atm']
        
        found_definitions = set()
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 1:
                    term = match.group(1).strip()
                    definition = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                else:
                    definition = match.group(1).strip()
                    term = None
                
                if len(definition) > 10 and len(definition) < 300:
                    if term:
                        found_definitions.add(f"{term}: {definition}")
                    else:
                        found_definitions.add(definition)
        
        # Also extract definitions for known terms
        for keyword in definition_keywords:
            if keyword in transcript:
                pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,100}}'
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    context = match.group(0)
                    found_definitions.add(context)
        
        for i, definition_text in enumerate(found_definitions):
            definitions.append({
                "definition_id": f"{video_id}_def_{i}",
                "text": definition_text,
                "confidence": 0.6
            })
        
        return definitions
    
    def _extract_rules(self, transcript: str, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Extract trading rules from transcript"""
        rules = []
        
        # Rule patterns
        rule_patterns = [
            r'(?:always|never|should|must|rule|principle|guideline)\s+([^\.]+)',
            r'(?:if|when)\s+([^\.]+?),\s+(?:then|you|do|execute)\s+([^\.]+)',
            r'(?:don\'t|do not|avoid|prevent)\s+([^\.]+)',
        ]
        
        found_rules = set()
        
        for pattern in rule_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 1:
                    rule_text = f"{match.group(1)} -> {match.group(2)}"
                else:
                    rule_text = match.group(1).strip()
                
                if len(rule_text) > 10 and len(rule_text) < 200:
                    found_rules.add(rule_text)
        
        for i, rule_text in enumerate(found_rules):
            rules.append({
                "rule_id": f"{video_id}_rule_{i}",
                "text": rule_text,
                "confidence": 0.65
            })
        
        return rules
    
    def _extract_examples(self, transcript: str, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Extract examples from transcript"""
        examples = []
        
        # Example patterns
        example_patterns = [
            r'(?:example|instance|case|scenario|situation)\s+([^\.]+)',
            r'(?:for example|for instance|say|imagine|suppose)\s+([^\.]+)',
            r'(?:let\'s say|let\'s say|if you|suppose you)\s+([^\.]+)',
        ]
        
        found_examples = set()
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                example_text = match.group(1).strip()
                if len(example_text) > 20 and len(example_text) < 300:
                    found_examples.add(example_text)
        
        for i, example_text in enumerate(found_examples):
            examples.append({
                "example_id": f"{video_id}_example_{i}",
                "text": example_text,
                "confidence": 0.6
            })
        
        return examples
    
    def _extract_risk_warnings(self, transcript: str, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Extract risk warnings from transcript"""
        risk_warnings = []
        
        # Risk warning patterns
        risk_patterns = [
            r'(?:risk|warning|caution|danger|loss|downside|drawback|problem|issue)\s+([^\.]+)',
            r'(?:be careful|watch out|beware|avoid|prevent)\s+([^\.]+)',
            r'(?:can lose|may lose|could lose|risk of|potential loss)\s+([^\.]+)',
        ]
        
        risk_keywords = ['risk', 'loss', 'danger', 'warning', 'caution', 'drawdown',
                        'margin call', 'assignment', 'early assignment', 'pin risk']
        
        found_warnings = set()
        
        for pattern in risk_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                warning_text = match.group(1).strip()
                if len(warning_text) > 10 and len(warning_text) < 200:
                    found_warnings.add(warning_text)
        
        # Also look for keyword-based warnings
        for keyword in risk_keywords:
            if keyword in transcript:
                pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,100}}'
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    context = match.group(0)
                    found_warnings.add(context)
        
        for i, warning_text in enumerate(found_warnings):
            risk_warnings.append({
                "warning_id": f"{video_id}_warning_{i}",
                "text": warning_text,
                "severity": self._assess_warning_severity(warning_text),
                "confidence": 0.7
            })
        
        return risk_warnings
    
    def _classify_strategy_type(self, text: str) -> str:
        """Classify the type of trading strategy"""
        text_lower = text.lower()
        
        strategy_types = {
            'iron_condor': ['iron condor'],
            'butterfly': ['butterfly'],
            'straddle': ['straddle'],
            'strangle': ['strangle'],
            'covered_call': ['covered call'],
            'cash_secured_put': ['cash secured put', 'csp'],
            'wheel': ['wheel'],
            'spread': ['spread'],
            'call': ['call'],
            'put': ['put']
        }
        
        for strategy_type, keywords in strategy_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return strategy_type
        
        return 'unknown'
    
    def _assess_warning_severity(self, text: str) -> str:
        """Assess the severity of a risk warning"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['extreme', 'severe', 'catastrophic', 'total loss']):
            return 'high'
        elif any(word in text_lower for word in ['significant', 'major', 'substantial', 'large']):
            return 'medium'
        else:
            return 'low'
    
    def _calculate_quality_score(self, strategies: List, definitions: List, 
                                 rules: List, examples: List, risk_warnings: List) -> float:
        """Calculate quality score for extracted knowledge"""
        score = 0.0
        
        # More strategies = higher score
        score += min(len(strategies) * 10, 30)
        
        # More definitions = higher score
        score += min(len(definitions) * 5, 20)
        
        # More rules = higher score
        score += min(len(rules) * 5, 20)
        
        # More examples = higher score
        score += min(len(examples) * 5, 15)
        
        # Risk warnings are important for quality
        score += min(len(risk_warnings) * 5, 15)
        
        return min(score, 100.0)
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")

# Fix: video_info is not defined in extract methods - need to pass it
# Let me fix this:
