# YouTube Transcript Integration for NAE

## Overview

Ralph now automatically fetches and processes YouTube video transcripts to learn trading strategies. This allows Ralph to learn from many videos without manual copy/paste.

## Features

- **Automatic Transcript Fetching**: Uses `yt-dlp` (legal for transcripts) to fetch YouTube video transcripts
- **Knowledge Extraction**: Automatically extracts:
  - Trading strategies
  - Definitions (delta, gamma, theta, etc.)
  - Trading rules
  - Examples
  - Risk warnings
- **Cross-Checking**: Strategies are validated against:
  - Existing strategy database
  - Performance data indicators
  - Historical backtest results
- **Quality Filtering**: Automatically filters out:
  - Low-quality strategies
  - Hype/risky strategies
  - Opinion-based content without performance data
- **Integration**: Seamlessly integrates into Ralph's existing pipeline

## How It Works

### 1. YouTube Transcript Fetching

Ralph uses `yt-dlp` to fetch transcripts from YouTube videos:

```python
from agents.ralph import RalphAgent

ralph = RalphAgent()

# Configure YouTube videos to process
ralph.config["youtube_video_urls"] = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2",
    # ... more video URLs
]

# Or use video IDs directly
ralph.config["youtube_video_urls"] = [
    "VIDEO_ID_1",
    "VIDEO_ID_2",
]
```

### 2. Automatic Processing

When Ralph runs his cycle, YouTube videos are automatically processed:

```python
# Ralph automatically processes YouTube videos during run_cycle()
result = ralph.run_cycle()

# YouTube strategies are included in the results
print(f"Processed {result['approved_count']} strategies")
```

### 3. Knowledge Extraction

For each video, Ralph extracts:
- **Strategies**: Trading strategies mentioned in the video
- **Definitions**: Key terms and concepts explained
- **Rules**: Trading rules and principles
- **Examples**: Concrete examples of strategies
- **Risk Warnings**: Risk management information

### 4. Cross-Checking and Validation

Ralph automatically:
- Compares strategies with existing database
- Checks for performance data indicators
- Identifies hype/risky content
- Validates against backtest results

### 5. Quality Filtering

Strategies are filtered based on:
- Cross-check scores (minimum 20)
- Quality scores (minimum 30)
- Presence of risk warnings
- Absence of hype indicators

### 6. Integration with Optimus

Filtered strategies are formatted as "usable strategy blocks" and sent to Optimus through the normal pipeline:

```python
# Get formatted strategy blocks for Optimus
strategy_blocks = ralph.format_strategies_for_optimus()

# Or send directly to Optimus agent
ralph.send_strategies_to_optimus(optimus_agent)
```

## Configuration

### YouTube Video URLs

Add YouTube video URLs to Ralph's config:

```python
ralph.config["youtube_video_urls"] = [
    "https://www.youtube.com/watch?v=VIDEO_ID",
    # ... more URLs
]
```

### Trusted Channels

Configure trusted YouTube channels for higher reputation scores:

```python
ralph.config["youtube_trusted_channels"] = [
    "Tastytrade",
    "Option Alpha",
    "Project Finance",
    # ... more trusted channels
]
```

### Source Reputation

Adjust YouTube source reputation (default: 50):

```python
ralph.config["source_reputations"]["youtube"] = 60  # Higher reputation
```

## Example Usage

```python
from agents.ralph import RalphAgent

# Initialize Ralph
ralph = RalphAgent()

# Configure YouTube videos
ralph.config["youtube_video_urls"] = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example video
]

# Configure trusted channels
ralph.config["youtube_trusted_channels"] = [
    "Tastytrade",
    "Option Alpha",
]

# Run Ralph's cycle (automatically processes YouTube videos)
result = ralph.run_cycle()

# Get top strategies (includes YouTube strategies)
top_strategies = ralph.top_strategies(5)

# Format for Optimus
strategy_blocks = ralph.format_strategies_for_optimus()

print(f"Extracted {len(strategy_blocks)} strategy blocks from YouTube")
for block in strategy_blocks:
    print(f"- {block['name']}: Trust Score {block['trust_score']:.1f}")
    if block.get('metadata', {}).get('youtube_video_title'):
        print(f"  Source: {block['metadata']['youtube_video_title']}")
```

## Important Notes

### YouTube Videos Contain Opinions, Not Truth

Ralph automatically handles this by:
- Cross-checking strategies with performance data
- Comparing with existing strategies
- Filtering out low-probability strategies
- Flagging hype/risky content
- Requiring performance data indicators

### Legal Compliance

- `yt-dlp` is legal for fetching transcripts (not video content)
- Transcripts are used for learning purposes only
- No video content is downloaded or stored

### Automatic Processing

YouTube processing happens automatically as part of Ralph's `run_cycle()`:
1. Fetch transcripts from configured videos
2. Extract knowledge (strategies, definitions, rules, examples, warnings)
3. Cross-check with existing data
4. Filter for quality
5. Integrate into strategy database
6. Format as usable strategy blocks for Optimus

## Troubleshooting

### No Transcripts Found

- Check that videos have captions/subtitles enabled
- Verify video URLs are correct
- Check that `yt-dlp` is installed: `pip install yt-dlp`

### Low Quality Scores

- Videos with more structured content score higher
- Videos with performance data score higher
- Videos with risk warnings score higher
- Videos from trusted channels score higher

### Strategies Filtered Out

- Check cross-check scores (minimum 20 required)
- Check quality scores (minimum 30 required)
- Check for hype flags (hype content is filtered)
- Check for performance data (opinion-only content scores lower)

## Integration with Existing Pipeline

YouTube strategies integrate seamlessly with Ralph's existing pipeline:

1. **Ingestion**: `ingest_from_youtube()` fetches and processes videos
2. **Normalization**: Strategies are normalized and merged with other sources
3. **Backtesting**: Strategies are backtested using existing methods
4. **Scoring**: Trust scores are calculated using existing logic
5. **Filtering**: Strategies are filtered using existing thresholds
6. **Storage**: Strategies are added to `strategy_database`
7. **Distribution**: Strategies are available via `top_strategies()` and sent to Optimus

## Future Enhancements

Potential future improvements:
- Automatic video discovery based on keywords
- Channel subscription management
- Video quality scoring based on engagement metrics
- Multi-language transcript support
- Advanced NLP for better extraction
- Integration with video metadata (views, likes, etc.)

