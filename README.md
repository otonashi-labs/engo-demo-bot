# ENGO Telegram Bot

A Telegram bot that provides real-time Ethereum blockchain analytics powered by ENGO infrastructure and AI analysis.

## Features

- 📊 Real-time blockchain data analysis
- 🤖 AI-powered insights using OpenRouter
- 💬 Conversational interface with context tracking
- 📈 Multiple brew granularities (1min to 60min)
- 🎯 Multiple baseline comparisons (30d, hour cadence, weekday cadence)
- 🔄 Follow-up questions on the same brew data
- ⚡ Automatic context management (50k token limit)

## Setup

### 1. Prerequisites

- Python 3.8+
- Telegram account
- OpenRouter API key
- Access to ENGO API

### 2. Create Telegram Bot

1. Talk to [@BotFather](https://t.me/botfather) on Telegram
2. Use `/newbot` command
3. Follow the prompts to create your bot
4. Save the bot token you receive

### 3. Get OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and get your API key
3. Add some credits to your account

### 4. Install Dependencies

```bash
cd demo_tg_bot
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the `demo_tg_bot` directory:

```env
# Telegram Bot Token from BotFather
TGBOT_KEY=your_telegram_bot_token_here

# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 6. Test the Setup

Test OpenRouter connection:
```bash
python llm_analyzer.py
```

You should see: `✅ OpenRouter API connection successful`

### 7. Run the Bot

```bash
python bot.py
```

You should see: `🚀 ENGO Bot starting...`

## Usage

### Starting the Bot

1. Find your bot on Telegram (search for the username you created)
2. Send `/start` to begin
3. Ask for analysis!

### Example Queries

**Simple queries:**
- "Analyze current market conditions"
- "What's happening right now?"
- "Give me a market overview"

**Specify brew granularity:**
- "Analyze with 30_min data"
- "Show me 60_min brew analysis"
- "Look at 5_min activity"

**Specify analysis objective:**
- "Analyze for trading signals"
- "Focus on stablecoin flows"
- "Look at MEV activity"
- "Analyze for whale movements"

**Combined:**
- "Give me 30_min brew analysis for trading signals"
- "Analyze 15_min data focusing on USDC flows"

**Follow-up questions:**
- "What about ETH transfers?"
- "Tell me more about the gas prices"
- "Why is that significant?"
- "Can you explain the MEV activity?"

### Commands

- `/start` - Show welcome message
- `/help` - Show detailed help
- `/new_brew` - Clear conversation and start fresh analysis

### Brew Types

| Type | Blocks | Best For |
|------|--------|----------|
| 1_min | 5 | Very short-term signals, high noise |
| 3_min | 15 | Short-term activity |
| 5_min | 25 | Quick market checks |
| **15_min** | 75 | **General analysis (default)** |
| 30_min | 150 | Stable patterns, trading signals |
| 60_min | 300 | Longer trends, lower noise |

### Baseline Types

The bot automatically uses three baseline types for comparison:

1. **30d** - 30-day historical average
   - Shows long-term deviation from normal

2. **hour_cadence_90d** - Hour-of-day patterns over 90 days
   - Identifies unusual activity for the specific hour
   - Example: Is 2 PM activity normal for 2 PM, or unusual?

3. **week_day_cadence_90d** - Day-of-week patterns over 90 days
   - Identifies unusual activity for the specific weekday
   - Example: Is Monday activity typical for Mondays?

**Why multiple baselines?**
They help distinguish truly unusual activity from normal daily/weekly patterns!

## Architecture

```
demo_tg_bot/
├── bot.py                 # Main Telegram bot logic
├── llm_analyzer.py        # OpenRouter API integration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md             # This file
└── api/
    ├── api_connector.py      # ENGO API connector
    ├── hybrid_api_connector.py
    ├── secure_api_connector.py
    └── prompts/
        ├── system_0.md
        ├── system_1.md
        ├── system_2.md
        └── system_3.md       # Main analysis prompt
```

## Key Features Explained

### Context Tracking

- The bot maintains conversation history up to 50k tokens
- You can ask follow-up questions about the same brew
- Use `/new_brew` to analyze a different time period
- Bot warns when approaching context limit

### Smart Query Parsing

The bot automatically detects:
- Brew granularity from your message (1_min, 3_min, etc.)
- Analysis objectives from phrases like "focus on...", "analyze for..."
- Whether you're asking a follow-up or requesting new data

### Formatted Reports

Reports include:
- 📊 Header with brew type, timestamp, and baselines
- 🔍 Detailed analysis from AI
- 💡 Highlighted Ethereum addresses with Etherscan links
- 💬 Token usage and context tracking

## Troubleshooting

### Bot not responding

1. Check if bot is running: `python bot.py`
2. Verify `.env` file exists with correct keys
3. Check bot token is valid in BotFather

### OpenRouter errors

1. Test connection: `python llm_analyzer.py`
2. Verify API key in `.env`
3. Check OpenRouter account has credits

### Import errors

1. Install dependencies: `pip install -r requirements.txt`
2. Make sure you're in the `demo_tg_bot` directory

### ENGO API errors

1. Check network connectivity
2. Verify API endpoint is accessible
3. Check if API key is valid (hardcoded in bot.py)

## Advanced Configuration

### Change LLM Model

Edit `llm_analyzer.py`:

```python
# Default model
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Other options:
# "anthropic/claude-3-opus"
# "openai/gpt-4-turbo"
# "google/gemini-pro-1.5"
```

### Adjust Context Limit

Edit `bot.py`:

```python
MAX_CONTEXT_TOKENS = 50000  # Adjust as needed
```

### Change System Prompt

The bot uses `api/prompts/system_bot.md` by default. You can:
- Edit this file to customize the bot's behavior and analysis style
- Or switch to a different prompt by changing `DEFAULT_PROMPT` in `llm_analyzer.py`
- Available prompts: `system_bot.md`, `system_0.md`, `system_1.md`, `system_3.md`

## Security Notes

- ⚠️ Keep your `.env` file secure and never commit it to git
- ⚠️ The ENGO API key is currently hardcoded - consider moving to `.env` for production
- ⚠️ Bot runs in polling mode - for production, consider webhook mode

## Support

For issues or questions:
1. Check the logs when running `python bot.py`
2. Test individual components (llm_analyzer.py, API connector)
3. Review system prompt at `api/prompts/system_3.md`

## License

Part of the ENGO project.

