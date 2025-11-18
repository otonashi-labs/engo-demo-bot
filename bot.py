"""
ENGO Telegram Bot - Blockchain Data Analysis Bot

This bot accepts user queries, fetches brew data from ENGO API,
analyzes it using LLM, and returns comprehensive reports.
"""

import os
import re
import time
import json
import logging
import asyncio
import signal
import sys
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from api.api_connector import create_external_connector
from llm_analyzer import analyze_brew_with_llm, estimate_token_count, OPENROUTER_API_KEY

# Load environment variables
load_dotenv()

# Configure logging with sensitive data filter
class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs (bot tokens, API keys)."""
    def __init__(self):
        super().__init__()
        # Pattern to match bot tokens in URLs: bot<digits>:<token>
        # Telegram bot tokens are typically 35+ characters after the colon
        self.bot_token_pattern = re.compile(r'bot\d+:[a-zA-Z0-9_-]{20,}', re.IGNORECASE)
        # Pattern to match API keys (Bearer tokens)
        self.api_key_pattern = re.compile(r'Bearer\s+[a-zA-Z0-9_-]+', re.IGNORECASE)
        # Pattern to match OpenRouter API keys
        self.openrouter_key_pattern = re.compile(r'sk-or-v1-[a-zA-Z0-9_-]+', re.IGNORECASE)
    
    def redact_sensitive_data(self, text):
        """Redact sensitive data from a text string."""
        if not isinstance(text, str):
            return text
        # Redact bot tokens (in URLs or standalone)
        text = self.bot_token_pattern.sub('[REDACTED_BOT_TOKEN]', text)
        # Redact API keys in Authorization headers
        text = self.api_key_pattern.sub('Bearer [REDACTED_API_KEY]', text)
        # Redact OpenRouter API keys
        text = self.openrouter_key_pattern.sub('[REDACTED_OPENROUTER_KEY]', text)
        return text
    
    def filter(self, record):
        """Redact sensitive data from log messages before they are formatted."""
        # Redact from msg attribute (the main message)
        if hasattr(record, 'msg') and record.msg:
            if isinstance(record.msg, str):
                record.msg = self.redact_sensitive_data(record.msg)
            else:
                # If msg is not a string, convert it first
                record.msg = self.redact_sensitive_data(str(record.msg))
        
        # Redact from args tuple (format arguments)
        if hasattr(record, 'args') and record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self.redact_sensitive_data(arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        
        # Redact from exc_text if present (formatted exception text)
        if hasattr(record, 'exc_text') and record.exc_text:
            record.exc_text = self.redact_sensitive_data(str(record.exc_text))
        
        # Redact from pathname and filename if they contain sensitive data
        if hasattr(record, 'pathname'):
            record.pathname = self.redact_sensitive_data(str(record.pathname))
        if hasattr(record, 'filename'):
            record.filename = self.redact_sensitive_data(str(record.filename))
        
        return True

# Custom formatter that also redacts sensitive data (backup safety measure)
class RedactingFormatter(logging.Formatter):
    """Formatter that redacts sensitive data from log messages."""
    def __init__(self, filter_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_instance = filter_instance
    
    def format(self, record):
        # First apply the filter
        self.filter_instance.filter(record)
        # Then format the message
        formatted = super().format(record)
        # Redact again in case anything slipped through
        return self.filter_instance.redact_sensitive_data(formatted)

# Configure logging
sensitive_filter = SensitiveDataFilter()
formatter = RedactingFormatter(sensitive_filter, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create handler with the redacting formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# Apply filter to root logger (catches all logs)
root_logger.addFilter(sensitive_filter)

# Apply filter to telegram-related loggers (httpx is used by telegram for HTTP requests)
# These loggers output HTTP requests which contain bot tokens in URLs
logging.getLogger('httpx').addFilter(sensitive_filter)
logging.getLogger('telegram').addFilter(sensitive_filter)
logging.getLogger('telegram.ext').addFilter(sensitive_filter)
logging.getLogger('telegram.bot').addFilter(sensitive_filter)
logging.getLogger('telegram.request').addFilter(sensitive_filter)

# Also filter urllib3 and requests loggers if they're used
logging.getLogger('urllib3').addFilter(sensitive_filter)
logging.getLogger('requests').addFilter(sensitive_filter)

logger = logging.getLogger(__name__)

# Constants
# ENGO API Key - can be overridden via ENGO_API_KEY environment variable
DEFAULT_ENGO_API_KEY = "b8c4e2a9d5f7e1a2c8b6d3f4e7a1b2c5d8e6f9a2b5c8d1e4f7a0b3c6d9e2f5a8"
ENGO_API_KEY = os.getenv('ENGO_API_KEY', DEFAULT_ENGO_API_KEY)
ENGO_GATEWAY_URL = "https://otonashi.taile38410.ts.net"
DEFAULT_BREW_TYPE = "15_min"
DEFAULT_ANALYSIS_OBJECTIVE = "general analytics, succinct objective"
MAX_CONTEXT_TOKENS = 50000
VALID_BREW_TYPES = ["1_min", "3_min", "5_min", "15_min", "30_min", "60_min"]

# Data availability safeguard - don't query brews older than this date
EARLIEST_BREW_DATE = datetime(2025, 11, 10, 0, 0, 0)  # November 10, 2025 00:00:00 UTC
EARLIEST_BREW_TIMESTAMP = EARLIEST_BREW_DATE.timestamp()

# Available models for selection
AVAILABLE_MODELS = {
    "sonnet": "anthropic/claude-sonnet-4.5",
    "grok-code": "x-ai/grok-code-fast-1",
    "gemini": "google/gemini-2.5-flash",
    "grok-fast": "x-ai/grok-4-fast",
    "deepseek": "deepseek/deepseek-chat-v3-0324",
    "gpt": "openai/gpt-5.1",
    "grok-4": "x-ai/grok-4",
    "qwen": "qwen/qwen3-235b-a22b-2507",
}
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

# Model display names
MODEL_DISPLAY_NAMES = {
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "x-ai/grok-code-fast-1": "Grok Code Fast",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "x-ai/grok-4-fast": "Grok 4 Fast",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek Chat v3",
    "openai/gpt-5.1": "GPT-5.1",
    "x-ai/grok-4": "Grok 4",
    "qwen/qwen3-235b-a22b-2507": "Qwen3 235B",
}

# Create ENGO API connector
connector = create_external_connector(
    gateway_url=ENGO_GATEWAY_URL,
    api_key=ENGO_API_KEY
)

# Dialogue logging configuration
# Use absolute path relative to script location
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)

# Message files configuration
MESSAGES_DIR = Path(__file__).parent
START_MESSAGE_FILE = MESSAGES_DIR / "start_message.txt"
HELP_MESSAGE_FILE = MESSAGES_DIR / "help_message.txt"


def load_message_file(file_path: Path, **format_kwargs) -> str:
    """
    Load a message from a text file and optionally format it.
    
    Args:
        file_path: Path to the message file
        **format_kwargs: Optional keyword arguments for string formatting
        
    Returns:
        Message content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Format the message if format_kwargs are provided
        if format_kwargs:
            content = content.format(**format_kwargs)
        
        return content
    except FileNotFoundError:
        logger.error(f"Message file not found: {file_path}")
        return "Message file not found. Please contact the administrator."
    except Exception as e:
        logger.error(f"Error loading message file {file_path}: {e}")
        return "Error loading message. Please contact the administrator."


def save_dialogue_log(chat_id: int, user_message: str, assistant_response: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save dialogue turn to JSON file per chat_id.
    
    Args:
        chat_id: Telegram chat ID
        user_message: User's message
        assistant_response: Assistant's response
        metadata: Optional metadata (timestamp, model, tokens, etc.)
    """
    try:
        history_file = HISTORY_DIR / f"chat_{chat_id}.json"
        
        # Load existing history or create new
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                'chat_id': chat_id,
                'created_at': datetime.utcnow().isoformat(),
                'turns': []
            }
        
        # Add new turn
        turn = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user_message,
            'assistant': assistant_response
        }
        
        # Add metadata if provided
        if metadata:
            turn['metadata'] = metadata
        
        history['turns'].append(turn)
        history['updated_at'] = datetime.utcnow().isoformat()
        history['total_turns'] = len(history['turns'])
        
        # Save to file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved dialogue turn for chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Failed to save dialogue log for chat_id {chat_id}: {e}", exc_info=True)


async def parse_user_query_llm(query: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Parse user query using LLM (Gemini 2.5 Flash) to extract metadata.
    
    Args:
        query: User's message
        
    Returns:
        Tuple of (brew_type, analysis_objective, model, timestamp_str)
    """
    import aiohttp
    
    # Build available models list for the prompt
    models_list = "\n".join([f"- {key}: {value}" for key, value in AVAILABLE_MODELS.items()])
    
    parser_prompt = f"""You are a query parser for a blockchain analysis bot. Extract structured information from the user's query.

Available brew types: {', '.join(VALID_BREW_TYPES)}
Default brew type: {DEFAULT_BREW_TYPE}

Available models:
{models_list}
Default model: {DEFAULT_MODEL}

IMPORTANT: For model selection:
- If user mentions "grok" without specifying "grok-fast" or "grok-code", select "x-ai/grok-4" (not "x-ai/grok-4-fast")
- Only select "x-ai/grok-4-fast" if user explicitly mentions "grok-fast", "grok 4 fast", or similar
- Only select "x-ai/grok-code-fast-1" if user explicitly mentions "grok-code" or "grokcode"

Default analysis objective: "{DEFAULT_ANALYSIS_OBJECTIVE}"

Extract the following from the user query:
1. brew_type: One of {VALID_BREW_TYPES} (default: "{DEFAULT_BREW_TYPE}")
2. analysis_objective: What the user wants to analyze (default: "{DEFAULT_ANALYSIS_OBJECTIVE}")
3. model: One of the available model IDs from the list above (default: null/None)
4. timestamp_str: Time specification like "14:30", "2025-11-18 14:30", "2:30pm", etc. (default: null/None)

Return ONLY valid JSON in this exact format:
{{
    "brew_type": "15_min",
    "analysis_objective": "general analytics, succinct objective",
    "model": null,
    "timestamp_str": null
}}

User query: "{query}"
"""
    
    # Use Gemini 2.5 Flash for parsing
    parser_model = "google/gemini-2.5-flash"
    
    if not OPENROUTER_API_KEY:
        # Fallback to defaults if API key not available
        logger.warning("OPENROUTER_API_KEY not available, using defaults for query parsing")
        return DEFAULT_BREW_TYPE, DEFAULT_ANALYSIS_OBJECTIVE, None, None
    
    api_key = OPENROUTER_API_KEY.strip()
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is empty, using defaults for query parsing")
        return DEFAULT_BREW_TYPE, DEFAULT_ANALYSIS_OBJECTIVE, None, None
    
    payload = {
        "model": parser_model,
        "messages": [
            {"role": "user", "content": parser_prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent parsing
        "max_tokens": 200
    }
    
    # Try to add response_format if supported (some models don't support it)
    # We'll parse JSON from the response regardless
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/engo-project",
        "X-Title": "ENGO Telegram Bot Query Parser"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Query parser API error: {response.status} - {error_text}, using defaults")
                    return DEFAULT_BREW_TYPE, DEFAULT_ANALYSIS_OBJECTIVE, None, None
                
                result = await response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response - handle markdown code blocks if present
                content_clean = content.strip()
                if '```json' in content_clean:
                    # Extract JSON from markdown code block
                    json_match = re.search(r'```json\s*(.*?)\s*```', content_clean, re.DOTALL)
                    if json_match:
                        content_clean = json_match.group(1).strip()
                elif '```' in content_clean:
                    # Extract from generic code block
                    json_match = re.search(r'```\s*(.*?)\s*```', content_clean, re.DOTALL)
                    if json_match:
                        content_clean = json_match.group(1).strip()
                
                # Parse JSON response
                parsed = json.loads(content_clean)
                
                brew_type = parsed.get('brew_type', DEFAULT_BREW_TYPE)
                analysis_objective = parsed.get('analysis_objective', DEFAULT_ANALYSIS_OBJECTIVE)
                model = parsed.get('model')  # Can be None
                timestamp_str = parsed.get('timestamp_str')  # Can be None
                
                # Validate brew_type
                if brew_type not in VALID_BREW_TYPES:
                    brew_type = DEFAULT_BREW_TYPE
                
                # Validate model if provided
                if model and model not in AVAILABLE_MODELS.values():
                    # Try to find by key
                    model = AVAILABLE_MODELS.get(model.lower(), None)
                
                # Fix: If user mentions "grok" but not specifically "grok-fast" or "grok-code",
                # select "grok-4" instead of "grok-4-fast"
                query_lower = query.lower()
                grok_fast_model = AVAILABLE_MODELS.get("grok-fast")
                grok_4_model = AVAILABLE_MODELS.get("grok-4")
                
                # Check if model is grok-fast (either by ID or key)
                is_grok_fast = (
                    model == grok_fast_model or 
                    model == "x-ai/grok-4-fast" or 
                    model == "grok-fast"
                )
                
                # Check if user specifically mentioned "grok-fast" variants
                mentioned_grok_fast = any(term in query_lower for term in [
                    "grok-fast", "grok 4 fast", "grok4fast", "grok fast"
                ])
                
                # Check if user mentioned "grok-code"
                mentioned_grok_code = any(term in query_lower for term in [
                    "grok-code", "grokcode", "grok code"
                ])
                
                # If model is grok-fast but user didn't specifically ask for it, use grok-4
                if is_grok_fast and not mentioned_grok_fast:
                    model = grok_4_model
                    logger.info(f"User mentioned 'grok' without 'fast', selecting grok-4 instead of grok-4-fast")
                # If user mentioned "grok" but no model was detected, and they didn't specify fast/code, use grok-4
                elif "grok" in query_lower and model is None and not mentioned_grok_fast and not mentioned_grok_code:
                    model = grok_4_model
                    logger.info(f"User mentioned 'grok' without specification, selecting grok-4")
                
                return brew_type, analysis_objective, model, timestamp_str
                
    except Exception as e:
        logger.warning(f"Error parsing query with LLM: {e}, using defaults")
        return DEFAULT_BREW_TYPE, DEFAULT_ANALYSIS_OBJECTIVE, None, None


def parse_user_query(query: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Legacy regex-based parser (kept for fallback).
    Now replaced by parse_user_query_llm().
    """
    brew_type = DEFAULT_BREW_TYPE
    analysis_objective = DEFAULT_ANALYSIS_OBJECTIVE
    model = None
    timestamp_str = None
    
    # Check for brew granularity in query
    for brew in VALID_BREW_TYPES:
        if brew in query.lower() or brew.replace("_", "-") in query.lower():
            brew_type = brew
            break
    
    # Check for model selection
    query_lower = query.lower()
    for model_key, model_value in AVAILABLE_MODELS.items():
        if model_key in query_lower or model_value.lower() in query_lower:
            model = model_value
            break
    
    # Check for time specifications
    time_patterns = [
        r'at (\d{4}-\d{2}-\d{2} \d{1,2}:\d{2})',
        r'at (\d{1,2}:\d{2})',
        r'^(\d{1,2}:\d{2})\s',
        r'\s(\d{1,2}:\d{2})$',
        r'(\d{1,2}:\d{2}[ap]m)',
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, query_lower)
        if match:
            timestamp_str = match.group(1)
            break
    
    # Extract analysis objective if specified
    objective_patterns = [
        r'analyze for (.+?)(?:\.|$)',
        r'focus on (.+?)(?:\.|$)',
        r'look at (.+?)(?:\.|$)',
        r'objective[:\s]+(.+?)(?:\.|$)',
        r'goal[:\s]+(.+?)(?:\.|$)',
    ]
    
    for pattern in objective_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            analysis_objective = match.group(1).strip()
            break
    
    return brew_type, analysis_objective, model, timestamp_str


def format_report_header(brew_type: str, timestamp: float, baselines: list, model: str = None) -> str:
    """
    Format the report header with brew information.
    
    Args:
        brew_type: Type of brew (e.g., "15_min")
        baselines: List of baseline types used
        model: LLM model used for analysis
        
    Returns:
        Formatted header string
    """
    dt = datetime.fromtimestamp(timestamp)
    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    header = f"""
📊 **ENGO Blockchain Analysis Report**

🕒 **Brew Type:** {brew_type}
📈 **Baselines:** {', '.join(baselines)}
🤖 **Model:** {model if model else 'Claude 3.5 Sonnet'}

---

"""
    return header


def format_addresses_section(text: str) -> str:
    """
    Extract and format any addresses mentioned in the analysis.
    
    Args:
        text: Analysis text
        
    Returns:
        Formatted text with highlighted addresses (monospace only, no links)
    """
    # Find Ethereum addresses (0x followed by 40 hex characters)
    address_pattern = r'(0x[a-fA-F0-9]{40})'
    addresses = re.findall(address_pattern, text)
    
    if addresses:
        # Make addresses monospace (no links - user can copy manually)
        for addr in set(addresses):
            # Only format if not already in backticks
            if f"`{addr}`" not in text:
                text = text.replace(addr, f"`{addr}`")
    
    return text


def parse_timestamp(time_str: str) -> Optional[float]:
    """
    Parse timestamp string to Unix timestamp.
    Assumes GMT/UTC timezone.
    Enforces minimum date of November 10, 2025.
    If parsed time is in the future, uses the most recent past occurrence.
    
    Args:
        time_str: Time string in various formats
        
    Returns:
        Unix timestamp or None if parsing fails or date too old
    """
    from dateutil import parser
    from datetime import timedelta
    
    try:
        # Handle various formats
        if ':' in time_str and '-' not in time_str:
            # Just time like "14:30" - use today's date
            today = datetime.utcnow().strftime('%Y-%m-%d')
            time_str = f"{today} {time_str}"
        
        # Parse and convert to UTC timestamp
        dt = parser.parse(time_str)
        timestamp = dt.timestamp()
        current_timestamp = time.time()
        
        # If timestamp is in the future, go back 24 hours (yesterday at that time)
        if timestamp > current_timestamp:
            logger.info(f"Timestamp {time_str} is in the future, using yesterday at that time")
            dt = dt - timedelta(days=1)
            timestamp = dt.timestamp()
        
        # Check if timestamp is too old
        if timestamp < EARLIEST_BREW_TIMESTAMP:
            logger.warning(f"Timestamp {time_str} is before earliest allowed date (2025-11-10)")
            return None
        
        return timestamp
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{time_str}': {e}")
        return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    welcome_message = load_message_file(START_MESSAGE_FILE)
    await update.message.reply_text(welcome_message)
    
    # Initialize conversation context
    context.user_data['conversation_history'] = []
    context.user_data['total_tokens'] = 0
    context.user_data['current_brew_data'] = None


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    earliest_date_str = EARLIEST_BREW_DATE.strftime('%Y-%m-%d')
    help_message = load_message_file(HELP_MESSAGE_FILE, earliest_date_str=earliest_date_str)
    await update.message.reply_text(help_message)


async def new_brew_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new_brew command."""
    context.user_data['conversation_history'] = []
    context.user_data['total_tokens'] = 0
    context.user_data['current_brew_data'] = None
    
    await update.message.reply_text(
        "🔄 **Context Unplugged**\n\n"
        "Conversation history has been cleared and context reset.\n"
        "Ready for a fresh analysis!\n\n"
        "Send me your query to analyze current blockchain data.",
        parse_mode='Markdown'
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /models command - list all available models."""
    try:
        models_text = "🤖 **Available AI Models**\n\n"
        
        for key, model_id in AVAILABLE_MODELS.items():
            display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id.split('/')[-1])
            is_default = " (default)" if model_id == DEFAULT_MODEL else ""
            models_text += f"**{key}** → {display_name}{is_default}\n"
            models_text += f"   `{model_id}`\n\n"
        
        models_text += f"\n💡 **Usage:**\n"
        models_text += f"Mention model name in your query:\n"
        models_text += f"• \"Analyze market with gemini\"\n"
        models_text += f"• \"30_min brew using grok-fast\"\n"
        models_text += f"• \"Analysis with qwen\"\n\n"
        models_text += f"Default: **{MODEL_DISPLAY_NAMES.get(DEFAULT_MODEL, DEFAULT_MODEL)}**"
        
        # Split if message is too long (Telegram limit is 4096 chars)
        if len(models_text) > 4000:
            # Send in chunks
            chunks = [models_text[i:i+4000] for i in range(0, len(models_text), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode='Markdown')
        else:
            await update.message.reply_text(models_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in models_command: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Error listing models. Please try again later.",
            parse_mode='Markdown'
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_message = update.message.text
    user_id = update.effective_user.id
    
    # Initialize user data if needed
    if 'conversation_history' not in context.user_data:
        context.user_data['conversation_history'] = []
        context.user_data['total_tokens'] = 0
        context.user_data['current_brew_data'] = None
    
    # Check context limit
    if context.user_data['total_tokens'] >= MAX_CONTEXT_TOKENS:
        await update.message.reply_text(
            "⚠️ **Context Limit Reached**\n\n"
            f"The conversation has reached {MAX_CONTEXT_TOKENS:,} tokens. "
            "Please use /new_brew to start a fresh analysis.",
            parse_mode='Markdown'
        )
        return
    
    # Send "thinking" message
    thinking_msg = await update.message.reply_text("🤔 Analyzing... Please wait.")
    
    try:
        # Check if user is asking about existing brew or requesting new one
        is_followup = (
            context.user_data['current_brew_data'] is not None and
            any(keyword in user_message.lower() for keyword in [
                'what about', 'how about', 'tell me more', 'explain', 
                'why', 'what', 'show', 'can you'
            ])
        )
        
        if not is_followup or context.user_data['current_brew_data'] is None:
            # Fetch new brew data - use LLM parser for first message
            brew_type, analysis_objective, selected_model, timestamp_str = await parse_user_query_llm(user_message)
            
            # Use default model if not specified
            if not selected_model:
                selected_model = DEFAULT_MODEL
            
            # Parse timestamp or use current time
            if timestamp_str:
                timestamp = parse_timestamp(timestamp_str)
                if not timestamp:
                    earliest_date_str = EARLIEST_BREW_DATE.strftime('%Y-%m-%d')
                    await thinking_msg.edit_text(
                        f"⚠️ Invalid time '{timestamp_str}' or date before {earliest_date_str}. "
                        f"Using current time instead.\n\n"
                        f"Note: Brew data only available from {earliest_date_str} onwards."
                    )
                    await asyncio.sleep(2)  # Let user read the warning
                    timestamp = time.time()
            else:
                timestamp = time.time()
            
            logger.info(f"User {user_id}: brew_type={brew_type}, model={selected_model}, timestamp={timestamp}, objective={analysis_objective}")
            
            # Fetch brew text from ENGO API
            await thinking_msg.edit_text("🔍 Fetching blockchain data from ENGO API...")
            
            baselines = ["30d", "week_day_cadence_90d", "hour_cadence_90d"]
            brew_text = connector.get_brew_text_at(
                timestamp=f"{timestamp}",
                brew_type=brew_type,
                max_steps_back=3,
                baseline_types=baselines
            )
            
            # Store brew data for follow-up questions
            context.user_data['current_brew_data'] = {
                'brew_text': brew_text,
                'brew_type': brew_type,
                'timestamp': timestamp,
                'baselines': baselines,
                'objective': analysis_objective,
                'model': selected_model
            }
        else:
            # Use existing brew data for follow-up
            brew_data = context.user_data['current_brew_data']
            brew_text = brew_data['brew_text']
            brew_type = brew_data['brew_type']
            timestamp = brew_data['timestamp']
            baselines = brew_data['baselines']
            selected_model = brew_data.get('model', DEFAULT_MODEL)
            analysis_objective = user_message  # Use the follow-up question as objective
        
        # Analyze with LLM
        model_name = MODEL_DISPLAY_NAMES.get(selected_model, selected_model.split('/')[-1])
        await thinking_msg.edit_text(f"🧠 Analyzing with {model_name}...")
        
        analysis_result = await analyze_brew_with_llm(
            brew_text=brew_text,
            user_query=user_message,
            conversation_history=context.user_data['conversation_history'],
            analysis_objective=analysis_objective if not is_followup else None,
            model=selected_model
        )
        
        # Update conversation history
        context.user_data['conversation_history'].append({
            'user': user_message,
            'assistant': analysis_result['response']
        })
        
        # Update token count - use actual API usage if available, otherwise estimate
        usage = analysis_result.get('usage', {})
        if usage and 'total_tokens' in usage:
            # Use actual token count from API response (most accurate)
            # This includes system prompt + brew_text + conversation history + current turn
            tokens_used_this_turn = usage['total_tokens']
            
            # If this is a new brew, reset the counter (new context)
            if not is_followup:
                context.user_data['total_tokens'] = tokens_used_this_turn
            else:
                # For followups, the API returns total tokens including all history
                # So we just update to the latest total
                context.user_data['total_tokens'] = tokens_used_this_turn
        else:
            # Fallback: estimate tokens (less accurate)
            # Count current turn
            tokens_used = estimate_token_count(user_message) + estimate_token_count(analysis_result['response'])
            
            if not is_followup:
                # For new brew: count system prompt + brew_text + current turn
                # Calculate actual system prompt size (without brew_text placeholder)
                from llm_analyzer import load_system_prompt
                system_prompt_base = load_system_prompt()
                system_prompt_tokens = estimate_token_count(system_prompt_base)
                brew_text_tokens = estimate_token_count(brew_text)
                tokens_used = system_prompt_tokens + brew_text_tokens + tokens_used
                context.user_data['total_tokens'] = tokens_used
            else:
                # For followups: only add the new turn (system prompt + brew_text already counted)
                context.user_data['total_tokens'] += tokens_used
        
        # Save dialogue to history file (after token count is updated)
        metadata = {
            'model': selected_model,
            'brew_type': brew_type,
            'timestamp': timestamp,
            'is_followup': is_followup,
            'total_tokens': context.user_data.get('total_tokens', 0),
            'usage': analysis_result.get('usage', {})
        }
        save_dialogue_log(user_id, user_message, analysis_result['response'], metadata)
        
        # Format response
        if not is_followup:
            model_name = MODEL_DISPLAY_NAMES.get(selected_model, selected_model.split('/')[-1])
            header = format_report_header(brew_type, timestamp, baselines, model_name)
            response_text = header + analysis_result['response']
        else:
            response_text = analysis_result['response']
        
        # Highlight any addresses
        response_text = format_addresses_section(response_text)
        
        # Add token usage footer
        footer = f"\n\n---\n💬 Context: {context.user_data['total_tokens']:,} / {MAX_CONTEXT_TOKENS:,} tokens"
        
        if context.user_data['total_tokens'] > MAX_CONTEXT_TOKENS * 0.8:
            footer += "\n⚠️ Approaching context limit. Consider using /new_brew soon."
        
        response_text += footer
        
        # Delete thinking message and send response
        await thinking_msg.delete()
        
        # Helper function to send message with fallback if Markdown fails
        async def send_message_safe(text: str):
            """Send message with Markdown, fall back to plain text if parsing fails."""
            try:
                await update.message.reply_text(text, parse_mode='Markdown', disable_web_page_preview=True)
            except Exception as e:
                if 'parse entities' in str(e).lower() or 'markdown' in str(e).lower():
                    # Markdown parsing failed, try without parse_mode
                    logger.warning(f"Markdown parsing failed, sending as plain text: {e}")
                    try:
                        await update.message.reply_text(text, disable_web_page_preview=True)
                    except Exception as e2:
                        # If still fails, send error message
                        logger.error(f"Failed to send message even without Markdown: {e2}")
                        await update.message.reply_text(
                            "⚠️ Error formatting response. The analysis was generated but couldn't be displayed properly. "
                            "Please try /new_brew to start fresh."
                        )
                else:
                    raise
        
        # Split long messages if needed (Telegram has 4096 char limit)
        if len(response_text) > 4000:
            chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
            for chunk in chunks:
                await send_message_safe(chunk)
        else:
            await send_message_safe(response_text)
        
        logger.info(f"User {user_id}: Successfully processed request. Tokens: {context.user_data['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        
        # Create user-friendly error message based on error type
        error_str = str(e)
        
        if "401" in error_str or "User not found" in error_str or "OpenRouter API error: 401" in error_str:
            error_message = (
                "❌ API Authentication Error\n\n"
                "Your OpenRouter API key appears to be invalid or expired.\n\n"
                "Please check:\n"
                "1. OPENROUTER_API_KEY is set correctly in .env file\n"
                "2. Your API key is valid at https://openrouter.ai/\n"
                "3. Your OpenRouter account has credits\n\n"
                "Contact the bot administrator if this persists."
            )
        elif "Can't parse entities" in error_str or "Markdown" in error_str:
            error_message = (
                "❌ Response Formatting Error\n\n"
                "The analysis was generated but couldn't be displayed properly.\n"
                "This is usually a temporary issue.\n\n"
                "Please try /new_brew to start fresh."
            )
        elif "Failed to get brew text" in error_str:
            error_message = (
                "❌ Data Fetch Error\n\n"
                "Could not fetch blockchain data from ENGO API.\n"
                "The requested time period might not have data available.\n\n"
                "Try:\n"
                "- Using current time (just 'Analyze market')\n"
                "- Different time period\n"
                "- /new_brew to start fresh"
            )
        else:
            # Generic error message (avoid showing raw error which may have special chars)
            error_message = (
                "❌ Error\n\n"
                "An error occurred while processing your request.\n\n"
                "Please try again or use /new_brew to start fresh.\n"
                "If this persists, contact the bot administrator."
            )
        
        try:
            await thinking_msg.edit_text(error_message)
        except Exception as edit_error:
            # If we can't edit the thinking message, send a new one
            logger.error(f"Could not edit thinking message: {edit_error}")
            try:
                await thinking_msg.delete()
                await update.message.reply_text(error_message)
            except:
                pass  # Give up gracefully


def main() -> None:
    """Start the bot with graceful shutdown handling for server deployment."""
    # Get Telegram bot token
    token = os.getenv('TGBOT_KEY')
    if not token:
        logger.error("TGBOT_KEY not found in environment variables")
        raise ValueError("TGBOT_KEY not found in environment variables")
    
    # Set up bot commands menu (appears when user types /)
    async def post_init(app: Application) -> None:
        """Set up bot commands after initialization."""
        commands = [
            BotCommand("start", "Welcome message"),
            BotCommand("help", "Help and usage guide"),
            BotCommand("models", "List available AI models"),
            BotCommand("new_brew", "Reset conversation context"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("✅ Bot commands menu configured")
    
    try:
        # Create application with post_init hook
        application = Application.builder().token(token).post_init(post_init).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("models", models_command))
        application.add_handler(CommandHandler("new_brew", new_brew_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Start bot with error handling
        logger.info("🚀 ENGO Bot starting...")
        logger.info("Press Ctrl+C to stop the bot gracefully")
        
        # Run polling - python-telegram-bot handles SIGINT/SIGTERM gracefully by default
        # This will block until stopped with Ctrl+C or SIGTERM
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Ignore updates received while bot was offline
        )
        
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt (Ctrl+C). Shutting down gracefully...")
        try:
            application.stop()
            application.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("✅ Bot stopped")
    except Exception as e:
        logger.error(f"Fatal error in bot: {e}", exc_info=True)
        try:
            application.stop()
            application.shutdown()
        except:
            pass
        # Re-raise to allow external monitoring/restart mechanisms to detect failure
        raise


if __name__ == '__main__':
    main()

