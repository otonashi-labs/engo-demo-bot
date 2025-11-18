"""
LLM Analyzer Module - Handles OpenRouter API calls for brew analysis

This module sends brew data to LLM via OpenRouter API and gets analysis results.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Strip whitespace from API key to avoid authentication issues
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Default model - using a capable model for analysis
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Available system prompts
SYSTEM_PROMPTS = {
    "bot": "api/prompts/system_bot.md",     # Main bot prompt - conversational, comprehensive
    "telegram": "api/prompts/system_0.md",  # Brief, Telegram-optimized (legacy)
    "detailed": "api/prompts/system_1.md",  # Detailed fund manager analysis
    "trading": "api/prompts/system_3.md",   # Trading decision (BUY/SELL)
}

# Default prompt for general use
DEFAULT_PROMPT = "bot"


def load_system_prompt(prompt_type: str = DEFAULT_PROMPT) -> str:
    """
    Load the system prompt from file.
    
    Args:
        prompt_type: Type of prompt to load ("telegram", "detailed", "trading")
        
    Returns:
        System prompt text
    """
    if prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(SYSTEM_PROMPTS.keys())}")
    
    prompt_path = SYSTEM_PROMPTS[prompt_type]
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt not found at {prompt_path}")


def estimate_token_count(text: str) -> int:
    """
    More accurate token count estimation.
    Approximation: ~3.5 characters per token for English text + technical content.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Use 3.5 chars per token for better accuracy with technical content
    return int(len(text) / 3.5)


def build_messages(
    brew_text: str,
    user_query: str,
    conversation_history: List[Dict[str, str]],
    analysis_objective: Optional[str] = None,
    prompt_type: str = DEFAULT_PROMPT
) -> List[Dict[str, str]]:
    """
    Build the messages array for the API call.
    
    Args:
        brew_text: The processed text brew data
        user_query: User's query
        conversation_history: Previous conversation turns
        analysis_objective: Optional analysis objective
        prompt_type: Type of system prompt to use
        
    Returns:
        List of message dictionaries
    """
    system_prompt = load_system_prompt(prompt_type)
    
    # Insert brew text into system prompt
    # The system_3.md has <processed_text_brew> tags at the end
    system_prompt = system_prompt.replace(
        "<processed_text_brew>\n\n</processed_text_brew>",
        f"<processed_text_brew>\n\n{brew_text}\n\n</processed_text_brew>"
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn['user']})
        messages.append({"role": "assistant", "content": turn['assistant']})
    
    # Add current user query
    if analysis_objective and analysis_objective != "general analytics, succinct objective":
        current_message = f"User Query: {user_query}\n\nAnalysis Objective: {analysis_objective}"
    else:
        current_message = user_query
    
    messages.append({"role": "user", "content": current_message})
    
    return messages


async def analyze_brew_with_llm(
    brew_text: str,
    user_query: str,
    conversation_history: List[Dict[str, str]],
    analysis_objective: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt_type: str = DEFAULT_PROMPT
) -> Dict[str, Any]:
    """
    Analyze brew data using LLM via OpenRouter API.
    
    Args:
        brew_text: The processed text brew data from ENGO API
        user_query: User's query
        conversation_history: Previous conversation turns
        analysis_objective: Optional specific analysis objective
        model: Model to use (default: Claude 3.5 Sonnet)
        prompt_type: Type of system prompt to use ("telegram", "detailed", "trading")
        
    Returns:
        Dictionary with 'response' and metadata
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Ensure API key is properly formatted (strip any whitespace)
    api_key = OPENROUTER_API_KEY.strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is empty after stripping whitespace")
    
    # Build messages
    messages = build_messages(brew_text, user_query, conversation_history, analysis_objective, prompt_type)
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8000,  # Increased for longer responses
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/engo-project",
        "X-Title": "ENGO Telegram Bot"
    }
    
    # Make API call
    async with aiohttp.ClientSession() as session:
        async with session.post(
            OPENROUTER_API_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
            
            result = await response.json()
    
    # Extract response
    try:
        assistant_message = result['choices'][0]['message']['content']
        
        # Extract content from <processed_text_brew> tags if present
        # (system_0.md uses these tags)
        import re
        brew_tag_match = re.search(r'<processed_text_brew>(.*?)</processed_text_brew>', 
                                    assistant_message, re.DOTALL)
        if brew_tag_match:
            assistant_message = brew_tag_match.group(1).strip()
        
        # Try to parse JSON if the response is meant to be JSON
        # (system_3.md asks for JSON output for trading decisions)
        try:
            # Remove markdown code blocks if present
            json_text = assistant_message
            if '```json' in json_text:
                json_text = re.sub(r'```json\s*', '', json_text)
                json_text = re.sub(r'```\s*$', '', json_text)
            elif '```' in json_text:
                json_text = re.sub(r'```\s*', '', json_text)
            
            parsed_json = json.loads(json_text)
            if 'action' in parsed_json and 'confidence' in parsed_json:
                # Format trading decision nicely
                confidence_value = parsed_json['confidence']
                if isinstance(confidence_value, str):
                    # Convert string percentage to float
                    confidence_value = float(confidence_value.strip('%')) / 100
                
                formatted_response = f"""
🎯 **Trading Decision**

**Action:** {parsed_json['action']} 
**Confidence:** {confidence_value:.1%}

📝 **Analysis:**
{parsed_json.get('explanation', 'No explanation provided.')}
"""
                assistant_message = formatted_response
        except (json.JSONDecodeError, ValueError, TypeError):
            # Not JSON or invalid format, use as-is
            pass
        
        # Extract usage information if available
        usage = result.get('usage', {})
        
        return {
            'response': assistant_message,
            'model': result.get('model', model),
            'usage': usage,  # Contains prompt_tokens, completion_tokens, total_tokens
            'prompt_type': prompt_type
        }
        
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {e}\nResponse: {result}")


async def test_openrouter_connection() -> bool:
    """
    Test OpenRouter API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found")
        return False
    
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 10
        }
        
        api_key = OPENROUTER_API_KEY.strip() if OPENROUTER_API_KEY else ""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    print("✅ OpenRouter API connection successful")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ OpenRouter API error: {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    asyncio.run(test_openrouter_connection())

