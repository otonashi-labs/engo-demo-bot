"""
ENGO API Connector - Now with Hybrid Access Support

This connector automatically detects and supports both access patterns:

1. DIRECT ACCESS (Default): 
   - Uses http://localhost:8020 or tailscale IP
   - No API key required
   - Full performance, no restrictions
   - For local development and internal network usage

2. EXTERNAL ACCESS:
   - Uses Traefik gateway with tailscale funnel
   - API key authentication required
   - Rate limiting and security features  
   - For public/external access

The connector automatically detects the appropriate mode based on the URL.
"""

from .hybrid_api_connector import (
    HybridBrewsConnector,
    HybridBrewsConnector as BrewsReaderConnector,
    create_direct_connector,
    create_external_connector, 
    quick_get_brew,
    quick_get_recent_brews,
    quick_get_swaps,
    quick_get_swaps_stats,
    quick_get_swaps_as_dataframe,
    quick_get_most_traded_count_tokens,
    quick_get_most_traded_value_tokens,
    quick_get_token_stats,
    quick_get_trader_stats,
    quick_get_trading_stats,
    AccessMode
)

# Also import secure connector for external use cases
from .secure_api_connector import (
    SecureBrewsConnector,
    get_available_api_keys,
    setup_api_key_env
)

# For backward compatibility
__all__ = [
    'BrewsReaderConnector',
    'HybridBrewsConnector', 
    'SecureBrewsConnector',
    'create_direct_connector',
    'create_external_connector',
    'quick_get_brew',
    'quick_get_recent_brews',
    'quick_get_swaps',
    'quick_get_swaps_stats',
    'quick_get_swaps_as_dataframe',
    'quick_get_most_traded_count_tokens',
    'quick_get_most_traded_value_tokens',
    'quick_get_token_stats',
    'quick_get_trader_stats',
    'quick_get_trading_stats',
    'get_available_api_keys',
    'setup_api_key_env',
    'AccessMode'
]