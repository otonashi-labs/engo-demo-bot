"""
Secure API Connector for ENGO Blockchain Infrastructure

A secure wrapper for accessing brew data through the Traefik API Gateway.
Includes API key authentication, rate limiting handling, and retry logic.
Supports the complete storage hierarchy: Redis → UNFULL → MinIO.
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import requests
from functools import wraps

# Set up logging for Jupyter
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SecureBrewsConnector:
    """
    Secure connector class for accessing brew data through the Traefik API Gateway.
    
    Features:
    - API key authentication for secure access
    - Rate limiting handling with automatic retries  
    - Connection health monitoring and failover
    - Comprehensive error handling and logging
    
    Supports the complete storage hierarchy:
    - Redis: Recently created brews (hot cache)
    - UNFULL: Accumulated brews (temporary MinIO storage)
    - MinIO: Permanent storage (cold storage)
    """
    
    def __init__(self, 
                 gateway_url: str = "https://otonashi.taile38410.ts.net",
                 api_key: str = None,
                 fallback_url: str = "https://otonashi.taile38410.ts.net",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the secure brews connector.
        
        Args:
            gateway_url: Traefik gateway URL (default: "https://otonashi.taile38410.ts.net")
            api_key: API key for authentication. If None, tries environment variable ENGO_API_KEY
            fallback_url: Direct service URL for fallback (default: "https://otonashi.taile38410.ts.net")
            max_retries: Maximum number of retries for rate-limited requests (default: 3)
            retry_delay: Base delay between retries in seconds (default: 1.0)
        """
        self.gateway_url = gateway_url.rstrip('/')
        self.fallback_url = fallback_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_fallback = False
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('ENGO_API_KEY')
        if not self.api_key:
            logger.warning("⚠️ No API key provided. Set ENGO_API_KEY environment variable or pass api_key parameter")
            logger.info("Available API keys in services/traefik/config/api-keys.txt")
        
        # Initialize session with headers
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json',
                'User-Agent': 'ENGO-SecureBrewsConnector/1.0'
            })
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to gateway and fallback URLs."""
        # Try gateway first
        try:
            response = self.session.get(f"{self.gateway_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to Traefik gateway at {self.gateway_url}")
                return
            elif response.status_code == 401:
                logger.error("❌ Authentication failed - invalid API key")
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded on connection test")
            else:
                logger.warning(f"⚠️ Gateway returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to gateway: {e}")
        
        # Try fallback without authentication
        try:
            fallback_session = requests.Session()
            response = fallback_session.get(f"{self.fallback_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"⚠️ Using fallback connection to {self.fallback_url} (no authentication)")
                self.use_fallback = True
            else:
                logger.error(f"❌ Fallback also failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Fallback connection failed: {e}")
    
    @property
    def base_url(self):
        """Get current base URL (gateway or fallback)."""
        return self.fallback_url if self.use_fallback else self.gateway_url
    
    def _make_request_with_retry(self, method: str, url: str, **kwargs):
        """
        Make HTTP request with retry logic for rate limiting and failures.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            url: Request URL
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response object or None if all retries failed
        """
        session = requests.Session() if self.use_fallback else self.session
        
        for attempt in range(self.max_retries + 1):
            try:
                response = session.request(method, url, timeout=30, **kwargs)
                
                # Success
                if response.status_code < 400:
                    return response
                    
                # Rate limited - retry with exponential backoff
                elif response.status_code == 429:
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"⏱️ Rate limited (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("❌ Rate limit exceeded - max retries reached")
                        return response
                
                # Authentication failed - don't retry
                elif response.status_code == 401:
                    logger.error("❌ Authentication failed - check your API key")
                    return response
                    
                # Other client/server errors
                else:
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Request failed with {response.status_code} (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"❌ Request failed with {response.status_code} - max retries reached")
                        return response
                        
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"⏱️ Request timeout (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error("❌ Request timeout - max retries reached")
                    return None
                    
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Request error: {e} (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Request error: {e} - max retries reached")
                    return None
        
        return None
    
    def get_brew(self, brew_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single brew by ID.
        
        Args:
            brew_id: The brew ID to retrieve
            
        Returns:
            Brew data dictionary or None if not found
        """
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew/{brew_id}")
        
        if not response:
            logger.error(f"❌ Failed to get brew {brew_id}: No response received")
            return None
            
        if response.status_code == 200:
            brew_data = response.json()
            logger.info(f"✅ Retrieved brew: {brew_id}")
            return brew_data
        elif response.status_code == 404:
            logger.warning(f"❌ Brew not found: {brew_id}")
            return None
        else:
            logger.error(f"❌ Error retrieving brew {brew_id}: {response.status_code}")
            return None
    
    def get_brew_at_format(self, brew_id: str, format: str) -> Optional[Dict[str, Any]]:
        """
        Get a single brew by ID in specified format.
        
        Args:
            brew_id: The brew ID to retrieve
            format: 'nested' or 'flattened'
            
        Returns:
            Brew data dictionary in requested format or None if not found
        """
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew/{brew_id}/format/{format}")
        
        if not response:
            logger.error(f"❌ Failed to get brew {brew_id} at format {format}: No response received")
            return None
            
        if response.status_code == 200:
            brew_data = response.json()
            logger.info(f"✅ Retrieved brew: {brew_id} (format: {format})")
            return brew_data
        elif response.status_code == 400:
            logger.error(f"❌ Invalid format '{format}': must be 'nested' or 'flattened'")
            return None
        elif response.status_code == 404:
            logger.warning(f"❌ Brew not found: {brew_id}")
            return None
        else:
            logger.error(f"❌ Error retrieving brew {brew_id}: {response.status_code}")
            return None
    
    def get_brew_at(self, timestamp: str, brew_type: str, max_steps_back: int = 0) -> Dict[str, Any]:
        """
        Get a brew by timestamp and type, with optional backward search.
        
        Args:
            timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
            brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
            
        Returns:
            Brew data dictionary
            
        Raises:
            ValueError: Invalid timestamp format or brew_type
            FileNotFoundError: Brew not found (after searching max_steps_back)
            RuntimeError: Other HTTP errors or connection issues
        """
        params = {}
        if max_steps_back > 0:
            params["max_steps_back"] = max_steps_back
            
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew_at/{timestamp}/{brew_type}", params=params)
        
        if not response:
            error_msg = f"Failed to get brew at {timestamp} ({brew_type}): No response received"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
            
        if response.status_code == 200:
            brew_data = response.json()
            metadata = brew_data.get('_search_metadata', {})
            steps_back = metadata.get('steps_back', 0)
            found_brew_id = metadata.get('found_brew_id', 'unknown')
            
            if steps_back > 0:
                logger.info(f"✅ Retrieved brew: {found_brew_id} (stepped back {steps_back} intervals)")
            else:
                logger.info(f"✅ Retrieved brew: {found_brew_id}")
            return brew_data
        elif response.status_code == 400:
            error_msg = f"Invalid request parameters for timestamp {timestamp}, brew_type {brew_type}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        elif response.status_code == 404:
            error_msg = f"Brew not found at {timestamp} ({brew_type}) with max_steps_back={max_steps_back}"
            logger.warning(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        else:
            error_msg = f"Error retrieving brew at {timestamp}: HTTP {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def get_brew_location(self, brew_id: str) -> Optional[str]:
        """
        Get the storage location for a brew.
        
        Args:
            brew_id: The brew ID to check
            
        Returns:
            Location string ("redis", "unfull", "minio") or None if not found
        """
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew/{brew_id}/location")
        
        if not response:
            logger.error(f"❌ Failed to get location for {brew_id}: No response received")
            return None
            
        if response.status_code == 200:
            location_data = response.json()
            location = location_data.get('location')
            logger.info(f"📍 Brew {brew_id} location: {location}")
            return location
        elif response.status_code == 404:
            logger.warning(f"❌ Brew not found: {brew_id}")
            return None
        else:
            logger.error(f"❌ Error getting location for {brew_id}: {response.status_code}")
            return None
    
    def get_brews_batch(self, brew_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple brews by IDs (max 1000).
        
        Args:
            brew_ids: List of brew IDs to retrieve
            
        Returns:
            List of brew data dictionaries (only found brews)
        """
        if len(brew_ids) > 1000:
            logger.warning(f"⚠️ Truncating batch request from {len(brew_ids)} to 1000 brews")
            brew_ids = brew_ids[:1000]
        
        payload = {"ids": brew_ids}
        response = self._make_request_with_retry('POST', f"{self.base_url}/brews/batch", json=payload)
        
        if not response:
            logger.error("❌ Failed batch request: No response received")
            return []
            
        if response.status_code == 200:
            brews_data = response.json()
            logger.info(f"✅ Retrieved {len(brews_data)} brews from batch of {len(brew_ids)}")
            return brews_data
        else:
            logger.error(f"❌ Error in batch request: {response.status_code}")
            return []
    
    def query_parquet_range(self, 
                           type_filter: str, 
                           request_field: str, 
                           filter_field: str, 
                           from_val: Union[int, float], 
                           to_val: Union[int, float]) -> List[Dict[str, Any]]:
        """
        Query brews by numeric range on a field - PARQUET FILES ONLY.
        
        Args:
            type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            request_field: Field to return in results
            filter_field: Field to filter on
            from_val: Minimum value (inclusive)
            to_val: Maximum value (inclusive)
            
        Returns:
            List of {"brew_id": ..., "value": ...} dictionaries
        """
        params = {
            "type": type_filter,
            "request_field": request_field,
            "filter_field": filter_field,
            "from": from_val,
            "to": to_val
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/query/parquet/range", params=params)
        
        if not response:
            logger.error("❌ Failed parquet range query: No response received")
            return []
            
        if response.status_code == 200:
            results = response.json()
            logger.info(f"✅ Parquet range query returned {len(results)} results")
            return results
        elif response.status_code == 503:
            logger.warning("⚠️ Service temporarily unavailable - writer is active, try again later")
            return []
        else:
            logger.error(f"❌ Parquet range query error: {response.status_code} - {response.text}")
            return []
    
    def query_parquet_equal(self, 
                           type_filter: str, 
                           request_field: str, 
                           filter_field: str, 
                           filter_value: Union[str, int, float]) -> List[Dict[str, Any]]:
        """
        Query brews by exact match on a field - PARQUET FILES ONLY.
        
        Args:
            type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            request_field: Field to return in results
            filter_field: Field to filter on
            filter_value: Exact value to match
            
        Returns:
            List of {"brew_id": ..., "value": ...} dictionaries
        """
        params = {
            "type": type_filter,
            "request_field": request_field,
            "filter_field": filter_field,
            "filter_value": filter_value
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/query/parquet/equal", params=params)
        
        if not response:
            logger.error("❌ Failed parquet equal query: No response received")
            return []
            
        if response.status_code == 200:
            results = response.json()
            logger.info(f"✅ Parquet equal query returned {len(results)} results")
            return results
        elif response.status_code == 503:
            logger.warning("⚠️ Service temporarily unavailable - writer is active, try again later")
            return []
        else:
            logger.error(f"❌ Parquet equal query error: {response.status_code} - {response.text}")
            return []
    
    def get_baseline(self, baseline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a baseline by ID.
        
        Args:
            baseline_id: The baseline ID/key to retrieve
            
        Returns:
            Baseline data dictionary or None if not found
        """
        response = self._make_request_with_retry('GET', f"{self.base_url}/baseline/{baseline_id}")
        
        if not response:
            logger.error(f"❌ Failed to get baseline {baseline_id}: No response received")
            return None
            
        if response.status_code == 200:
            baseline_data = response.json()
            logger.info(f"✅ Retrieved baseline: {baseline_id}")
            return baseline_data
        elif response.status_code == 404:
            logger.warning(f"❌ Baseline not found: {baseline_id}")
            return None
        else:
            logger.error(f"❌ Error retrieving baseline {baseline_id}: {response.status_code}")
            return None

    def list_baselines(self, brew_type: str = None, baseline_type: str = None) -> Optional[Dict[str, Any]]:
        """
        List available baselines with optional filtering.
        
        Args:
            brew_type: Optional filter by brew type (e.g., "5_min", "15_min", "30_min")
            baseline_type: Optional filter by baseline type (e.g., "30d", "hour_cadence_30d", "week_day_cadence_90d")
            
        Returns:
            Dictionary containing:
            - "total_count": Total number of matching baselines
            - "baselines": List of baseline names/keys
            - "filters_applied": Summary of applied filters
            - "brew_types_found": List of unique brew types in results
            - "baseline_types_found": List of unique baseline types in results
            
        Examples:
            - list_baselines() -> All baselines
            - list_baselines(brew_type="5_min") -> All 5-minute baselines  
            - list_baselines(baseline_type="30d") -> All 30-day baselines
            - list_baselines(brew_type="5_min", baseline_type="hour_cadence_30d") -> Combined filtering
        """
        params = {}
        if brew_type:
            params["brew_type"] = brew_type
        if baseline_type:
            params["baseline_type"] = baseline_type
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/baselines/list", params=params)
        
        if not response:
            logger.error("❌ Failed to list baselines: No response received")
            return None
            
        if response.status_code == 200:
            data = response.json()
            total_count = data.get('total_count', 0)
            filters_str = ", ".join(data.get('filters_applied', ['none']))
            logger.info(f"✅ Listed {total_count} baselines (filters: {filters_str})")
            return data
        elif response.status_code == 503:
            logger.warning("❌ Baseline storage not available")
            return None
        else:
            logger.error(f"❌ Error listing baselines: {response.status_code}")
            return None
    
    def get_brew_with_baselines(self, brew_id: str, baseline_types: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a brew along with its relevant baselines.
        
        Args:
            brew_id: The brew ID to retrieve
            baseline_types: Optional list of baseline types to filter by 
                           (e.g., ["4h", "24h", "hour_cadence_30d"])
            
        Returns:
            Dictionary containing:
            - "brew": brew data
            - "baselines": dictionary mapping baseline types to baseline data
            Or None if brew not found
        """
        params = {}
        if baseline_types:
            params["baseline_types"] = baseline_types
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew/{brew_id}/baselines", params=params)
        
        if not response:
            logger.error(f"❌ Failed to get brew with baselines {brew_id}: No response received")
            return None
            
        if response.status_code == 200:
            data = response.json()
            baseline_count = len(data.get('baselines', {}))
            logger.info(f"✅ Retrieved brew {brew_id} with {baseline_count} baselines")
            return data
        elif response.status_code == 404:
            logger.warning(f"❌ Brew not found: {brew_id}")
            return None
        else:
            logger.error(f"❌ Error retrieving brew with baselines {brew_id}: {response.status_code}")
            return None

    def get_brew_with_baselines_at(self, timestamp: str, brew_type: str, max_steps_back: int = 0, baseline_types: List[str] = None) -> Dict[str, Any]:
        """
        Get a brew by timestamp and type with baselines, with optional backward search.
        
        Args:
            timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
            brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
            baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
            
        Returns:
            Dictionary containing:
            - "brew": brew data (flattened format)
            - "baselines": dictionary mapping baseline types to baseline data
            - "_search_metadata": information about the timestamp search process
            
        Raises:
            ValueError: Invalid timestamp format or brew_type
            FileNotFoundError: Brew not found (after searching max_steps_back)
            RuntimeError: Other HTTP errors or connection issues
        """
        params = {}
        if max_steps_back > 0:
            params["max_steps_back"] = max_steps_back
        if baseline_types:
            params["baseline_types"] = baseline_types
            
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew_with_baselines_at/{timestamp}/{brew_type}", params=params)
        
        if not response:
            error_msg = f"Failed to get brew with baselines at {timestamp} ({brew_type}): No response received"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
            
        if response.status_code == 200:
            data = response.json()
            search_metadata = data.get('_search_metadata', {})
            baseline_metadata = data.get('_baseline_metadata', {})
            steps_back = search_metadata.get('steps_back', 0)
            found_brew_id = search_metadata.get('found_brew_id', 'unknown')
            baseline_count = baseline_metadata.get('baseline_count', 0)
            
            if steps_back > 0:
                logger.info(f"✅ Retrieved brew with baselines: {found_brew_id} (stepped back {steps_back} intervals, {baseline_count} baselines)")
            else:
                logger.info(f"✅ Retrieved brew with baselines: {found_brew_id} ({baseline_count} baselines)")
            return data
        elif response.status_code == 400:
            error_msg = f"Invalid request parameters for timestamp {timestamp}, brew_type {brew_type}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        elif response.status_code == 404:
            error_msg = f"Brew not found at {timestamp} ({brew_type}) with max_steps_back={max_steps_back}"
            logger.warning(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        else:
            error_msg = f"Error retrieving brew with baselines at {timestamp}: HTTP {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def get_brew_text(self, brew_id: str, baseline_types: List[str] = None) -> Optional[str]:
        """
        Get a brew processed into text format using texter.
        
        Args:
            brew_id: The brew ID to retrieve
            baseline_types: Optional list of baseline types to filter by 
                           (e.g., ["4h", "24h", "hour_cadence_30d"])
            
        Returns:
            Processed brew text or None if brew not found
        """
        params = {}
        if baseline_types:
            params["baseline_types"] = baseline_types
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew/{brew_id}/text", params=params)
        
        if not response:
            logger.error(f"❌ Failed to get brew text {brew_id}: No response received")
            return None
            
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Retrieved processed text for brew {brew_id}")
            return data.get('text')
        elif response.status_code == 404:
            logger.warning(f"❌ Brew or baselines not found: {brew_id}")
            return None
        elif response.status_code == 500:
            logger.error(f"❌ Error processing text for brew {brew_id}: {response.text}")
            return None
        else:
            logger.error(f"❌ Error retrieving brew text {brew_id}: {response.status_code}")
            return None

    def get_brew_text_at(self, timestamp: str, brew_type: str, max_steps_back: int = 0, baseline_types: List[str] = None) -> str:
        """
        Get a brew by timestamp and type processed into text format, with optional backward search.
        
        Args:
            timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
            brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
            baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
            
        Returns:
            Processed brew text
            
        Raises:
            ValueError: Invalid timestamp format or brew_type
            FileNotFoundError: Brew not found (after searching max_steps_back) or no baselines found
            RuntimeError: Other HTTP errors or connection issues
        """
        params = {}
        if max_steps_back > 0:
            params["max_steps_back"] = max_steps_back
        if baseline_types:
            params["baseline_types"] = baseline_types
            
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew_text_at/{timestamp}/{brew_type}", params=params)
        
        if not response:
            error_msg = f"Failed to get brew text at {timestamp} ({brew_type}): No response received"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
            
        if response.status_code == 200:
            data = response.json()
            search_metadata = data.get('_search_metadata', {})
            baseline_metadata = data.get('_baseline_metadata', {})
            text_metadata = data.get('_text_metadata', {})
            
            steps_back = search_metadata.get('steps_back', 0)
            found_brew_id = search_metadata.get('found_brew_id', 'unknown')
            baseline_count = baseline_metadata.get('baseline_count', 0)
            text_length = text_metadata.get('text_length', 0)
            
            if steps_back > 0:
                logger.info(f"✅ Generated text for brew: {found_brew_id} (stepped back {steps_back} intervals, {baseline_count} baselines, {text_length} chars)")
            else:
                logger.info(f"✅ Generated text for brew: {found_brew_id} ({baseline_count} baselines, {text_length} chars)")
            
            return data.get('text', '')
        elif response.status_code == 400:
            error_msg = f"Invalid request parameters for timestamp {timestamp}, brew_type {brew_type}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        elif response.status_code == 404:
            error_msg = f"Brew or baselines not found at {timestamp} ({brew_type}) with max_steps_back={max_steps_back}"
            logger.warning(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        elif response.status_code == 500:
            error_msg = f"Error processing text for brew at {timestamp}: {response.text}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        else:
            error_msg = f"Error retrieving brew text at {timestamp}: HTTP {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def get_brew_enhanced_at(self, timestamp: str, brew_type: str, max_steps_back: int = 0, baseline_types: List[str] = None) -> Dict[str, Any]:
        """
        Get a brew by timestamp and type with enhanced baseline analysis, with optional backward search.
        
        Args:
            timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
            brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
            baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
            
        Returns:
            Enhanced brew structure with baseline analysis
            
        Raises:
            ValueError: Invalid timestamp format or brew_type
            FileNotFoundError: Brew not found (after searching max_steps_back) or no baselines found
            RuntimeError: Other HTTP errors or connection issues
        """
        params = {}
        if max_steps_back > 0:
            params["max_steps_back"] = max_steps_back
        if baseline_types:
            params["baseline_types"] = baseline_types
            
        response = self._make_request_with_retry('GET', f"{self.base_url}/brew_enhanced_at/{timestamp}/{brew_type}", params=params)
        
        if not response:
            error_msg = f"Failed to get enhanced brew at {timestamp} ({brew_type}): No response received"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
            
        if response.status_code == 200:
            data = response.json()
            search_metadata = data.get('_search_metadata', {})
            baseline_metadata = data.get('_baseline_metadata', {})
            enhancement_metadata = data.get('_enhancement_metadata', {})
            
            steps_back = search_metadata.get('steps_back', 0)
            found_brew_id = search_metadata.get('found_brew_id', 'unknown')
            baseline_count = baseline_metadata.get('baseline_count', 0)
            fields_analyzed = enhancement_metadata.get('fields_analyzed', 0)
            
            if steps_back > 0:
                logger.info(f"✅ Generated enhanced analysis for brew: {found_brew_id} (stepped back {steps_back} intervals, {baseline_count} baselines, {fields_analyzed} fields)")
            else:
                logger.info(f"✅ Generated enhanced analysis for brew: {found_brew_id} ({baseline_count} baselines, {fields_analyzed} fields)")
            
            return data
        elif response.status_code == 400:
            error_msg = f"Invalid request parameters for timestamp {timestamp}, brew_type {brew_type}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        elif response.status_code == 404:
            error_msg = f"Brew or baselines not found at {timestamp} ({brew_type}) with max_steps_back={max_steps_back}"
            logger.warning(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        elif response.status_code == 500:
            error_msg = f"Error processing enhanced analysis for brew at {timestamp}: {response.text}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        else:
            error_msg = f"Error retrieving enhanced brew at {timestamp}: HTTP {response.status_code}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def batch_query_parquet_range(self,
                                 type_filter: str,
                                 request_fields: List[str],
                                 filter_field: str,
                                 from_val: Union[int, float],
                                 to_val: Union[int, float]) -> Optional[Dict[str, List[Any]]]:
        """
        Batch query brews by numeric range on a field - PARQUET FILES ONLY.
        
        Performance optimized for multiple fields in single query.
        Returns optimized format without brew_ids for maximum performance.
        
        Args:
            type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            request_fields: List of fields to return in results
            filter_field: Field to filter on
            from_val: Minimum value (inclusive)
            to_val: Maximum value (inclusive)
            
        Returns:
            Dict with field names as keys and arrays of values as values
            Format: {"field1": [val1, val2, ...], "field2": [val3, val4, ...]}
        """
        payload = {
            "type": type_filter,
            "request_fields": request_fields,
            "filter_field": filter_field,
            "from_val": from_val,
            "to_val": to_val
        }
        
        logger.info(f"🔍 Batch range query: {len(request_fields)} fields, range {from_val}-{to_val}")
        
        response = self._make_request_with_retry('POST', f"{self.base_url}/query/parquet/batch_range", json=payload)
        
        if not response:
            logger.error("❌ Failed batch range query: No response received")
            return None
            
        if response.status_code == 200:
            results = response.json()
            sample_field = next(iter(results.keys())) if results else None
            data_points = len(results[sample_field]) if sample_field and results[sample_field] else 0
            logger.info(f"✅ Batch range query returned {data_points} data points")
            return results
        elif response.status_code == 503:
            logger.warning("⚠️ Service temporarily unavailable - writer is active, try again later")
            return None
        else:
            logger.error(f"❌ Batch range query error: {response.status_code} - {response.text}")
            return None

    def cadence_batch_query_parquet(self,
                                   type_filter: str,
                                   request_fields: List[str],
                                   filter_field: str,
                                   ranges_list: List[List[Union[int, float]]]) -> Optional[Dict[str, List[Any]]]:
        """
        Cadence batch query brews by multiple numeric ranges - PARQUET FILES ONLY.
        
        Designed for querying data across multiple time periods (e.g., all Mondays from T to T-30D).
        Efficiently handles up to 200 ranges using optimized Polars OR conditions.
        
        Args:
            type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
            request_fields: List of fields to return in results
            filter_field: Field to filter on
            ranges_list: List of [min, max] range pairs (up to 200 ranges)
            
        Returns:
            Dict with field names as keys and arrays of values as values
            Format: {"field1": [val1, val2, ...], "field2": [val3, val4, ...]}
        """
        if len(ranges_list) > 200:
            logger.warning(f"⚠️ Truncating ranges_list from {len(ranges_list)} to 200")
            ranges_list = ranges_list[:200]
        
        payload = {
            "type": type_filter,
            "request_fields": request_fields,
            "filter_field": filter_field,
            "ranges_list": ranges_list
        }
        
        logger.info(f"🔍 Cadence batch query: {len(request_fields)} fields, {len(ranges_list)} ranges")
        
        response = self._make_request_with_retry('POST', f"{self.base_url}/query/parquet/cadence_batch", json=payload)
        
        if not response:
            logger.error("❌ Failed cadence batch query: No response received")
            return None
            
        if response.status_code == 200:
            results = response.json()
            sample_field = next(iter(results.keys())) if results else None
            data_points = len(results[sample_field]) if sample_field and results[sample_field] else 0
            logger.info(f"✅ Cadence batch query returned {data_points} data points")
            return results
        elif response.status_code == 503:
            logger.warning("⚠️ Service temporarily unavailable - writer is active, try again later")
            return None
        else:
            logger.error(f"❌ Cadence batch query error: {response.status_code} - {response.text}")
            return None
    
    def get_recent_brews(self, type_filter: str = "5_min", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Convenience method to get recent brews of a specific type.
        
        Args:
            type_filter: Time type (default: 5_min)
            limit: Maximum number of brews to return
            
        Returns:
            List of recent brew data dictionaries
        """
        logger.info(f"🔍 Getting {limit} recent {type_filter} brews...")
        
        # Query for brews with blocks_count field (most brews should have this)
        results = self.query_parquet_range(
            type_filter=type_filter,
            request_field="brew_id",
            filter_field="blocks_count",
            from_val=1,
            to_val=1000  # Large range to get all brews
        )
        
        if not results:
            logger.warning(f"⚠️ No brews found for type {type_filter}")
            return []
        
        # Sort by brew_id (which contains timestamp) and take most recent
        sorted_results = sorted(results, key=lambda x: x["brew_id"], reverse=True)
        recent_brew_ids = [r["brew_id"] for r in sorted_results[:limit]]
        
        # Get full brew data
        return self.get_brews_batch(recent_brew_ids)
    
    def get_location_stats(self, type_filter: str = "5_min", sample_size: int = 50) -> Dict[str, Any]:
        """
        Get statistics about brew storage locations for a given type.
        
        Args:
            type_filter: Time type to analyze
            sample_size: Number of recent brews to check
            
        Returns:
            Dictionary with location statistics
        """
        logger.info(f"📊 Analyzing storage locations for {sample_size} recent {type_filter} brews...")
        
        # Get recent brew IDs
        results = self.query_parquet_range(
            type_filter=type_filter,
            request_field="brew_id",
            filter_field="blocks_count",
            from_val=1,
            to_val=1000
        )
        
        if not results:
            return {"error": "No brews found", "sample_size": 0}
        
        # Sort and take sample
        sorted_results = sorted(results, key=lambda x: x["brew_id"], reverse=True)
        sample_brew_ids = [r["brew_id"] for r in sorted_results[:sample_size]]
        
        # Check locations
        location_counts = {"redis": 0, "unfull": 0, "minio": 0, "unknown": 0}
        
        for brew_id in sample_brew_ids:
            location = self.get_brew_location(brew_id)
            if location in location_counts:
                location_counts[location] += 1
            else:
                location_counts["unknown"] += 1
        
        stats = {
            "type_filter": type_filter,
            "sample_size": len(sample_brew_ids),
            "location_distribution": location_counts,
            "percentages": {
                loc: round((count / len(sample_brew_ids)) * 100, 1)
                for loc, count in location_counts.items()
            }
        }
        
        logger.info(f"📊 Location stats: {stats['percentages']}")
        return stats
    
    def to_dataframe(self, brews_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert brew data to a pandas DataFrame for analysis.
        
        Args:
            brews_data: List of brew dictionaries
            
        Returns:
            DataFrame with brew data
        """
        if not brews_data:
            logger.warning("⚠️ No data to convert to DataFrame")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(brews_data)
            logger.info(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to create DataFrame: {e}")
            return pd.DataFrame()

    def get_swaps(self,
                  from_timestamp: Optional[int] = None,
                  to_timestamp: Optional[int] = None,
                  token_in: Optional[str] = None,
                  token_out: Optional[str] = None,
                  pool_address: Optional[str] = None,
                  trader_address: Optional[str] = None,
                  max_entries: int = 1000) -> List[Dict[str, Any]]:
        """
        Get swaps based on filtering criteria.
        
        Args:
            from_timestamp: Minimum timestamp (inclusive). None means no lower bound.
            to_timestamp: Maximum timestamp (inclusive). None means no upper bound.
            token_in: Token input address (hex format with 0x prefix). None means any token.
            token_out: Token output address (hex format with 0x prefix). None means any token.
            pool_address: Pool address (hex format with 0x prefix). None means any pool.
            trader_address: Trader address (hex format with 0x prefix). None means any trader.
            max_entries: Maximum number of entries to return (1-10000, default: 1000)
            
        Returns:
            List of swap dictionaries matching the criteria
        """
        params = {}
        if from_timestamp is not None:
            params["from_timestamp"] = from_timestamp
        if to_timestamp is not None:
            params["to_timestamp"] = to_timestamp
        if token_in is not None:
            params["token_in"] = token_in
        if token_out is not None:
            params["token_out"] = token_out
        if pool_address is not None:
            params["pool_address"] = pool_address
        if trader_address is not None:
            params["trader_address"] = trader_address
        if max_entries != 1000:
            params["max_entries"] = max_entries
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps", params=params)
        
        if not response:
            logger.error("❌ Failed to get swaps: No response received")
            return []
            
        if response.status_code == 200:
            swaps_data = response.json()
            logger.info(f"✅ Retrieved {len(swaps_data)} swaps")
            return swaps_data
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for swaps query: {response.text}")
            return []
        else:
            logger.error(f"❌ Error retrieving swaps: {response.status_code}")
            return []
    
    def get_swaps_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics about available swap files.
        
        Returns:
            Dictionary containing statistics about stored swap files or None if error
        """
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/stats")
        
        if not response:
            logger.error("❌ Failed to get swaps stats: No response received")
            return None
            
        if response.status_code == 200:
            stats_data = response.json()
            total_files = stats_data.get('total_files', 0)
            total_size_mb = stats_data.get('total_size_mb', 0)
            logger.info(f"✅ Retrieved swaps stats: {total_files} files, {total_size_mb:.1f}MB")
            return stats_data
        else:
            logger.error(f"❌ Error retrieving swaps stats: {response.status_code}")
            return None

    def get_most_traded_count_tokens(self,
                                   timestamp_start: int,
                                   timestamp_end: int,
                                   side: str = "both",
                                   against: List[str] = None,
                                   limit: int = 100,
                                   include_distributions: bool = False,
                                   exclude_weth_stables: bool = True,
                                   return_text: bool = False) -> Union[List[Dict[str, Any]], str]:
        """
        Get the most traded tokens by count within a time range.
        
        Args:
            timestamp_start: Start timestamp (inclusive)
            timestamp_end: End timestamp (inclusive)
            side: 'bought', 'sold', or 'both' - which side to count
            against: List of token categories to trade against ['weth', 'stablecoins', 'other']
            limit: Maximum number of tokens to return (1-1000, default: 100)
            include_distributions: Include ETH and stablecoin swap size distributions
            exclude_weth_stables: Exclude WETH and stablecoin tokens from results (default: True)
            return_text: If True, returns formatted markdown text instead of JSON data
            
        Returns:
            List of dictionaries with token trading data by count, or formatted text if return_text=True
        """
        if against is None:
            against = ["weth", "stablecoins", "other"]
            
        params = {
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "side": side,
            "limit": limit,
            "include_distributions": include_distributions,
            "exclude_weth_stables": exclude_weth_stables,
            "return_text": return_text
        }
        for category in against:
            params.setdefault("against", []).append(category)
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/most_traded_count_tokens", params=params)
        
        if not response:
            return "" if return_text else []
            
        if response.status_code == 200:
            if return_text:
                logger.info(f"✅ Retrieved most traded tokens by count as text")
                return response.text
            else:
                results = response.json()
                logger.info(f"✅ Retrieved {len(results)} most traded tokens by count")
                return results
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for most traded count tokens: {response.text}")
            return "" if return_text else []
        else:
            logger.error(f"❌ Error retrieving most traded count tokens: {response.status_code}")
            return "" if return_text else []

    def get_most_traded_value_tokens(self,
                                   timestamp_start: int,
                                   timestamp_end: int,
                                   side: str = "both",
                                   against: str = "weth",
                                   limit: int = 100,
                                   include_distributions: bool = False,
                                   exclude_weth_stables: bool = True,
                                   return_text: bool = False) -> Union[List[Dict[str, Any]], str]:
        """
        Get the most traded tokens by volume within a time range.
        
        Args:
            timestamp_start: Start timestamp (inclusive)
            timestamp_end: End timestamp (inclusive)
            side: 'bought', 'sold', or 'both' - which side to count
            against: 'weth' or 'stablecoins' - which currency to measure volume in
            limit: Maximum number of tokens to return (1-1000, default: 100)
            include_distributions: Include swap size distribution statistics
            exclude_weth_stables: Exclude WETH and stablecoin tokens from results (default: True)
            return_text: If True, returns formatted markdown text instead of JSON data
            
        Returns:
            List of dictionaries with token trading data by volume, or formatted text if return_text=True
        """
        params = {
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "side": side,
            "against": against,
            "limit": limit,
            "include_distributions": include_distributions,
            "exclude_weth_stables": exclude_weth_stables,
            "return_text": return_text
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/most_traded_value_tokens", params=params)
        
        if not response:
            return "" if return_text else []
            
        if response.status_code == 200:
            if return_text:
                logger.info(f"✅ Retrieved most traded tokens by {against} volume as text")
                return response.text
            else:
                results = response.json()
                logger.info(f"✅ Retrieved {len(results)} most traded tokens by {against} volume")
                return results
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for most traded value tokens: {response.text}")
            return "" if return_text else []
        else:
            logger.error(f"❌ Error retrieving most traded value tokens: {response.status_code}")
            return "" if return_text else []

    def get_token_stats(self,
                       token_address: str,
                       timestamp_start: int,
                       timestamp_end: int,
                       include_distributions: bool = False,
                       return_text: bool = False) -> Union[Optional[Dict[str, Any]], str]:
        """
        Get detailed statistics for a specific token within a time range.
        
        Args:
            token_address: Token address to analyze (hex format with 0x prefix)
            timestamp_start: Start timestamp (inclusive)
            timestamp_end: End timestamp (inclusive)
            include_distributions: Include ETH and stablecoin swap size distributions
            return_text: Return formatted text instead of JSON data
            
        Returns:
            Dictionary with token statistics, formatted text (if return_text=True), or None if error
        """
        params = {
            "token_address": token_address,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "include_distributions": include_distributions,
            "return_text": return_text
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/token_stats", params=params)
        
        if not response:
            return None
            
        if response.status_code == 200:
            if return_text:
                # Return text response directly
                text_data = response.text
                logger.info(f"✅ Retrieved token stats text for {token_address}")
                return text_data
            else:
                # Return JSON response
                stats_data = response.json()
                total_swaps = stats_data.get('total_swaps', 0)
                logger.info(f"✅ Retrieved token stats for {token_address}: {total_swaps} swaps")
                return stats_data
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for token stats: {response.text}")
            return None
        else:
            logger.error(f"❌ Error retrieving token stats: {response.status_code}")
            return None

    def get_trader_stats(self,
                        trader_address: str,
                        timestamp_start: int,
                        timestamp_end: int,
                        include_distributions: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific trader within a time range.
        
        Args:
            trader_address: Trader address to analyze (hex format with 0x prefix)
            timestamp_start: Start timestamp (inclusive)
            timestamp_end: End timestamp (inclusive)
            include_distributions: Include ETH and stablecoin swap size distributions
            
        Returns:
            Dictionary with trader statistics or None if error
        """
        params = {
            "trader_address": trader_address,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "include_distributions": include_distributions
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/trader_stats", params=params)
        
        if not response:
            return None
            
        if response.status_code == 200:
            stats_data = response.json()
            total_swaps = stats_data.get('total_swaps', 0)
            logger.info(f"✅ Retrieved trader stats for {trader_address}: {total_swaps} swaps")
            return stats_data
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for trader stats: {response.text}")
            return None
        else:
            logger.error(f"❌ Error retrieving trader stats: {response.status_code}")
            return None

    def get_trading_stats(self,
                         timestamp_start: int,
                         timestamp_end: int,
                         side: str = "both",
                         against: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get trading statistics summary for the specified criteria.
        
        Args:
            timestamp_start: Start timestamp (inclusive)
            timestamp_end: End timestamp (inclusive)
            side: 'bought', 'sold', or 'both'
            against: List of token categories to trade against
            
        Returns:
            Dictionary with trading statistics summary or None if error
        """
        if against is None:
            against = ["weth", "stablecoins", "other"]
            
        params = {
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "side": side
        }
        for category in against:
            params.setdefault("against", []).append(category)
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/swaps/trading_stats", params=params)
        
        if not response:
            logger.error("❌ Failed to get trading stats: No response received")
            return None
            
        if response.status_code == 200:
            stats_data = response.json()
            total_swaps = stats_data.get('total_swaps', 0)
            unique_tokens = stats_data.get('unique_tokens', 0)
            logger.info(f"✅ Retrieved trading stats: {total_swaps} swaps, {unique_tokens} unique tokens")
            return stats_data
        elif response.status_code == 400:
            logger.error(f"❌ Invalid parameters for trading stats: {response.text}")
            return None
        else:
            logger.error(f"❌ Error retrieving trading stats: {response.status_code}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about available brews and service status.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "connection_status": "unknown",
            "service_url": self.base_url,
            "gateway_url": self.gateway_url,
            "fallback_url": self.fallback_url,
            "using_fallback": self.use_fallback,
            "api_key_configured": bool(self.api_key),
            "storage_tiers": ["redis", "unfull", "minio"]
        }
        
        response = self._make_request_with_retry('GET', f"{self.base_url}/health")
        if response and response.status_code == 200:
            stats["connection_status"] = "healthy"
        else:
            stats["connection_status"] = "unhealthy"
            
        logger.info(f"📊 Secure Brews Connector Stats: {stats}")
        return stats


# Create a compatibility alias for existing code
BrewsReaderConnector = SecureBrewsConnector


# Updated convenience functions
def quick_get_brew(brew_id: str, 
                   gateway_url: str = "http://localhost:8081",
                   api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Quick function to get a single brew using secure connection.
    
    Args:
        brew_id: The brew ID to retrieve
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Brew data dictionary or None if not found
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew(brew_id)


def quick_get_recent_brews(type_filter: str = "5_min", 
                          limit: int = 5, 
                          gateway_url: str = "http://localhost:8081",
                          api_key: str = None) -> pd.DataFrame:
    """
    Quick function to get recent brews as a DataFrame using secure connection.
    
    Args:
        type_filter: Time type (default: 5_min)
        limit: Maximum number of brews to return
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        DataFrame with recent brew data
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    
    # Query for brews with blocks_count field (most brews should have this)
    results = connector.query_parquet_range(
        type_filter=type_filter,
        request_field="brew_id",
        filter_field="blocks_count",
        from_val=1,
        to_val=1000  # Large range to get all brews
    )
    
    if not results:
        logger.warning(f"⚠️ No brews found for type {type_filter}")
        return pd.DataFrame()
    
    # Sort by brew_id (which contains timestamp) and take most recent
    sorted_results = sorted(results, key=lambda x: x["brew_id"], reverse=True)
    recent_brew_ids = [r["brew_id"] for r in sorted_results[:limit]]
    
    # Get full brew data
    brews_data = connector.get_brews_batch(recent_brew_ids)
    return pd.DataFrame(brews_data) if brews_data else pd.DataFrame()


def quick_get_brew_at_format(brew_id: str, format: str,
                            gateway_url: str = "http://localhost:8081",
                            api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Quick function to get a brew in specified format using secure connection.
    
    Args:
        brew_id: The brew ID to retrieve
        format: 'nested' or 'flattened'
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Brew data dictionary in requested format or None if not found
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_at_format(brew_id, format)


def quick_get_brew_at(timestamp: str, brew_type: str, max_steps_back: int = 0,
                     gateway_url: str = "http://localhost:8081",
                     api_key: str = None) -> Dict[str, Any]:
    """
    Quick function to get a brew by timestamp and type using secure connection.
    
    Args:
        timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
        brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Brew data dictionary
        
    Raises:
        ValueError: Invalid timestamp format or brew_type
        FileNotFoundError: Brew not found (after searching max_steps_back)
        RuntimeError: Other HTTP errors or connection issues
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_at(timestamp, brew_type, max_steps_back)


def quick_query_parquet_range(type_filter: str, 
                             request_field: str, 
                             filter_field: str, 
                             from_val: Union[int, float], 
                             to_val: Union[int, float],
                             gateway_url: str = "http://localhost:8081",
                             api_key: str = None) -> pd.DataFrame:
    """
    Quick function to perform a parquet range query and return as DataFrame using secure connection.
    
    Args:
        type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        request_field: Field to return in results
        filter_field: Field to filter on
        from_val: Minimum value (inclusive)
        to_val: Maximum value (inclusive)
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        DataFrame with query results
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.query_parquet_range(type_filter, request_field, filter_field, from_val, to_val)
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_query_parquet_equal(type_filter: str, 
                             request_field: str, 
                             filter_field: str, 
                             filter_value: Union[str, int, float],
                             gateway_url: str = "http://localhost:8081",
                             api_key: str = None) -> pd.DataFrame:
    """
    Quick function to perform a parquet equal query and return as DataFrame using secure connection.
    
    Args:
        type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        request_field: Field to return in results
        filter_field: Field to filter on
        filter_value: Exact value to match
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        DataFrame with query results
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.query_parquet_equal(type_filter, request_field, filter_field, filter_value)
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_get_location_stats(type_filter: str = "5_min", 
                            sample_size: int = 50,
                            gateway_url: str = "http://localhost:8081",
                            api_key: str = None) -> Dict[str, Any]:
    """
    Quick function to get storage location statistics using secure connection.
    
    Args:
        type_filter: Time type to analyze
        sample_size: Number of recent brews to check
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Dictionary with location statistics
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_location_stats(type_filter, sample_size)


def quick_batch_query_parquet_range(type_filter: str,
                                   request_fields: List[str],
                                   filter_field: str,
                                   from_val: Union[int, float],
                                   to_val: Union[int, float],
                                   gateway_url: str = "http://localhost:8081",
                                   api_key: str = None) -> pd.DataFrame:
    """
    Quick function to perform a batch parquet range query and return as DataFrame using secure connection.
    
    Args:
        type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        request_fields: List of fields to return in results
        filter_field: Field to filter on
        from_val: Minimum value (inclusive)
        to_val: Maximum value (inclusive)
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        DataFrame with query results
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.batch_query_parquet_range(type_filter, request_fields, filter_field, from_val, to_val)
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_cadence_batch_query_parquet(type_filter: str,
                                     request_fields: List[str],
                                     filter_field: str,
                                     ranges_list: List[List[Union[int, float]]],
                                     gateway_url: str = "http://localhost:8081",
                                     api_key: str = None) -> pd.DataFrame:
    """
    Quick function to perform a cadence batch parquet query and return as DataFrame using secure connection.
    
    Args:
        type_filter: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        request_fields: List of fields to return in results
        filter_field: Field to filter on
        ranges_list: List of [min, max] range pairs (up to 200 ranges)
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        DataFrame with query results
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.cadence_batch_query_parquet(type_filter, request_fields, filter_field, ranges_list)
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_get_baseline(baseline_id: str,
                      gateway_url: str = "http://localhost:8081",
                      api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Quick function to get a baseline by ID using secure connection.
    
    Args:
        baseline_id: The baseline ID/key to retrieve
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Baseline data dictionary or None if not found
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_baseline(baseline_id)


def quick_get_brew_with_baselines(brew_id: str, 
                                 baseline_types: List[str] = None,
                                 gateway_url: str = "http://localhost:8081",
                                 api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Quick function to get a brew with its relevant baselines using secure connection.
    
    Args:
        brew_id: The brew ID to retrieve
        baseline_types: Optional list of baseline types to filter by 
                       (e.g., ["4h", "24h", "hour_cadence_30d"])
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Dictionary containing brew data and baselines or None if brew not found
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_with_baselines(brew_id, baseline_types)


def quick_get_brew_with_baselines_at(timestamp: str, brew_type: str, max_steps_back: int = 0,
                                    baseline_types: List[str] = None,
                                    gateway_url: str = "http://localhost:8081",
                                    api_key: str = None) -> Dict[str, Any]:
    """
    Quick function to get a brew by timestamp and type with baselines using secure connection.
    
    Args:
        timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
        brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
        baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Dictionary containing brew data and baselines with search metadata
        
    Raises:
        ValueError: Invalid timestamp format or brew_type
        FileNotFoundError: Brew not found (after searching max_steps_back)
        RuntimeError: Other HTTP errors or connection issues
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_with_baselines_at(timestamp, brew_type, max_steps_back, baseline_types)


def quick_get_brew_text(brew_id: str, 
                       baseline_types: List[str] = None,
                       gateway_url: str = "http://localhost:8081",
                       api_key: str = None) -> Optional[str]:
    """
    Quick function to get a brew processed into text format using texter with secure connection.
    
    Args:
        brew_id: The brew ID to retrieve
        baseline_types: Optional list of baseline types to filter by 
                       (e.g., ["4h", "24h", "hour_cadence_30d"])
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Processed brew text or None if brew not found
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_text(brew_id, baseline_types)


def quick_get_brew_text_at(timestamp: str, brew_type: str, max_steps_back: int = 0,
                          baseline_types: List[str] = None,
                          gateway_url: str = "http://localhost:8081",
                          api_key: str = None) -> str:
    """
    Quick function to get a brew by timestamp and type processed into text format using secure connection.
    
    Args:
        timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
        brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
        baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Processed brew text
        
    Raises:
        ValueError: Invalid timestamp format or brew_type
        FileNotFoundError: Brew not found (after searching max_steps_back) or no baselines found
        RuntimeError: Other HTTP errors or connection issues
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_text_at(timestamp, brew_type, max_steps_back, baseline_types)


def quick_get_brew_enhanced_at(timestamp: str, brew_type: str, max_steps_back: int = 0,
                              baseline_types: List[str] = None,
                              gateway_url: str = "http://localhost:8081",
                              api_key: str = None) -> Dict[str, Any]:
    """
    Quick function to get a brew by timestamp and type with enhanced baseline analysis using secure connection.
    
    Args:
        timestamp: Timestamp in various formats (epoch, ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DD)
        brew_type: Time type (1_min, 3_min, 5_min, 15_min, 30_min, 60_min)
        max_steps_back: Maximum number of interval steps to go back if brew not found (default: 0)
        baseline_types: Optional list of baseline types to filter by (e.g., ["4h", "24h", "hour_cadence_30d"])
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Enhanced brew structure with baseline analysis
        
    Raises:
        ValueError: Invalid timestamp format or brew_type
        FileNotFoundError: Brew not found (after searching max_steps_back) or no baselines found
        RuntimeError: Other HTTP errors or connection issues
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_brew_enhanced_at(timestamp, brew_type, max_steps_back, baseline_types)


def quick_list_baselines(brew_type: str = None, 
                        baseline_type: str = None,
                        gateway_url: str = "http://localhost:8081",
                        api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Quick function to list available baselines with optional filtering using secure connection.
    
    Args:
        brew_type: Optional filter by brew type (e.g., "5_min", "15_min", "30_min")
        baseline_type: Optional filter by baseline type (e.g., "30d", "hour_cadence_30d", "week_day_cadence_90d")
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        Dictionary containing baseline list and statistics, or None if error
        
    Examples:
        - quick_list_baselines() -> All baselines
        - quick_list_baselines(brew_type="5_min") -> All 5-minute baselines  
        - quick_list_baselines(baseline_type="30d") -> All 30-day baselines
        - quick_list_baselines(brew_type="5_min", baseline_type="hour_cadence_30d") -> Combined filtering
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.list_baselines(brew_type, baseline_type)


def quick_get_swaps(from_timestamp: Optional[int] = None,
                   to_timestamp: Optional[int] = None,
                   token_in: Optional[str] = None,
                   token_out: Optional[str] = None,
                   pool_address: Optional[str] = None,
                   trader_address: Optional[str] = None,
                   max_entries: int = 1000,
                   gateway_url: str = "http://localhost:8081",
                   api_key: str = None) -> List[Dict[str, Any]]:
    """
    Quick function to get swaps based on filtering criteria using secure connection.
    
    Args:
        from_timestamp: Minimum timestamp (inclusive). None means no lower bound.
        to_timestamp: Maximum timestamp (inclusive). None means no upper bound.
        token_in: Token input address (hex format with 0x prefix). None means any token.
        token_out: Token output address (hex format with 0x prefix). None means any token.
        pool_address: Pool address (hex format with 0x prefix). None means any pool.
        trader_address: Trader address (hex format with 0x prefix). None means any trader.
        max_entries: Maximum number of entries to return (1-10000, default: 1000)
        gateway_url: Traefik gateway URL
        api_key: API key for authentication
        
    Returns:
        List of swap dictionaries matching the criteria
    """
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_swaps(
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        token_in=token_in,
        token_out=token_out,
        pool_address=pool_address,
        trader_address=trader_address,
        max_entries=max_entries
    )


def quick_get_swaps_stats(gateway_url: str = "http://localhost:8081",
                         api_key: str = None) -> Optional[Dict[str, Any]]:
    """Quick function to get statistics about available swap files using secure connection."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_swaps_stats()


def quick_get_swaps_as_dataframe(from_timestamp: Optional[int] = None,
                               to_timestamp: Optional[int] = None,
                               token_in: Optional[str] = None,
                               token_out: Optional[str] = None,
                               pool_address: Optional[str] = None,
                               trader_address: Optional[str] = None,
                               max_entries: int = 1000,
                               gateway_url: str = "http://localhost:8081",
                               api_key: str = None) -> pd.DataFrame:
    """Quick function to get swaps as a pandas DataFrame using secure connection."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    swaps_data = connector.get_swaps(
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        token_in=token_in,
        token_out=token_out,
        pool_address=pool_address,
        trader_address=trader_address,
        max_entries=max_entries
    )
    return pd.DataFrame(swaps_data) if swaps_data else pd.DataFrame()


def quick_get_most_traded_count_tokens(timestamp_start: int,
                                     timestamp_end: int,
                                     side: str = "both",
                                     against: List[str] = None,
                                     limit: int = 100,
                                     include_distributions: bool = False,
                                     exclude_weth_stables: bool = True,
                                     return_text: bool = False,
                                     gateway_url: str = "http://localhost:8081",
                                     api_key: str = None) -> Union[pd.DataFrame, str]:
    """Quick function to get most traded tokens by count as a DataFrame or text."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.get_most_traded_count_tokens(
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        side=side,
        against=against,
        limit=limit,
        include_distributions=include_distributions,
        exclude_weth_stables=exclude_weth_stables,
        return_text=return_text
    )
    if return_text:
        return results if results else ""
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_get_most_traded_value_tokens(timestamp_start: int,
                                     timestamp_end: int,
                                     side: str = "both",
                                     against: str = "weth",
                                     limit: int = 100,
                                     include_distributions: bool = False,
                                     exclude_weth_stables: bool = True,
                                     return_text: bool = False,
                                     gateway_url: str = "http://localhost:8081",
                                     api_key: str = None) -> Union[pd.DataFrame, str]:
    """Quick function to get most traded tokens by volume as a DataFrame or text."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    results = connector.get_most_traded_value_tokens(
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        side=side,
        against=against,
        limit=limit,
        include_distributions=include_distributions,
        exclude_weth_stables=exclude_weth_stables,
        return_text=return_text
    )
    if return_text:
        return results if results else ""
    return pd.DataFrame(results) if results else pd.DataFrame()


def quick_get_token_stats(token_address: str,
                         timestamp_start: int,
                         timestamp_end: int,
                         include_distributions: bool = False,
                         return_text: bool = False,
                         gateway_url: str = "http://localhost:8081",
                         api_key: str = None) -> Union[Optional[Dict[str, Any]], str]:
    """Quick function to get detailed token statistics."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_token_stats(token_address, timestamp_start, timestamp_end, include_distributions, return_text)


def quick_get_trader_stats(trader_address: str,
                          timestamp_start: int,
                          timestamp_end: int,
                          include_distributions: bool = False,
                          gateway_url: str = "http://localhost:8081",
                          api_key: str = None) -> Optional[Dict[str, Any]]:
    """Quick function to get detailed trader statistics."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_trader_stats(trader_address, timestamp_start, timestamp_end, include_distributions)


def quick_get_trading_stats(timestamp_start: int,
                           timestamp_end: int,
                           side: str = "both",
                           against: List[str] = None,
                           gateway_url: str = "http://localhost:8081",
                           api_key: str = None) -> Optional[Dict[str, Any]]:
    """Quick function to get trading statistics summary using secure connection."""
    connector = SecureBrewsConnector(gateway_url=gateway_url, api_key=api_key)
    return connector.get_trading_stats(timestamp_start, timestamp_end, side, against)


# Environment setup helper
def setup_api_key_env(api_key: str):
    """
    Helper function to set the API key in environment variables.
    
    Args:
        api_key: The API key to set
    """
    os.environ['ENGO_API_KEY'] = api_key
    logger.info("✅ API key set in ENGO_API_KEY environment variable")


def get_available_api_keys() -> List[str]:
    """
    Get list of available API keys from the configuration file.
    
    Returns:
        List of available API keys
    """
    return [
        "18a31db7e7add36f0fed7701918a019de73cfdae840e59b4adda6bb7cd0f9471",
        "37bade08b7d52c8822dd1d42805ece06fc748ba1750f8474320c3ea0f0f7cf8a", 
        "5ee7063eb7b13aa31ef134240775502a401de5338f4604789a59b782d8a472bb",
        "25a44b5301e50fb2c114dd32fcb9c268edf4eeb39ad83879bd0de18a5033dbcd",
        "b8c4e2a9d5f7e1a2c8b6d3f4e7a1b2c5d8e6f9a2b5c8d1e4f7a0b3c6d9e2f5a8"
    ]


def print_usage_example():
    """Print usage examples for the secure connector."""
    print("""
🔐 ENGO Secure API Connector Usage Examples:

# 1. Set up API key (choose one from the list)
from shared.secure_api_connector import setup_api_key_env, get_available_api_keys

api_keys = get_available_api_keys()
setup_api_key_env(api_keys[0])  # Use first key

# 2. Create connector (will use API key from environment)
from shared.secure_api_connector import SecureBrewsConnector

connector = SecureBrewsConnector()

# 3. Or pass API key directly
connector = SecureBrewsConnector(api_key="18a31db7e7add36f0fed7701918a019de73cfdae840e59b4adda6bb7cd0f9471")

# 4. Get a brew
brew_data = connector.get_brew("5_min_blocks_2025_01_15_10_30")

# 5. Get brew by timestamp (raises FileNotFoundError if not found)
try:
    brew_data = connector.get_brew_at("2025-01-15T10:33:00Z", "5_min")  # Gets 10:30 brew
    brew_data = connector.get_brew_at("1641398580", "15_min", max_steps_back=3)  # Search back 3 intervals
except FileNotFoundError as e:
    print(f"Brew not found: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")

# 6. Get brew by timestamp with baselines
try:
    brew_with_baselines = connector.get_brew_with_baselines_at(
        "2025-01-15T10:33:00Z", "5_min", 
        max_steps_back=2,
        baseline_types=["4h", "24h", "hour_cadence_30d"]
    )
    print(f"Found brew: {brew_with_baselines['_search_metadata']['found_brew_id']}")
    print(f"Baselines: {brew_with_baselines['_baseline_metadata']['baseline_count']}")
except FileNotFoundError as e:
    print(f"Brew not found: {e}")

# 7. Get brew text by timestamp
try:
    brew_text = connector.get_brew_text_at(
        "2025-01-15T10:33:00Z", "5_min",
        max_steps_back=2,
        baseline_types=["4h", "24h"]
    )
    print(f"Generated text length: {len(brew_text)} characters")
except FileNotFoundError as e:
    print(f"Brew not found: {e}")

# 8. Get enhanced brew analysis by timestamp
try:
    enhanced_brew = connector.get_brew_enhanced_at(
        "2025-01-15T10:33:00Z", "5_min",
        max_steps_back=2,
        baseline_types=["4h", "24h"]
    )
    print(f"Enhanced analysis: {enhanced_brew['type']}")
    print(f"Fields analyzed: {enhanced_brew['_enhancement_metadata']['fields_analyzed']}")
    print(f"Baselines used: {enhanced_brew['baselines']['amount']}")
except FileNotFoundError as e:
    print(f"Brew not found: {e}")

# 9. Get recent brews
recent_brews = quick_get_recent_brews(type_filter="5_min", limit=10)

# 10. Query parquet data
results = connector.query_parquet_range(
    type_filter="5_min",
    request_field="brew_id",
    filter_field="blocks_count", 
    from_val=5,
    to_val=50
)

# 11. Get swaps data
swaps = connector.get_swaps(
    from_timestamp=1640995200,
    to_timestamp=1641081600,
    max_entries=100
)
print(f"Found {len(swaps)} swaps")

# 12. Get swaps for specific token pair
usdc_swaps = connector.get_swaps(
    token_in="0xA0b86a33E6417c4f49Dd22a6dFC54d0d2C94dB6F",
    token_out="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    max_entries=50
)

# 13. Get swaps for specific pool
pool_swaps = connector.get_swaps(
    pool_address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    max_entries=200
)

# 14. Get swaps statistics
swaps_stats = connector.get_swaps_stats()
print(f"Total swap files: {swaps_stats['total_files']}")

# 15. Quick swaps functions
recent_swaps = quick_get_swaps(max_entries=50)
swaps_df = quick_get_swaps_as_dataframe(max_entries=100)

# 16. Check connection status
stats = connector.get_stats()
print(stats)

🌐 Endpoints:
- Gateway (secure): http://localhost:8081
- Fallback (direct): http://localhost:8020
- Dashboard: http://localhost:8080
""")