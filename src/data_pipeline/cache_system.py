import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import requests
from requests.exceptions import RequestException

class FileCache:
    """
    A simple file-based caching system that checks for existing cached results
    before running expensive operations.
    """
    def __init__(self, cache_dir: str = os.path.join(os.path.dirname(__file__), "../data/cache"), cache_duration: Optional[timedelta] = None):
        """
        Initialize the cache handler.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Optional timedelta for cache expiration
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, cache_key: str) -> str:
        """Generate a filepath for a given cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pickle")
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Try to retrieve a cached result.
        
        Returns:
            The cached data if valid, None otherwise
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
            
        # Check if cache has expired
        if self.cache_duration:
            modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - modified_time > self.cache_duration:
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, IOError):
            return None
    
    def save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save results to cache file."""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def process_with_cache(self, cache_key: str, process_func, *args, **kwargs):
        """
        Main method to handle processing with caching.
        
        Args:
            cache_key: Unique identifier for the cached result
            process_func: The function to run if cache miss
            *args, **kwargs: Arguments to pass to process_func
        """
        # Try to get cached result
        result = self.get_cached_result(cache_key)
        
        if result is not None:
            print(f"Cache hit for key: {cache_key}")
            return result
        
        # Cache miss - run the process
        print(f"Cache miss for key: {cache_key}. Running process...")
        result = process_func(*args, **kwargs)
        
        # Save to cache
        self.save_to_cache(cache_key, result)
        return result
    


class APICache(FileCache):
    """
    Extension of FileCache specifically for handling API requests with quota limitations.
    """
    def __init__(self, 
                 cache_dir: str = "api_cache",
                 cache_duration: Optional[timedelta] = None,
                 api_key: Optional[str] = None):
        super().__init__(cache_dir, cache_duration)
        self.api_key = api_key

    def fetch_data_with_cache(self, 
                            endpoint: str,
                            params: Dict[str, Any] = None,
                            force_cache: bool = False) -> Dict[str, Any]:
        """
        Fetch data from API with caching and fallback mechanism.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters for the API
            force_cache: If True, skip API call and use cache directly
        
        Returns:
            API response data or cached data
        
        Raises:
            Exception: If no cached data exists and API call fails
        """
        # Create a cache key based on endpoint and params
        cache_key = f"{endpoint}_{str(params)}"
        
        if not force_cache:
            try:
                # Try API first
                headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
                response = requests.get(endpoint, params=params, headers=headers)
                
                # Check for API quota errors (adjust status codes as needed)
                if response.status_code == 429:  # Too Many Requests
                    print("API quota exceeded, falling back to cache...")
                    return self._fallback_to_cache(cache_key)
                
                response.raise_for_status()
                data = response.json()
                
                # Cache the successful response
                self.save_to_cache(cache_key, data)
                print("Successfully fetched and cached new data from API")
                return data
                
            except RequestException as e:
                print(f"API request failed: {str(e)}")
                return self._fallback_to_cache(cache_key)
        
        return self._fallback_to_cache(cache_key)
    
    def _fallback_to_cache(self, cache_key: str) -> Dict[str, Any]:
        """Helper method to handle cache fallback logic."""
        cached_data = self.get_cached_result(cache_key)
        if cached_data is not None:
            print("Successfully retrieved data from cache")
            return cached_data
        raise Exception("No cached data available and API request failed")