import json
import os
import logging
from logging import handlers
import traceback
import sys

# Default configuration settings
DEFAULT_CONFIG = {
    "api_key": "YOUR_API_KEY_HERE",  # Default API key
    "log_level": "INFO",  # Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "error_handling": {
        "throw_errors": True,  # Whether to throw errors or just log them
        "max_error_logs": 10,   # Maximum number of error logs to keep
        "log_errors_to_file": True  # Whether to log errors to a file
    },
    "output_paths": {
        "images": "output/livepeer/images",
        "videos": "output/livepeer/videos",
        "audio": "output/livepeer/audio",
    },
    "default_retry_settings": {
        "max_retries": 3,
        "retry_delay": 2.0
    },
    "default_timeout": 120.0,  # Default timeout in seconds
    "default_models": {
        "A2T": "openai/whisper-large-v3",
        "I2I": "timbrooks/instruct-pix2pix",
        "I2T": "Salesforce/blip-image-captioning-large",
        "I2V": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "segment": "facebook/sam2-hiera-large",
        "T2I": "SG161222/RealVisXL_V4.0_Lightning",
        "T2S": "parler-tts/parler-tts-large-v1",
        "T2V": "stabilityai/stable-video-diffusion-img2vid-xt",
        "upscale": "stabilityai/stable-diffusion-x4-upscaler"
    }
}

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class LivepeerConfigManager:
    """Manages configuration for ComfyUI-Livepeer nodes."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        # Singleton pattern to ensure only one config manager exists
        if cls._instance is None:
            cls._instance = super(LivepeerConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Avoid re-initialization of the singleton
        if LivepeerConfigManager._initialized:
            return
            
        self.logger = None
        self.config = {}
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        # Initialize configuration and logger
        self._load_config()
        self._setup_logging()
        
        LivepeerConfigManager._initialized = True
    
    def _load_config(self):
        """Load configuration from file or create default if it doesn't exist."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                    # Update with any missing default keys
                    self._update_missing_config_items(DEFAULT_CONFIG, self.config)
            else:
                # Create default config
                self.config = DEFAULT_CONFIG.copy()
                self._save_config()
                print(f"Created default Livepeer config at {self.config_path}")
        except Exception as e:
            print(f"Error loading Livepeer config: {str(e)}")
            self.config = DEFAULT_CONFIG.copy()
    
    def _update_missing_config_items(self, source, target):
        """Recursively update target dict with missing items from source."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                self._update_missing_config_items(value, target[key])
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving Livepeer config: {str(e)}")
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        # Create logger
        self.logger = logging.getLogger("livepeer")
        log_level_str = self.config.get("log_level", "INFO")
        log_level = LOG_LEVELS.get(log_level_str, logging.INFO)
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Add file handler if error logging is enabled
        if self.config["error_handling"]["log_errors_to_file"]:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "livepeer_errors.log")
            file_handler = handlers.RotatingFileHandler(
                log_file, 
                maxBytes=1024*1024,  # 1MB
                backupCount=self.config["error_handling"]["max_error_logs"]
            )
            file_handler.setLevel(logging.ERROR)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def get_api_key(self):
        """Get the configured API key."""
        return self.config.get("api_key", DEFAULT_CONFIG["api_key"])
    
    def set_api_key(self, api_key):
        """Set a new API key and save configuration."""
        self.config["api_key"] = api_key
        self._save_config()
    
    def get_retry_settings(self):
        """Get the configured retry settings."""
        return (
            self.config.get("default_retry_settings", {}).get("max_retries", DEFAULT_CONFIG["default_retry_settings"]["max_retries"]),
            self.config.get("default_retry_settings", {}).get("retry_delay", DEFAULT_CONFIG["default_retry_settings"]["retry_delay"])
        )
    
    def get_timeout(self):
        """Get the configured timeout setting."""
        return self.config.get("default_timeout", DEFAULT_CONFIG["default_timeout"])
    
    def get_default_model(self, job_type):
        """Get the default model for a specific job type.
        
        Args:
            job_type (str): The job type (A2T, T2I, I2V, etc.)
            
        Returns:
            str: The default model ID for the specified job type
        """
        default_models = self.config.get("default_models", DEFAULT_CONFIG["default_models"])
        return default_models.get(job_type, None)
    
    def get_output_path(self, output_type="videos"):
        """Get configured output path for given type (images, videos, audio)."""
        output_paths = self.config.get("output_paths", DEFAULT_CONFIG["output_paths"])
        base_path = output_paths.get(output_type, f"output/livepeer/{output_type}")
        
        # Convert to absolute path based on Comfy root directory
        comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        abs_path = os.path.join(comfy_root, base_path)
        
        # Ensure directory exists
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    def should_throw_errors(self):
        """Check if errors should be thrown or just logged."""
        return self.config.get("error_handling", {}).get("throw_errors", DEFAULT_CONFIG["error_handling"]["throw_errors"])
    
    def handle_error(self, error, message=None, raise_error=None):
        """Handle an error according to configuration."""
        error_str = str(error)
        error_trace = traceback.format_exc()
        
        if message:
            self.logger.error(f"{message}: {error_str}")
        else:
            self.logger.error(f"Error: {error_str}")
        
        self.logger.debug(error_trace)
        
        # Determine whether to raise the error
        should_raise = raise_error if raise_error is not None else self.should_throw_errors()
        
        if should_raise:
            raise error
        
        return error_str
    
    def log(self, level, message):
        """Log a message at the specified level."""
        if level.upper() in LOG_LEVELS:
            log_method = getattr(self.logger, level.lower())
            log_method(message)
        else:
            self.logger.info(message)

# Global instance
config_manager = LivepeerConfigManager() 