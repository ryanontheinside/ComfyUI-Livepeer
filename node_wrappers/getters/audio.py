import os
from ...src.livepeer_job_getter import LivepeerJobGetterBase
from ...src.livepeer_media_processor import LivepeerMediaProcessor
from ...src.livepeer_response_handler import LivepeerResponseHandler
from ...config_manager import config_manager

class LivepeerAudioJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["t2s"]  # Audio from text-to-speech
    PROCESSED_RESULT_KEYS = ['processed_audio_output', 'processed_audio_ready'] 
    DEFAULT_OUTPUTS = (None, False) # audio_output, audio_ready

    # Define specific input type for audio jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("audio_job", {})
         },
         "optional": {
            "download_audio": ("BOOLEAN", {"default": True})
        },
        "hidden": {
            "unique_id": "UNIQUE_ID"
        }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("AUDIO", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # audio_output, audio_ready
    RETURN_NAMES = ("audio_output", "audio_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_audio_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw audio response, downloads if needed, returns processed data."""
        try:
            # Use LivepeerResponseHandler to check and validate audio response
            has_audio_data, response_obj, audio_url = LivepeerResponseHandler.extract_audio_data(job_id, job_type, raw_result)
            
            if has_audio_data and response_obj is not None and audio_url:
                audio_ready = False
                audio_output = None
                audio_path_out = None
                
                # Check kwargs for download flag, default to True if not provided
                download_audio = kwargs.get('download_audio', True)
                
                # Download the audio if needed
                if download_audio:
                    config_manager.log("info", f"Livepeer Job Getter: Downloading audio for job {job_id} from {audio_url}")
                    try:
                        # Download the audio file
                        audio_path_out = LivepeerMediaProcessor.download_media(audio_url, "audio")
                        
                        # Handle audio format if needed
                        if hasattr(response_obj, 'format') and response_obj.format:
                            base_ext = os.path.splitext(audio_path_out)[0]
                            new_path = f"{base_ext}.{response_obj.format}"
                            os.rename(audio_path_out, new_path)
                            audio_path_out = new_path
                        
                        # Load audio as waveform for ComfyUI
                        if os.path.exists(audio_path_out):
                            try:
                                # Use the centralized audio loading method
                                audio_output = LivepeerMediaProcessor.load_audio_to_tensor(audio_path_out)
                                if audio_output is not None:
                                    audio_ready = True
                                else:
                                    config_manager.log("error", f"Failed to load audio from {audio_path_out}")
                                    return None, None
                            
                            except Exception as e:
                                error_msg = config_manager.handle_error(e, f"Error loading audio file {audio_path_out}", raise_error=False)
                                config_manager.log("error", f"Failed to load audio: {error_msg}")
                                return None, None
                    except Exception as e:
                        error_msg = config_manager.handle_error(e, f"Error downloading audio for job {job_id}", raise_error=False)
                        config_manager.log("error", f"Failed to download audio: {error_msg}")
                        return None, None
                else: # If download_audio is False
                     audio_output = None # No tensor generated
                     audio_ready = False # Not ready if not downloaded/loaded
                
                # Store the actual output values needed for caching
                processed_data_to_store = {
                        'processed_audio_output': audio_output, 
                        'processed_audio_ready': audio_ready
                        # We don't necessarily need to store url/path if not directly outputting them
                    }
                # Return the node-specific outputs tuple and the data to store
                return (audio_output, audio_ready), processed_data_to_store
            else:
                # If validation failed, return None
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Audio) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_audio_job_result(self, job_id, unique_id):
        return self._get_or_process_job_result(job_id=job_id, unique_id=unique_id)


# Mappings for __init__.py
NODE_CLASS_MAPPINGS = {
    "LivepeerAudioJobGetter": LivepeerAudioJobGetter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerAudioJobGetter": "Get Livepeer Audio Job",
} 