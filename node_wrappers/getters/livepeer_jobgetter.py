import time
import torch
import traceback
import os
from ...src.livepeer_core import LivepeerBase
from ...src.livepeer_job_base import LivepeerJobGetterBase, BLANK_IMAGE, BLANK_HEIGHT, BLANK_WIDTH
from ...config_manager import config_manager

class LivepeerImageJobGetter(LivepeerJobGetterBase):
    # Expected job types that produce images
    EXPECTED_JOB_TYPES = ["t2i", "i2i", "upscale", "segment"] 
    PROCESSED_RESULT_KEYS = ['processed_image'] # Key used to store the processed image tensor
    DEFAULT_OUTPUTS = (BLANK_IMAGE, False) # image_output, image_ready

    # Define specific input type for image-related jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("image_job", {})
         }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("IMAGE", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES
    RETURN_NAMES = ("image_output", "image_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_image_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw image response into tensor and returns processed data."""
        if raw_result and hasattr(raw_result, 'image_response') and raw_result.image_response:
            try:
                base_processor = LivepeerBase() # Instantiate to access processing methods
                image_out = base_processor.process_image_response(raw_result)
                image_ready = image_out is not None and image_out.shape[1] > BLANK_HEIGHT
                return (image_out, image_ready), {'processed_image': image_out} # Return processed tuple and data to store
            except Exception as e:
                config_manager.handle_error(e, f"Error in _process_raw_result (Image) for job {job_id}", raise_error=False)
                return None, None # Indicate failure
        else:
            config_manager.log("error", f"Job {job_id} ({job_type}) has no valid image_response in its raw result.")
            return None, None # Indicate failure

    def get_image_job_result(self, job_id):
        # Delegate all logic to the base class handler
        return self._get_or_process_job_result(job_id)


class LivepeerVideoJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2v", "live2video"]  # Added live2video job type
    PROCESSED_RESULT_KEYS = ['processed_url', 'processed_path'] # Keys used to store results
    DEFAULT_OUTPUTS = (None, None, False) # video_url, video_path, video_ready

    # Define specific input type for video jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("video_job", {})
         },
         "optional": {
            "download_video": ("BOOLEAN", {"default": True})
        }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # video_url, video_path, video_ready
    RETURN_NAMES = ("video_url", "video_path", "video_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_video_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw video response, downloads if needed, returns processed data."""
        if raw_result and hasattr(raw_result, 'video_response') and raw_result.video_response:
            try:
                base_processor = LivepeerBase()
                processed_urls = base_processor.process_video_response(raw_result)
                video_url_out = processed_urls[0] if processed_urls else None
                video_ready = bool(video_url_out)
                video_path_out = None
                
                # Check kwargs for download flag, default to True if not provided
                download_video = kwargs.get('download_video', True)
                
                if video_ready and download_video:
                    config_manager.log("info", f"Livepeer Job Getter: Downloading video for job {job_id} from {video_url_out}")
                    try:
                        video_path_out = base_processor.download_video(video_url_out)
                    except Exception as e:
                        error_msg = config_manager.handle_error(e, f"Error downloading video for job {job_id}", raise_error=False)
                        video_path_out = f"Error: {error_msg}" # Store error in path field
                
                processed_data_to_store = {
                         'processed_url': video_url_out, 
                         'processed_path': video_path_out
                     }
                return (video_url_out, video_path_out, video_ready), processed_data_to_store
            except Exception as e:
                config_manager.handle_error(e, f"Error in _process_raw_result (Video) for job {job_id}", raise_error=False)
                return None, None # Indicate failure
        else:
            config_manager.log("error", f"Job {job_id} ({job_type}) has no valid video_response in its raw result.")
            return None, None # Indicate failure

    def get_video_job_result(self, job_id, download_video=True):
        # Delegate to base class, passing extra arguments via kwargs
        return self._get_or_process_job_result(job_id, download_video=download_video)

class LivepeerTextJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2t", "llm"]  # Added llm job type
    PROCESSED_RESULT_KEYS = ['processed_text'] # Key used to store processed text
    DEFAULT_OUTPUTS = ("", False) # text_output, text_ready

    # Define specific input type for text jobs
    INPUT_TYPES_DICT = {
        "required": {
            "job_id": ("text_job", {})
         }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("STRING", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # text_output, text_ready
    RETURN_NAMES = ("text_output", "text_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_text_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw text response and returns processed data."""
        text_out = ""
        try:
            # Adapt text extraction logic
            if hasattr(raw_result, 'text_response') and hasattr(raw_result.text_response, 'text') and raw_result.text_response.text is not None: 
                text_out = str(raw_result.text_response.text)
            elif hasattr(raw_result, 'text'): # Direct attribute on result object
                 text_out = str(raw_result.text)
            elif isinstance(raw_result, str): # Raw string result
                text_out = raw_result
            # Handle LLM response format
            elif hasattr(raw_result, 'choices') and raw_result.choices:
                if hasattr(raw_result.choices[0], 'message') and hasattr(raw_result.choices[0].message, 'content'):
                    text_out = str(raw_result.choices[0].message.content)
            
            if text_out: # Check if we got some text
                 text_ready = True
                 return (text_out, text_ready), {'processed_text': text_out}
            else:
                 config_manager.log("error", f"Job {job_id} ({job_type}) did not contain expected text output in result: {raw_result}")
                 return None, None # Indicate failure
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Text) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_text_job_result(self, job_id):
        # Delegate all logic to the base class handler
        return self._get_or_process_job_result(job_id)

# New AudioJobGetter for text-to-speech operations
class LivepeerAudioJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["t2s"]  # Audio from text-to-speech
    PROCESSED_RESULT_KEYS = ['processed_url', 'processed_path'] # Keys used to store results
    DEFAULT_OUTPUTS = (None, None, False) # audio_url, audio_path, audio_ready

    # Define specific input type for audio jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("audio_job", {})
         },
         "optional": {
            "download_audio": ("BOOLEAN", {"default": True})
        }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # audio_url, audio_path, audio_ready
    RETURN_NAMES = ("audio_url", "audio_path", "audio_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_audio_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw audio response, downloads if needed, returns processed data."""
        if raw_result and hasattr(raw_result, 'audio_response'):
            try:
                base_processor = LivepeerBase()
                # Extract audio URL from response
                audio_url_out = raw_result.audio_response.url if hasattr(raw_result.audio_response, 'url') else None
                audio_ready = bool(audio_url_out)
                audio_path_out = None
                
                # Check kwargs for download flag, default to True if not provided
                download_audio = kwargs.get('download_audio', True)
                
                if audio_ready and download_audio:
                    config_manager.log("info", f"Livepeer Job Getter: Downloading audio for job {job_id} from {audio_url_out}")
                    try:
                        # Use the download method to get audio, but specify audio output path
                        audio_path_out = base_processor.download_video(audio_url_out, config_manager.get_output_path("audio"))
                        # Rename extension based on format if needed (assuming mp3 default)
                        if hasattr(raw_result, 'format') and raw_result.format:
                            base_ext = os.path.splitext(audio_path_out)[0]
                            new_path = f"{base_ext}.{raw_result.format}"
                            os.rename(audio_path_out, new_path)
                            audio_path_out = new_path
                    except Exception as e:
                        error_msg = config_manager.handle_error(e, f"Error downloading audio for job {job_id}", raise_error=False)
                        audio_path_out = f"Error: {error_msg}" # Store error in path field
                
                processed_data_to_store = {
                         'processed_url': audio_url_out, 
                         'processed_path': audio_path_out
                     }
                return (audio_url_out, audio_path_out, audio_ready), processed_data_to_store
            except Exception as e:
                config_manager.handle_error(e, f"Error in _process_raw_result (Audio) for job {job_id}", raise_error=False)
                return None, None # Indicate failure
        else:
            config_manager.log("error", f"Job {job_id} ({job_type}) has no valid audio_response in its raw result.")
            return None, None # Indicate failure

    def get_audio_job_result(self, job_id, download_audio=True):
        # Delegate to base class, passing extra arguments via kwargs
        return self._get_or_process_job_result(job_id, download_audio=download_audio)


# Mappings for __init__.py
NODE_CLASS_MAPPINGS = {
    "LivepeerImageJobGetter": LivepeerImageJobGetter,
    "LivepeerVideoJobGetter": LivepeerVideoJobGetter,
    "LivepeerTextJobGetter": LivepeerTextJobGetter,
    "LivepeerAudioJobGetter": LivepeerAudioJobGetter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerImageJobGetter": "Get Livepeer Image Job",
    "LivepeerVideoJobGetter": "Get Livepeer Video Job",
    "LivepeerTextJobGetter": "Get Livepeer Text Job",
    "LivepeerAudioJobGetter": "Get Livepeer Audio Job",
} 