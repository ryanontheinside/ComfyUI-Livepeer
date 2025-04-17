import os
import numpy as np
import soundfile as sf
import torch

from ...src.livepeer_job_getter import LivepeerJobGetterBase, BLANK_IMAGE, BLANK_HEIGHT, BLANK_WIDTH
from ...src.livepeer_media_processor import LivepeerMediaProcessor
from ...src.livepeer_response_handler import LivepeerResponseHandler
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
        try:
            # Use LivepeerResponseHandler to check and validate response type
            has_image_data, response_obj = LivepeerResponseHandler.extract_image_data(job_id, job_type, raw_result)
            
            if has_image_data and response_obj is not None:
                # Process the validated response
                image_out = LivepeerMediaProcessor.process_image_response(response_obj)
                image_ready = image_out is not None and image_out.shape[1] > BLANK_HEIGHT
                return (image_out, image_ready), {'processed_image': image_out}
            else:
                # If validation failed, return None
                return None, None
            
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Image) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_image_job_result(self, job_id):
        # Delegate all logic to the base class handler
        return self._get_or_process_job_result(job_id)


class LivepeerVideoJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2v", "live2video"]  # Added live2video job type
    PROCESSED_RESULT_KEYS = ['processed_url', 'processed_path', 'processed_frames', 'processed_audio'] # Keys used to store results
    DEFAULT_OUTPUTS = (None, None, None, None, False) # video_url, video_path, frames, audio, video_ready

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
        
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "AUDIO", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # video_url, video_path, frames, audio, video_ready
    RETURN_NAMES = ("video_url", "video_path", "frames", "audio", "video_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_video_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw video response, downloads if needed, returns processed data."""
        try:
            # Use LivepeerResponseHandler to check and validate response type
            has_video_data, response_obj, video_urls = LivepeerResponseHandler.extract_video_data(job_id, job_type, raw_result)
            
            if has_video_data and response_obj is not None:
                video_url_out = video_urls[0] if video_urls else None
                video_ready = False  # Only set to True after successful processing
                video_path_out = None
                video_frames = None
                video_audio = None
                
                # Check kwargs for download flag, default to True if not provided
                download_video = kwargs.get('download_video', True)
                
                # Download the video if needed
                if video_url_out and download_video:
                    config_manager.log("info", f"Livepeer Job Getter: Downloading video for job {job_id} from {video_url_out}")
                    try:
                        video_path_out = LivepeerMediaProcessor.download_media(video_url_out, "videos")
                        
                        # Load the video into tensor format
                        video_info = LivepeerMediaProcessor.load_video_to_tensor(
                            video_path=video_path_out, 
                            extract_audio=True
                        )
                        
                        if video_info is None:
                            raise ValueError(f"Failed to load video from {video_path_out}")
                            
                        video_frames = video_info['frames']
                        video_audio = video_info['audio']
                        video_ready = True
                        config_manager.log("info", f"Successfully loaded video from {video_path_out}")
                            
                    except Exception as e:
                        error_msg = config_manager.handle_error(e, f"Error processing video for job {job_id}", raise_error=False)
                        config_manager.log("error", f"Failed to process video: {error_msg}")
                        # Don't continue with processing if video loading failed
                        return None, None
                
                processed_data_to_store = {
                         'processed_url': video_url_out, 
                         'processed_path': video_path_out,
                         'processed_frames': video_frames,
                         'processed_audio': video_audio
                     }
                return (video_url_out, video_path_out, video_frames, video_audio, video_ready), processed_data_to_store
            else:
                # If validation failed, return None
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Video) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_video_job_result(self, job_id, download_video=True):
        # Delegate to base class, passing extra arguments via kwargs
        return self._get_or_process_job_result(job_id, download_video=download_video)

class LivepeerTextJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2t", "llm", "a2t"]  # Added a2t job type for audio-to-text
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
        try:
            # Use LivepeerResponseHandler to check and validate text response
            has_text_data, text_out = LivepeerResponseHandler.extract_text_data(job_id, job_type, raw_result)
            
            if has_text_data and text_out:
                text_ready = True
                return (text_out, text_ready), {'processed_text': text_out}
            else:
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Text) for job {job_id}", raise_error=False)
            return None, None

    def get_text_job_result(self, job_id):
        # Delegate all logic to the base class handler
        return self._get_or_process_job_result(job_id)

# New AudioJobGetter for text-to-speech operations
class LivepeerAudioJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["t2s"]  # Audio from text-to-speech
    PROCESSED_RESULT_KEYS = ['processed_url', 'processed_path', 'processed_audio'] # Keys used to store results
    DEFAULT_OUTPUTS = (None, False) # audio_output, audio_ready

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
                
                processed_data_to_store = {
                        'processed_url': audio_url, 
                        'processed_path': audio_path_out,
                        'processed_audio': audio_output
                    }
                return (audio_output, audio_ready), processed_data_to_store
            else:
                # If validation failed, return None
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Audio) for job {job_id}", raise_error=False)
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