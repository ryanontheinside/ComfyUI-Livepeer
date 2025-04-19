from ...src.livepeer_job_getter import LivepeerJobGetterBase
from ...src.livepeer_media_processor import LivepeerMediaProcessor
from ...src.livepeer_response_handler import LivepeerResponseHandler
from ...config_manager import config_manager

class LivepeerVideoJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2v", "live2v"]  # Added live2v job type
    PROCESSED_RESULT_KEYS = [
        'processed_video_url', 
        'processed_video_path', 
        'processed_frames', 
        'processed_audio',
        'processed_video_ready'
    ] 
    DEFAULT_OUTPUTS = (None, None, None, None, False) # video_url, video_path, frames, audio, video_ready

    # Define specific input type for video jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("video_job", {})
         },
         "optional": {
            "download_video": ("BOOLEAN", {"default": True})
        },
        "hidden": {
            "unique_id": "UNIQUE_ID"  # Added hidden input
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
                
                # Store the actual output values needed for caching
                processed_data_to_store = {
                         'processed_video_url': video_url_out, 
                         'processed_video_path': video_path_out,
                         'processed_frames': video_frames,
                         'processed_audio': video_audio,
                         'processed_video_ready': video_ready # Store the boolean flag
                     }
                # Return the node-specific outputs tuple and the data to store
                return (video_url_out, video_path_out, video_frames, video_audio, video_ready), processed_data_to_store
            else:
                # If validation failed, return None
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Video) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_video_job_result(self, job_id, unique_id, download_video=True):
        # Pass job_id, unique_id, and download_video (via kwargs) to base method
        return self._get_or_process_job_result(job_id=job_id, unique_id=unique_id, download_video=download_video)


NODE_CLASS_MAPPINGS = {
    "LivepeerVideoJobGetter": LivepeerVideoJobGetter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerVideoJobGetter": "Get Livepeer Video Job",
} 