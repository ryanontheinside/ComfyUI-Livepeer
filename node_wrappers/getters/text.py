from ...src.livepeer_job_getter import LivepeerJobGetterBase
from ...src.livepeer_response_handler import LivepeerResponseHandler
from ...config_manager import config_manager

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

NODE_CLASS_MAPPINGS = {
    "LivepeerTextJobGetter": LivepeerTextJobGetter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerTextJobGetter": "Get Livepeer Text Job",
} 