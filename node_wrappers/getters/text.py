from ...src.livepeer_job_getter import LivepeerJobGetterBase
from ...src.livepeer_response_handler import LivepeerResponseHandler
from ...config_manager import config_manager

class LivepeerTextJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["a2t", "i2t", "llm"]
    PROCESSED_RESULT_KEYS = ['processed_text_output']
    DEFAULT_OUTPUTS = ("",) # text_output

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("text_job", {"lazy": True})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID" # Added hidden input
            }
        }

    RETURN_TYPES = ("STRING",) + LivepeerJobGetterBase.RETURN_TYPES
    RETURN_NAMES = ("text_output",) + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_text_job_result"

    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes raw text response and returns processed data."""
        try:
            # Use LivepeerResponseHandler to check and validate text response
            has_text_data, text_out = LivepeerResponseHandler.extract_text_data(job_id, job_type, raw_result)
            
            if has_text_data and text_out:
                text_ready = True
                # Store the actual output values needed for caching
                processed_data_to_store = {
                    'processed_text_output': text_out,
                    'processed_text_ready': text_ready
                }
                # Return the node-specific outputs tuple and the data to store
                return (text_out, text_ready), processed_data_to_store
            else:
                return None, None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Text) for job {job_id}", raise_error=False)
            return None, None

    def get_text_job_result(self, job_id):
        return self._get_or_process_job_result(job_id=job_id)

NODE_CLASS_MAPPINGS = {
    "LivepeerTextJobGetter": LivepeerTextJobGetter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerTextJobGetter": "Get Livepeer Text Job",
} 