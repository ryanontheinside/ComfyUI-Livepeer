from ...src.livepeer_job_getter import LivepeerJobGetterBase, BLANK_IMAGE, BLANK_HEIGHT, BLANK_WIDTH
from ...src.livepeer_media_processor import LivepeerMediaProcessor
from ...src.livepeer_response_handler import LivepeerResponseHandler
from ...config_manager import config_manager

class LivepeerImageJobGetter(LivepeerJobGetterBase):
    # Expected job types that produce images
    EXPECTED_JOB_TYPES = ["t2i", "i2i", "upscale", "segment"] 
    PROCESSED_RESULT_KEYS = ['processed_image_output', 'processed_image_ready'] # Key used to store the processed image tensor
    DEFAULT_OUTPUTS = (BLANK_IMAGE, False) # image_output, image_ready

    # Define specific input type for image-related jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("image_job", {})
         },
         "hidden": {
             "unique_id": "UNIQUE_ID", # Ensure unique_id is present
             # Removed the previous custom hidden input
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
                # Store the actual output values needed for caching
                processed_data_to_store = {
                    'processed_image_output': image_out,
                    'processed_image_ready': image_ready
                }
                # Return the node-specific outputs tuple and the data to store
                return (image_out, image_ready), processed_data_to_store
            else:
                # If validation failed, return None
                return None, None
            
        except Exception as e:
            config_manager.handle_error(e, f"Error in _process_raw_result (Image) for job {job_id}", raise_error=False)
            return None, None # Indicate failure

    def get_image_job_result(self, job_id, unique_id):
        # Delegate all logic to the base class handler, passing unique_id
        return self._get_or_process_job_result(job_id=job_id, unique_id=unique_id)

# Mappings for __init__.py
NODE_CLASS_MAPPINGS = {
    "LivepeerImageJobGetter": LivepeerImageJobGetter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerImageJobGetter": "Get Livepeer Image Job",
} 