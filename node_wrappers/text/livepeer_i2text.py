import traceback
from ...src.livepeer_base import LivepeerBase
from livepeer_ai.models import components
from ...config_manager import config_manager
import uuid # Ensure uuid is imported

class LivepeerI2T(LivepeerBase):
    JOB_TYPE = "i2t" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default I2T model from config
        default_model = config_manager.get_default_model("I2T") or "Salesforce/blip-image-captioning-large"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                # Add optional parameters from BodyGenImageToText
                "prompt": ("STRING", {"multiline": True, "default": ""}), # Optional prompt to guide captioning
                "model_id": ("STRING", {"multiline": False, "default": default_model}) # e.g., "Salesforce/blip-image-captioning-large"
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    # Output text and job_id
    RETURN_TYPES = ("text_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "image_to_text"
    CATEGORY = "Livepeer"

    def image_to_text(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, image, prompt="", model_id=""):
        # Skip API call if disabled
        if not enabled:
            return (None,)

        # Prepare the input image using the base class method
        # Get the image data from prepare_image
        livepeer_image = self.prepare_image(image)
        
        # Create proper BodyGenImageToTextImage instance instead of using Image directly
        i2t_image = components.BodyGenImageToTextImage(
            file_name=livepeer_image.file_name,
            content=livepeer_image.content
        )

        # Prepare arguments for the Livepeer API call using the correct image type
        i2t_params = components.BodyGenImageToText(
            image=i2t_image,
            prompt=prompt if prompt else None,
            model_id=model_id if model_id else None
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            # Explicitly use livepeer.generate based on SDK structure
            return livepeer.generate.image_to_text(request=i2t_params)

        if run_async:
            # Trigger async job and return job ID
            job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
            return (job_id,)
        else:
            # Execute synchronously with retry logic
            try:
                response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, synchronous_timeout=synchronous_timeout)

                # Generate Job ID and store result directly for sync mode
                job_id = str(uuid.uuid4())
                self._store_sync_result(job_id, self.JOB_TYPE, response)
                return (job_id,)
            except Exception as e:
                 print(f"Error during Livepeer I2T sync execution: {e}")
                 traceback.print_exc()
                 return (f"Error: {str(e)}",) # Return error message in text field


NODE_CLASS_MAPPINGS = {
    "LivepeerI2T": LivepeerI2T,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerI2T": "Livepeer I2T",
} 