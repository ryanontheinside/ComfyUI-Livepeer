import traceback
from .livepeer_core import LivepeerBase
from livepeer_ai.models import components

class LivepeerI2T(LivepeerBase):
    JOB_TYPE = "i2t" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                # Add optional parameters from BodyGenImageToText
                "prompt": ("STRING", {"multiline": True, "default": ""}), # Optional prompt to guide captioning
                "model_id": ("STRING", {"multiline": False, "default": "Salesforce/blip-image-captioning-large"}) # e.g., "Salesforce/blip-image-captioning-large"
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    # Output text and job_id
    RETURN_TYPES = ("STRING", "text_job")
    RETURN_NAMES = ("text", "job_id")
    FUNCTION = "image_to_text"
    CATEGORY = "Livepeer"

    def image_to_text(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, image, prompt="", model_id=""):
        # Skip API call if disabled
        if not enabled:
            return ("", None)

        # Prepare the input image using the base class method
        livepeer_image = self.prepare_image(image)

        # Prepare arguments for the Livepeer API call
        # Note: SDK uses BodyGenImageToText as the request object
        i2t_params = components.BodyGenImageToText(
            image=livepeer_image,
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
            # Return None for text, job_id for the string output
            return (None, job_id)
        else:
            # Execute synchronously with retry logic
            try:
                response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, synchronous_timeout=synchronous_timeout)

                # Process the response to get the text string
                text_result = ""
                # Check common response attributes for text output
                if hasattr(response, 'text'):
                    text_result = str(response.text)
                elif hasattr(response, 'text_response'): # Less common, but check
                    text_result = str(response.text_response)
                elif isinstance(response, str): # If response itself is the text
                     text_result = response
                # Check if response is the direct SDK output object with a text attr
                elif hasattr(response, 'image_to_text_response') and hasattr(response.image_to_text_response, 'text'):
                    text_result = str(response.image_to_text_response.text)
                else:
                     print(f"Warning: Livepeer I2T response format unexpected: {response}")
                     text_result = f"Error: Unexpected response format {type(response)}"

                # Return text string, None for job_id
                return (text_result, None)
            except Exception as e:
                 print(f"Error during Livepeer I2T sync execution: {e}")
                 traceback.print_exc()
                 return (f"Error: {str(e)}", None) # Return error message in text field


NODE_CLASS_MAPPINGS = {
    "LivepeerI2T": LivepeerI2T,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerI2T": "Livepeer I2T",
} 