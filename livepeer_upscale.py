from .livepeer_core import LivepeerBase
import torch
from livepeer_ai.models import components
import traceback

class LivepeerUpscale(LivepeerBase):
    JOB_TYPE = "upscale" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True, "default": "Make this image high resolution"}), # Prompt seems required by BodyGenUpscale
            },
            "optional": {
                 "model_id": ("STRING", {"multiline": False, "default": "stabilityai/stable-diffusion-x4-upscaler"}),
                 "safety_check": ("BOOLEAN", {"default": True}),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use 0 for None/random
                 "num_inference_steps": ("INT", {"default": 75, "min": 1, "max": 150, "step": 1}),
             }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("IMAGE", "STRING") # IMAGE for sync, STRING (job_id) for async
    RETURN_NAMES = ("upscaled_image", "job_id")
    FUNCTION = "upscale_image"
    CATEGORY = "Livepeer"

    def upscale_image(self, api_key, max_retries, retry_delay, run_async, image, prompt, model_id="", safety_check=True, seed=0, num_inference_steps=75):

        # Prepare the input image using the base class method
        livepeer_image = self.prepare_image(image)

        # Prepare arguments for the Livepeer API call
        # Note: SDK uses BodyGenUpscale as the request object
        upscale_params = components.BodyGenUpscale(
            image=livepeer_image,
            prompt=prompt, # Prompt is required in BodyGenUpscale
            model_id=model_id if model_id else None,
            safety_check=safety_check,
            seed=seed if seed != 0 else None,
            num_inference_steps=num_inference_steps
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            # Adjust SDK call as needed - explicitly use livepeer.generate
            return livepeer.generate.upscale(request=upscale_params)

        if run_async:
            # Trigger async job and return job ID
            job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
            # Return None for image, job_id for the string output
            return (None, job_id)
        else:
            # Execute synchronously with retry logic
            try:
                response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func)

                # Process the response to get the image tensor
                # Assume response structure is similar to t2i/i2i
                if hasattr(response, 'image_response') and response.image_response:
                     upscaled_image_tensor = self.process_image_response(response)
                     return (upscaled_image_tensor, None)
                else:
                    print(f"Error: Livepeer Upscale response missing image_response.")
                    # Attempt to extract error from response if available
                    error_msg = getattr(response, 'error', 'Unknown error')
                    print(f"Livepeer Upscale Error: {error_msg}")
                    # Return None for both outputs on error
                    # Consider adding an error output string later if needed
                    return (None, None)

            except Exception as e:
                 print(f"Error during Livepeer Upscale sync execution: {e}")
                 traceback.print_exc()
                 return (None, None)

NODE_CLASS_MAPPINGS = {
    "LivepeerUpscale": LivepeerUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerUpscale": "Livepeer Upscale",
} 