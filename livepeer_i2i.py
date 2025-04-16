from .livepeer_core import LivepeerBase
import torch
from livepeer_ai.models import components

class LivepeerI2I(LivepeerBase):
    JOB_TYPE = "i2i" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True, "default": "Make this racoon wear a pirate costume"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"multiline": False, "default": "timbrooks/instruct-pix2pix"}),
                "loras": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "safety_check": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use 0 for None/random
                "num_inference_steps": ("INT", {"default": 100, "min": 1, "max": 150, "step": 1}), # Default increased based on SDK
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("IMAGE", "image_job") # Use specific type "image_job"
    RETURN_NAMES = ("image", "job_id")
    FUNCTION = "image_to_image"
    CATEGORY = "Livepeer"

    def image_to_image(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, image, prompt, negative_prompt="", model_id="", loras="", strength=0.8, guidance_scale=7.5, image_guidance_scale=1.5, safety_check=True, seed=0, num_inference_steps=100, num_images_per_prompt=1):
        # Skip API call if disabled
        if not enabled:
            return (image, None)

        # Prepare the input image using the base class method
        livepeer_image = self.prepare_image(image)

        # Prepare arguments for the Livepeer API call using all parameters
        # Note: The SDK uses BodyGenImageToImage, which is effectively the params object here.
        i2i_args = components.BodyGenImageToImage( # Changed from ImageToImageParams based on SDK file
            image=livepeer_image,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            model_id=model_id if model_id else None,
            loras=loras if loras else None,
            strength=strength,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            safety_check=safety_check,
            seed=seed if seed != 0 else None, # Map 0 back to None
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            # Explicitly use livepeer.generate based on SDK structure
            return livepeer.generate.image_to_image(request=i2i_args)

        if run_async:
            # Trigger async job and return job ID
            job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
            # Return None for image, job_id for the string output
            return (None, job_id)
        else:
            # Execute synchronously with retry logic
            response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, synchronous_timeout=synchronous_timeout)

            # Process the response to get the image tensor
            image_tensor = self.process_image_response(response)
            # Return image tensor, None for job_id
            return (image_tensor, None)

NODE_CLASS_MAPPINGS = {
    "LivepeerI2I": LivepeerI2I,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerI2I": "Livepeer I2I",
} 