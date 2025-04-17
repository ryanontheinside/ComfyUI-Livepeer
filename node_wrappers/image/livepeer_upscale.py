from ...src.livepeer_core import LivepeerBase
import torch
from livepeer_ai.models import components
from ...config_manager import config_manager
import traceback
import uuid # Ensure uuid is imported

class LivepeerUpscale(LivepeerBase):
    JOB_TYPE = "upscale" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default upscale model from config
        default_model = config_manager.get_default_model("upscale") or "stabilityai/stable-diffusion-x4-upscaler"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True, "default": "Make this image high resolution"}), # Prompt seems required by BodyGenUpscale
            },
            "optional": {
                 "model_id": ("STRING", {"multiline": False, "default": default_model}),
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

    RETURN_TYPES = ("image_job",) # Use specific type "image_job"
    RETURN_NAMES = ("job_id",)
    FUNCTION = "upscale_image"
    CATEGORY = "Livepeer"

    def upscale_image(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, image, prompt, model_id="", safety_check=True, seed=0, num_inference_steps=75):
        # Skip API call if disabled
        if not enabled:
            return (None,)

        # Prepare the input image using the base class method
        livepeer_image = self.prepare_image(image)
        upscl_img = components.BodyGenUpscaleImage(
            file_name=livepeer_image.file_name,
            content=livepeer_image.content,
        )
        # Prepare arguments for the Livepeer API call
        # Note: SDK uses BodyGenUpscale as the request object
        upscale_params = components.BodyGenUpscale(
            image=upscl_img,
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
            return (job_id,)
        else:
            # Execute synchronously with retry logic
            response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, synchronous_timeout=synchronous_timeout)
            # Generate Job ID and store result directly for sync mode
            job_id = str(uuid.uuid4())
            self._store_sync_result(job_id, self.JOB_TYPE, response)
            return (job_id,)

NODE_CLASS_MAPPINGS = {
    "LivepeerUpscale": LivepeerUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerUpscale": "Livepeer Upscale",
} 