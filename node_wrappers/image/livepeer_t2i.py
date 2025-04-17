import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_base import LivepeerBase
from ...config_manager import config_manager
import uuid

class LivepeerT2I(LivepeerBase):
    JOB_TYPE = "t2i"

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default T2I model from config
        default_model = config_manager.get_default_model("T2I") or "ByteDance/SDXL-Lightning"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of a baby raccoon wearing an intricate steampunk outfit"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"multiline": False, "default": default_model}),
                "loras": ("STRING", {"multiline": True, "default": ""}), # e.g., "{ \"latent-consistency/lcm-lora-sdxl\": 1.0 }"
                "height": ("INT", {"default": 576, "min": 64, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "safety_check": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use 0 for None/random in ComfyUI
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 150, "step": 1}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        # Make sure 'enabled' is listed first if modifying common_inputs directly or ensure it's passed correctly
        return node_inputs

    RETURN_TYPES = ("image_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "text_to_image"
    CATEGORY = "Livepeer"

    def text_to_image(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, prompt, negative_prompt="", model_id="", loras="", height=576, width=1024, guidance_scale=7.5, safety_check=True, seed=0, num_inference_steps=50, num_images_per_prompt=1):
        # Skip API call if disabled
        if not enabled:
            return (None,)
            
        try:
            t2i_args = components.TextToImageParams(
                 prompt=prompt,
                 negative_prompt=negative_prompt if negative_prompt else None,
                 model_id=model_id if model_id else None,
                 loras=loras if loras else None,
                 height=height,
                 width=width,
                 guidance_scale=guidance_scale,
                 safety_check=safety_check,
                 seed=seed if seed != 0 else None,
                 num_inference_steps=num_inference_steps,
                 num_images_per_prompt=num_images_per_prompt
            )

            # Log the generation request
            config_manager.log("info", f"Generating image with prompt: '{prompt[:50]}...' using model: {model_id or 'default'}")

            # Define the operation function for retry/async logic
            def operation_func(livepeer):
                return livepeer.generate.text_to_image(request=t2i_args)

            if run_async:
                job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
                return (job_id,)
            else:
                # Execute synchronously
                response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, synchronous_timeout=synchronous_timeout)
                # Generate Job ID and store result directly for sync mode
                job_id = str(uuid.uuid4())
                self._store_sync_result(job_id, self.JOB_TYPE, response)
                return (job_id,)
        except Exception as e:
            # Use the configuration manager to handle errors
            config_manager.handle_error(e, "Error in text-to-image generation")
            # Return None even on error to avoid breaking the workflow
            return (None,)

NODE_CLASS_MAPPINGS = {
    "LivepeerT2I": LivepeerT2I,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerT2I": "Livepeer T2I", 
}
