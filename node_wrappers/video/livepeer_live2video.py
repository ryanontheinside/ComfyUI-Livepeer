import torch
import numpy as np
import json
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_base import LivepeerBase
import uuid

class LivepeerLive2Video(LivepeerBase):
    JOB_TYPE = "live2video"  # Unique job type for live video to video

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),  # Stream URL or playback ID
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": ""}),  # Transformer model ID
                "params": ("STRING", {"multiline": True, "default": "{}"}),  # JSON object with model parameters
                "webhook_id": ("STRING", {"multiline": False, "default": ""}),  # Webhook ID for notifications
                "output_type": ("STRING", {"multiline": False, "default": "rtmp"}),  # "rtmp", "hls", etc.
                "output_location": ("STRING", {"multiline": False, "default": ""}),  # Output location/URL
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("video_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "live_to_video"
    CATEGORY = "Livepeer"

    def live_to_video(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, 
                      url, model_id="", params="{}", webhook_id="", output_type="rtmp", output_location=""):
        # Skip API call if disabled
        if not enabled:
            return (None,)
        
        # Parse model parameters JSON
        try:
            params_dict = json.loads(params) if params and params != "{}" else {}
        except json.JSONDecodeError:
            raise ValueError("Invalid params JSON format")
        
        # Create Params object if params were provided
        params_obj = None
        if params_dict:
            params_obj = components.Params(**params_dict)
        
        live2video_args = components.LiveVideoToVideoParams(
            url=url,
            model_id=model_id if model_id else None,
            params=params_obj,
            webhook_id=webhook_id if webhook_id else None,
            output_type=output_type if output_type else None,
            output_location=output_location if output_location else None
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            return livepeer.generate.live_video_to_video(request=live2video_args)

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

NODE_CLASS_MAPPINGS = {
    "LivepeerLive2Video": LivepeerLive2Video,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerLive2Video": "Livepeer Live to Video",
} 