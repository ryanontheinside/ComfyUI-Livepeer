import torch
import numpy as np
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_base import LivepeerBase
from ...config_manager import config_manager
import uuid

class LivepeerSegment(LivepeerBase):
    JOB_TYPE = "segment"  # Unique job type for segmentation

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default segment model from config
        default_model = config_manager.get_default_model("segment") or "facebook/sam2-hiera-large"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": default_model}),
                "points": ("STRING", {"multiline": True, "default": "[]"}),  # JSON array of [x, y] coordinates
                "boxes": ("STRING", {"multiline": True, "default": "[]"}),  # JSON array of [x1, y1, x2, y2] boxes
                "return_masks": ("BOOLEAN", {"default": True}),
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("image_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "segment_image"
    CATEGORY = "Livepeer"

    def segment_image(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, 
                      image, model_id="", points="[]", boxes="[]", return_masks=True):
        # Skip API call if disabled
        if not enabled:
            return (None,)
        
        # Prepare image for segmentation
        prepared_image = self.prepare_image(image)
        
        segment_args = components.BodyGenSegmentAnything2(
            image=prepared_image,
            model_id=model_id if model_id else None,
            points=points,
            boxes=boxes,
            return_masks=return_masks
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            return livepeer.generate.segment_anything2(request=segment_args)

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
    "LivepeerSegment": LivepeerSegment,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerSegment": "Livepeer Segment Anything",
} 