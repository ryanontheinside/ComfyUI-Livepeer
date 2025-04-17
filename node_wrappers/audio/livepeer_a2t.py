import requests
import torch
import numpy as np
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_core import LivepeerBase
from ...config_manager import config_manager
import uuid
import os

class LivepeerA2T(LivepeerBase):
    JOB_TYPE = "i2t"  # Using the same job type as image-to-text since both return text

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default A2T model from config
        default_model = config_manager.get_default_model("A2T") or "openai/whisper-large-v3"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "audio_file": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": default_model}),
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g., "en" for English
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "response_format": ("STRING", {"multiline": False, "default": "text"}),  # text, json, verbose_json, etc.
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("text_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "audio_to_text"
    CATEGORY = "Livepeer"

    def audio_to_text(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, 
                      audio_file, model_id="", language="", prompt="", temperature=0.0, response_format="text"):
        # Skip API call if disabled
        if not enabled:
            return (None,)
        
        # Check if audio file exists
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file does not exist: {audio_file}")
        
        # Prepare audio file
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        audio = components.Audio(
            file_name=os.path.basename(audio_file),
            content=audio_data
        )
        
        a2t_args = components.BodyGenAudioToText(
            audio=audio,
            model_id=model_id if model_id else None,
            language=language if language else None,
            prompt=prompt if prompt else None,
            temperature=temperature,
            response_format=response_format if response_format else None
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            return livepeer.generate.audio_to_text(request=a2t_args)

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
    "LivepeerA2T": LivepeerA2T,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerA2T": "Livepeer Audio to Text",
} 