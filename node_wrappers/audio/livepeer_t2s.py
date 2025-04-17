import torch
import numpy as np
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_base import LivepeerBase
from ...config_manager import config_manager
import uuid

class LivepeerT2S(LivepeerBase):
    JOB_TYPE = "t2s"  # Unique job type for text-to-speech

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default T2S model from config
        default_model = config_manager.get_default_model("T2S") or "parler-tts/parler-tts-large-v1"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, welcome to Livepeer text to speech."}),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": default_model}),
                "description": ("STRING", {"multiline": False, "default": "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."}),  # e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer"
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("audio_job",)  # Create a new job type for audio
    RETURN_NAMES = ("job_id",)
    FUNCTION = "text_to_speech"
    CATEGORY = "Livepeer"

    def text_to_speech(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, 
                       text, model_id="", description=""):
        # Skip API call if disabled
        if not enabled:
            return (None,)
        
        t2s_args = components.TextToSpeechParams(
            text=text,
            model_id=model_id if model_id else None,
            description=description if description else None
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            return livepeer.generate.text_to_speech(request=t2s_args)

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
    "LivepeerT2S": LivepeerT2S,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerT2S": "Livepeer Text to Speech",
} 