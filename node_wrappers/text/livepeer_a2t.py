import requests
import torch
import numpy as np
import tempfile
import soundfile as sf
import os
from livepeer_ai.models import components
from ...src.livepeer_base import LivepeerBase
from ...src.livepeer_media_processor import LivepeerMediaProcessor
from ...config_manager import config_manager
import uuid

class LivepeerA2T(LivepeerBase):
    JOB_TYPE = "a2t"

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Get default A2T model from config
        default_model = config_manager.get_default_model("A2T") or "openai/whisper-large-v3"
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": default_model}),
                "return_timestamps": (["none", "sentence", "word"], {"default": "none"}),
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
                    audio, model_id="", return_timestamps="none"):
        """
        Convert audio data to text using Livepeer's Audio-to-Text API.
        
        This method handles preprocessing of audio data and sending it to Livepeer for transcription.
        """
        # Skip API call if disabled
        if not enabled:
            return (None,)
        
        # Early validation of audio input
        if audio is None:
            config_manager.log("error", "No audio data provided")
            raise ValueError("No audio data provided to Livepeer Audio-to-Text")
        
        try:
            # Log basic info about the audio for debugging
            config_manager.log("info", f"Processing audio input for transcription using model: {model_id}")
            
            # Convert audio to temporary file - use the centralized processor with error handling
            temp_path = None
            file_handle = None
            
            try:
                # Convert ComfyUI audio format to a file suitable for the API
                temp_path, file_handle = LivepeerMediaProcessor.prepare_audio_from_comfy_format(audio)
                
                if temp_path is None or file_handle is None:
                    raise ValueError("Failed to prepare audio data from the provided ComfyUI audio format")
                
                config_manager.log("info", f"Audio prepared successfully: {temp_path}")
                
                # Convert return_timestamps to the format expected by the API
                if return_timestamps == "none":
                    timestamp_value = "false"
                elif return_timestamps == "sentence":
                    timestamp_value = "true"
                elif return_timestamps == "word":
                    timestamp_value = "word"
                else:
                    timestamp_value = "true"  # SDK default is "true"
                
                # Create the audio component
                audio_obj = components.Audio(
                    file_name=os.path.basename(temp_path),
                    content=file_handle
                )
                
                # Prepare API request
                a2t_args = components.BodyGenAudioToText(
                    audio=audio_obj,
                    model_id=model_id if model_id else "",
                    return_timestamps=timestamp_value
                )
                
                # Define the operation function for retry/async logic
                def operation_func(livepeer):
                    config_manager.log("info", f"Sending audio to Livepeer API")
                    return livepeer.generate.audio_to_text(request=a2t_args)
                
                # Execute synchronously or asynchronously based on user preference
                if run_async:
                    # Trigger async job and return job ID
                    job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
                    return (job_id,)
                else:
                    # Execute synchronously with timeout
                    response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func, 
                                                    synchronous_timeout=synchronous_timeout)
                    
                    # Generate Job ID and store result directly for sync mode
                    job_id = str(uuid.uuid4())
                    self._store_sync_result(job_id, self.JOB_TYPE, response)
                    return (job_id,)
                
            finally:
                # Clean up resources
                if file_handle:
                    try:
                        file_handle.close()
                    except:
                        pass
                        
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        config_manager.log("info", f"Temporary audio file deleted")
                    except:
                        config_manager.log("warning", f"Failed to delete temporary audio file")
                
        except Exception as e:
            config_manager.log("error", f"Error in audio-to-text processing: {str(e)}")
            raise RuntimeError(f"Livepeer Audio-to-Text failed: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "LivepeerA2T": LivepeerA2T,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerA2T": "Livepeer Audio to Text",
} 