import torch
import numpy as np
import json
from livepeer_ai import Livepeer
from livepeer_ai.models import components
from ...src.livepeer_core import LivepeerBase
import uuid

class LivepeerLLM(LivepeerBase):
    JOB_TYPE = "llm"  # Unique job type for LLM operations

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "What's the meaning of life?"}),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": ""}),  # e.g., "gpt-4", "mistral-large-latest"
                "messages": ("STRING", {"multiline": True, "default": "[]"}),  # JSON array of message objects
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),  # System prompt instruction
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("text_job",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "run_llm"
    CATEGORY = "Livepeer"

    def run_llm(self, enabled, api_key, max_retries, retry_delay, run_async, synchronous_timeout, 
               prompt, model_id="", messages="[]", system_prompt="", temperature=0.7, top_p=1.0, 
               max_tokens=1024, frequency_penalty=0.0, presence_penalty=0.0):
        # Skip API call if disabled
        if not enabled:
            return (None,)

        # Process messages parameter
        parsed_messages = []
        
        # If messages string is provided, parse it
        if messages and messages != "[]":
            try:
                parsed_messages = json.loads(messages)
            except json.JSONDecodeError:
                raise ValueError("Invalid messages JSON format")
        
        # If system prompt is provided and no messages, create a system message
        if system_prompt and not parsed_messages:
            parsed_messages.append({"role": "system", "content": system_prompt})
        
        # Add user message with prompt if not in messages
        if not any(msg.get("role") == "user" for msg in parsed_messages):
            parsed_messages.append({"role": "user", "content": prompt})
        
        # Convert to LLMMessage objects
        llm_messages = []
        for msg in parsed_messages:
            llm_messages.append(components.LLMMessage(
                role=msg.get("role"),
                content=msg.get("content")
            ))
        
        llm_args = components.LLMRequest(
            model=model_id if model_id else None,
            messages=llm_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            return livepeer.generate.llm(request=llm_args)

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
    "LivepeerLLM": LivepeerLLM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerLLM": "Livepeer LLM",
} 