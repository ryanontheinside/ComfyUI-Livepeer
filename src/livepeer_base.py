import time
import numpy as np
import torch
from PIL import Image as PILImage
from io import BytesIO
import requests
from livepeer_ai import Livepeer
from livepeer_ai.models.components import Image
import threading
import uuid
import traceback
import os
import shutil
import sys

import comfy.model_management
from ..config_manager import config_manager

from .livepeer_job_getter import _livepeer_job_store, _job_store_lock

class LivepeerBase:
    """Base class for Livepeer nodes with common functionality and retry logic"""
    
    @classmethod
    def get_common_inputs(cls):
        """Common input parameters for all Livepeer nodes"""
        # Get default values from config
        default_api_key = config_manager.get_api_key()
        max_retries, retry_delay = config_manager.get_retry_settings()
        default_timeout = config_manager.get_timeout()
        
        return {
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "When disabled, API calls will be skipped"}),
            "api_key": ("STRING", {"default": default_api_key}),
            "max_retries": ("INT", {"default": max_retries, "min": 1, "max": 10}),
            "retry_delay": ("FLOAT", {"default": retry_delay, "min": 0.5, "max": 10.0}),
            "run_async": ("BOOLEAN", {"default": False}), 
            "synchronous_timeout": ("FLOAT", {"default": default_timeout, "min": 5.0, "max": 600.0, "tooltip": "Timeout for synchronous operations (per retry). For async operations, this is ignored."}),
        }
    #TODO change to simply timeout for both
    
    def _execute_livepeer_operation(self, api_key, max_retries, retry_delay, operation_func, job_id, job_type):
        """Internal method to run the operation and update the job store."""
        global _livepeer_job_store, _job_store_lock
        try:
            # Store job type immediately for getter node reference
            with _job_store_lock:
                 if job_id in _livepeer_job_store:
                     _livepeer_job_store[job_id]['type'] = job_type

            config_manager.log("info", f"Livepeer Async Job {job_id} ({job_type}): Starting execution.")
            result = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func)
            config_manager.log("info", f"Livepeer Async Job {job_id} ({job_type}): Execution successful.")
            with _job_store_lock:
                # Set intermediate status indicating completion but pending delivery by getter node
                if job_id in _livepeer_job_store: # Check if job wasn't cancelled/removed
                    _livepeer_job_store[job_id].update({'status': 'completed_pending_delivery', 'result': result})
        except Exception as e:
            config_manager.log("error", f"Livepeer Async Job {job_id} ({job_type}): Execution failed.")
            config_manager.handle_error(e, f"Error in async job {job_id}", raise_error=False)
            with _job_store_lock:
                 if job_id in _livepeer_job_store: # Check if job wasn't cancelled/removed
                    _livepeer_job_store[job_id].update({'status': 'failed', 'error': str(e)})

    def trigger_async_job(self, api_key, max_retries, retry_delay, operation_func, job_type):
        """Initiates the Livepeer operation in a background thread and returns a job ID."""
        global _livepeer_job_store, _job_store_lock
        job_id = str(uuid.uuid4())

        with _job_store_lock:
            _livepeer_job_store[job_id] = {'status': 'pending', 'type': job_type} # Initial state

        config_manager.log("info", f"Livepeer Async Job {job_id} ({job_type}): Triggered.")

        thread = threading.Thread(
            target=self._execute_livepeer_operation,
            args=(api_key, max_retries, retry_delay, operation_func, job_id, job_type),
            daemon=True # Ensure threads don't block ComfyUI exit
        )
        thread.start()
        return job_id
        
    def _store_sync_result(self, job_id, job_type, result):
        """Stores synchronous operation result in the job store for retrieval by getter nodes."""

        #NOTE: the choice to make sync results retrievable only by using the job getter is to handle 
        # both syncronous and asyncronuous operations in ComfyUI - if the job is asyncrounous, 
        # we cannot have the main node return some arbitrary result, like a blank image of arbitrary size
        global _livepeer_job_store, _job_store_lock
        
        with _job_store_lock:
            _livepeer_job_store[job_id] = {
                'status': 'completed_pending_delivery',
                'type': job_type,
                'result': result
            }
        config_manager.log("info", f"Livepeer Sync Job {job_id} ({job_type}): Result stored for getter")

    # Helper for synchronous execution thread
    def _run_operation_thread(self, api_key, operation_func, result_container):
        """Executes the actual Livepeer SDK call in a separate thread."""
        try:
            # Check for interruption before starting
            if comfy.model_management.processing_interrupted():
                result_container['error'] = comfy.model_management.InterruptProcessingException("Operation cancelled")
                return
                
            # Create Livepeer instance within the thread
            with Livepeer(http_bearer=api_key) as livepeer:
                result = operation_func(livepeer)
            result_container['result'] = result
        except Exception as e:
            result_container['error'] = e
        finally:
            # Signal completion regardless of outcome
            result_container['done'] = True

    def execute_with_retry(self, api_key, max_retries, retry_delay, operation_func, synchronous_timeout=120.0):
        """
        Execute a Livepeer API operation with retry logic.
        Uses non-blocking polling for UI responsiveness and cancellation.
        """
        attempts = 0
        last_error = None
        current_retry_delay = float(retry_delay) # Ensure it's float

        # Helper function to check if processing should be interrupted
        def check_interrupt():
            # Check ComfyUI's global interrupt flag
            if comfy.model_management.processing_interrupted():
                raise comfy.model_management.InterruptProcessingException("Operation cancelled by user")

        while attempts < max_retries:
            # Check for interruption at the start of each attempt
            check_interrupt()
            
            config_manager.log("info", f"Livepeer Attempt {attempts + 1}/{max_retries}")
            result_container = {'result': None, 'error': None, 'done': False}
            cancelled_during_poll = False # Flag to track cancellation within the poll loop

            thread = threading.Thread(
                target=self._run_operation_thread,
                args=(api_key, operation_func, result_container),
                daemon=True 
            )
            thread.start()

            start_time = time.time()
            timed_out = False
            
            # Polling loop - yields control to ComfyUI
            while not result_container['done']:
                # Check for interruption regularly
                try:
                    check_interrupt()
                except comfy.model_management.InterruptProcessingException:
                    # Handle cancellation detected DURING polling
                    last_error = comfy.model_management.InterruptProcessingException("Operation cancelled by user")
                    config_manager.log("info", "Livepeer operation cancelled by user during poll.")
                    # TODO: Pending Livepeer API cancellation support, send a cancellation request
                    # to the server here to stop processing and free up GPU resources.
                    cancelled_during_poll = True
                    break # Exit polling loop

                # Check for timeout
                if time.time() - start_time > synchronous_timeout:
                    timed_out = True
                    config_manager.log("warning", f"Livepeer Attempt {attempts + 1} timed out after {synchronous_timeout} seconds.")
                    # We don't kill the thread, just stop waiting for it in this attempt
                    break 
                
                # Yield control to allow UI updates and cancellation checks
                time.sleep(0.1) 

            # --- Processing after polling loop ---

            # 1. Handle cancellation that occurred *during* the poll loop
            if cancelled_during_poll:
                 raise last_error # Propagate the stored exception

            # 2. Check for cancellation that occurred *after* the poll loop finished
            try:
                check_interrupt()
            except comfy.model_management.InterruptProcessingException as e:
                 # If interrupted here, just raise immediately. Don't proceed.
                 config_manager.log("info", "Livepeer operation cancelled by user after polling loop.")
                 raise e # Propagate immediately

            # 3. Handle normal completion or timeout (only if not cancelled above)
            if not timed_out and result_container['done']:
                 thread.join(timeout=1.0) # Short timeout for join after done flag is set

            # 4. Process result/error from this attempt
            if timed_out: # This now only reflects actual timeouts, not cancellations
                 last_error = TimeoutError(f"Operation timed out after {synchronous_timeout} seconds")
                 attempts += 1 # Count timeout as a failed attempt
            elif result_container['error']:
                attempts += 1
                last_error = result_container['error']
                config_manager.log("error", f"Livepeer Attempt {attempts} failed: {str(last_error)}")
            elif 'result' in result_container and result_container['result'] is not None:
                 config_manager.log("info", f"Livepeer Attempt {attempts + 1} successful.")
                 return result_container['result'] # Success! Exit the function.
            else: # Should not happen if done is True and no error, but handle defensively
                 attempts += 1
                 last_error = RuntimeError("Unknown error: Operation thread finished but provided no result or error.")
                 config_manager.log("error", f"Livepeer Attempt {attempts} failed with unknown error.")

            # One final cancellation check before considering retry
            check_interrupt()

            # If attempt failed (error or timeout) and more retries left:
            if attempts < max_retries:
                config_manager.log("info", f"Retrying in {current_retry_delay} seconds...")
                # Use polling sleep for retry delay to remain responsive
                delay_start_time = time.time()
                while time.time() - delay_start_time < current_retry_delay:
                    # Check for interruption during retry delay
                    try:
                        check_interrupt()
                    except comfy.model_management.InterruptProcessingException:
                        # Propagate the interruption immediately
                        raise
                        
                    # Yield control during retry delay sleep
                    time.sleep(0.1) 
                current_retry_delay *= 1.5 # Exponential backoff
            
        # If loop finishes, all retries failed
        error_msg = f"Operation failed after {max_retries} attempts. Last error: {str(last_error)}"
        return config_manager.handle_error(
            RuntimeError(error_msg), 
            "Livepeer operation failed after all retry attempts"
        )

    def process_image_response(self, response):
        """Process image response into BHWC tensor format"""
        try:
            images = []
            for img_data in response.image_response.images:
                image_url = img_data.url
                
                # Get image data
                img_response = requests.get(image_url).content
                img = PILImage.open(BytesIO(img_response)).convert("RGB")
                
                # Convert to numpy array with proper normalization
                img_np = np.array(img).astype(np.float32) / 255.0
                images.append(img_np)
            
            # Stack into batch tensor [B, H, W, C]
            img_batch = np.stack(images, axis=0)
            return torch.from_numpy(img_batch)
        except Exception as e:
            return config_manager.handle_error(e, "Error processing image response")
    
    
    def download_video(self, url, output_dir=None):
        """
        Download video from URL to ComfyUI output directory
        
        Args:
            url: Video URL to download
            output_dir: Custom output directory (defaults to ComfyUI output)
            
        Returns:
            str: Full path to downloaded video file
        """
        try:
            # Use configured output directory if not specified
            if output_dir is None:
                output_dir = config_manager.get_output_path("videos")
            
            # Generate unique filename using timestamp
            timestamp = int(time.time())
            video_filename = f"livepeer_video_{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            # Download the video
            config_manager.log("info", f"Downloading video from {url} to {video_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
                
            return video_path
        except Exception as e:
            return config_manager.handle_error(e, "Error downloading video")
    
    def prepare_image(self, image_batch):
        """
        Convert ComfyUI image tensor batch to file for API upload
        """
        try:
            # For now, we can only process one image at a time with the Livepeer API
            # Future API versions might support batch processing
            #TODO: review, decide if we risk handling batches here manually. Probably.
            if image_batch.shape[0] > 1:
                config_manager.log("warning", f"Livepeer API only supports one image at a time. Using first image from batch of {image_batch.shape[0]}")
            
            # Convert tensor to PIL Image for uploading
            img = image_batch[0]  # First image in batch
            pil_img = torch.clamp(img * 255, 0, 255).cpu().numpy().astype(np.uint8)
            pil_img = PILImage.fromarray(pil_img)
            
            # Save to BytesIO for uploading
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Convert BytesIO to bytes for the Livepeer API
            img_bytes = img_byte_arr.getvalue()
            
            # Create file-like object for the API using the correct Image class
            return Image(
                file_name="input_image.png",
                content=img_bytes
            )
        except Exception as e:
            return config_manager.handle_error(e, "Error preparing image") 