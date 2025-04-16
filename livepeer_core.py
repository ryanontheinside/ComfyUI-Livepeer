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
import traceback # Added for better error logging in threads
import os
import shutil
import sys

import comfy.model_management

# Global store for async job status and results
# Structure: {job_id: {'status': 'pending'/'completed'/'failed', 'result': ..., 'error': ..., 'type': 't2i'/'i2i'/...}}
_livepeer_job_store = {}
_job_store_lock = threading.Lock()

class LivepeerBase:
    """Base class for Livepeer nodes with common functionality and retry logic"""
    
    @classmethod
    def get_common_inputs(cls):
        """Common input parameters for all Livepeer nodes"""
        return {
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "When disabled, API calls will be skipped"}),
            "api_key": ("STRING", {"default": "17101937-98f4-4c99-bdb2-e6499fda7ef8"}),
            "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            "retry_delay": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0}),
            "run_async": ("BOOLEAN", {"default": False}), # Add async toggle
            "synchronous_timeout": ("FLOAT", {"default": 120.0, "min": 5.0, "max": 600.0, "tooltip": "Timeout for synchronous operations (per retry). For async operations, this is ignored."}), # Timeout for sync mode
        }
    
    def _execute_livepeer_operation(self, api_key, max_retries, retry_delay, operation_func, job_id, job_type):
        """Internal method to run the operation and update the job store."""
        global _livepeer_job_store, _job_store_lock
        try:
            # Store job type immediately for getter node reference
            with _job_store_lock:
                 if job_id in _livepeer_job_store:
                     _livepeer_job_store[job_id]['type'] = job_type

            print(f"Livepeer Async Job {job_id} ({job_type}): Starting execution.")
            result = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func)
            print(f"Livepeer Async Job {job_id} ({job_type}): Execution successful.")
            with _job_store_lock:
                # Set intermediate status indicating completion but pending delivery by getter node
                if job_id in _livepeer_job_store: # Check if job wasn't cancelled/removed
                    _livepeer_job_store[job_id].update({'status': 'completed_pending_delivery', 'result': result})
        except Exception as e:
            print(f"Livepeer Async Job {job_id} ({job_type}): Execution failed.")
            traceback.print_exc() # Print full traceback for debugging
            with _job_store_lock:
                 if job_id in _livepeer_job_store: # Check if job wasn't cancelled/removed
                    _livepeer_job_store[job_id].update({'status': 'failed', 'error': str(e)})

    def trigger_async_job(self, api_key, max_retries, retry_delay, operation_func, job_type):
        """Initiates the Livepeer operation in a background thread and returns a job ID."""
        global _livepeer_job_store, _job_store_lock
        job_id = str(uuid.uuid4())

        with _job_store_lock:
            _livepeer_job_store[job_id] = {'status': 'pending', 'type': job_type} # Initial state

        print(f"Livepeer Async Job {job_id} ({job_type}): Triggered.")

        thread = threading.Thread(
            target=self._execute_livepeer_operation,
            args=(api_key, max_retries, retry_delay, operation_func, job_id, job_type),
            daemon=True # Ensure threads don't block ComfyUI exit
        )
        thread.start()
        return job_id

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
        Applies timeout to each attempt.
        Checks ComfyUI's interrupt flag directly for proper cancellation.
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
            
            print(f"Livepeer Attempt {attempts + 1}/{max_retries}")
            result_container = {'result': None, 'error': None, 'done': False}
            
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
                    # Mark the thread for termination
                    timed_out = True
                    last_error = comfy.model_management.InterruptProcessingException("Operation cancelled by user")
                    print("Livepeer operation cancelled by user.")
                    # Don't wait for thread to complete - allow ComfyUI to cancel
                    break
                
                # Check for timeout
                if time.time() - start_time > synchronous_timeout:
                    timed_out = True
                    print(f"Livepeer Attempt {attempts + 1} timed out after {synchronous_timeout} seconds.")
                    # We don't kill the thread, just stop waiting for it in this attempt
                    break 
                
                # Yield control to allow UI updates and cancellation checks
                time.sleep(0.1) 

            # Check one more time after polling loop ends
            try:
                check_interrupt()
            except comfy.model_management.InterruptProcessingException:
                if not timed_out:  # Only set if not already set
                    timed_out = True
                    last_error = comfy.model_management.InterruptProcessingException("Operation cancelled by user")
                    print("Livepeer operation cancelled by user.")

            # Handle result
            if not timed_out and result_container['done']:
                thread.join(timeout=1.0) # Short timeout for join after done flag is set

            # Process result/error from this attempt
            if timed_out and isinstance(last_error, comfy.model_management.InterruptProcessingException):
                # If the interruption was requested by user, propagate the exception immediately
                raise last_error
            elif timed_out:
                last_error = TimeoutError(f"Operation timed out after {synchronous_timeout} seconds")
                attempts += 1 # Count timeout as a failed attempt
            elif result_container['error']:
                attempts += 1
                last_error = result_container['error']
                print(f"Livepeer Attempt {attempts} failed: {str(last_error)}")
            elif 'result' in result_container and result_container['result'] is not None:
                 print(f"Livepeer Attempt {attempts + 1} successful.")
                 return result_container['result'] # Success! Exit the function.
            else: # Should not happen if done is True and no error, but handle defensively
                 attempts += 1
                 last_error = RuntimeError("Unknown error: Operation thread finished but provided no result or error.")
                 print(f"Livepeer Attempt {attempts} failed with unknown error.")

            # One final cancellation check before considering retry
            check_interrupt()

            # If attempt failed (error or timeout) and more retries left:
            if attempts < max_retries:
                print(f"Retrying in {current_retry_delay} seconds...")
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
        raise RuntimeError(f"Operation failed after {max_retries} attempts. Last error: {str(last_error)}")

    def process_image_response(self, response):
        """Process image response into BHWC tensor format"""
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
    
    def process_video_response(self, response):
        """Process video response - return URLs"""
        return [image.url for image in response.video_response.images]
    
    def download_video(self, url, output_dir=None):
        """
        Download video from URL to ComfyUI output directory
        
        Args:
            url: Video URL to download
            output_dir: Custom output directory (defaults to ComfyUI output)
            
        Returns:
            str: Full path to downloaded video file
        """
        # Determine ComfyUI output directory if not specified
        if output_dir is None:
            # Default ComfyUI output is in the 'output' folder at the project root
            output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'output')
            # Ensure 'livepeer' subfolder exists
            output_dir = os.path.join(output_dir, 'livepeer')
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename using timestamp
        import time
        timestamp = int(time.time())
        video_filename = f"livepeer_video_{timestamp}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # Download the video
        print(f"Downloading video from {url} to {video_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(video_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
            
        return video_path
    
    def prepare_image(self, image_batch):
        """
        Convert ComfyUI image tensor batch to file for API upload
        For nodes that only support single image input, the caller should handle
        processing batch elements separately
        
        Args:
            image_batch: Tensor of shape [B,H,W,C] with one or more images
            
        Returns:
            Image: Single Livepeer Image object from the first batch element
            This is a limitation of the current Livepeer API which doesn't support
            batch processing for image inputs
        """
        # For now, we can only process one image at a time with the Livepeer API
        # Future API versions might support batch processing
        if image_batch.shape[0] > 1:
            print(f"Warning: Livepeer API only supports one image at a time. Using first image from batch of {image_batch.shape[0]}")
        
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

# --- Old Job Getter Node Logic Removed ---
# The LivepeerJobGetter class has been moved and refactored into
# livepeer_jobgetter.py with a base class and specific implementations.

# --- End Removed Logic --- 