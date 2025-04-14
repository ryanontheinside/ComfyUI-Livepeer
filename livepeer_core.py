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
            "api_key": ("STRING", {"default": "17101937-98f4-4c99-bdb2-e6499fda7ef8"}),
            "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            "retry_delay": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0}),
            "run_async": ("BOOLEAN", {"default": False}), # Add async toggle
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
                _livepeer_job_store[job_id].update({'status': 'completed', 'result': result})
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

    def execute_with_retry(self, api_key, max_retries, retry_delay, operation_func):
        """Execute a Livepeer API operation with retry logic"""
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            try:
                with Livepeer(http_bearer=api_key) as livepeer:
                    result = operation_func(livepeer)
                return result
            except Exception as e:
                attempts += 1
                last_error = e
                print(f"Livepeer API attempt {attempts} failed: {str(e)}")
                
                if attempts < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 1.5
        
        # If all retries failed
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")
    
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
        return [video.url for video in response.video_response.videos]
    
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
        # This is a limitation in the API, not our code
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
        
        # Create file-like object for the API using the correct Image class
        return Image(
            file_name="input_image.png",
            content=img_byte_arr
        ) 

# --- New Job Getter Node Logic ---

class LivepeerJobGetter:
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING") # Output image, video urls (as string), job_status, error_message
    RETURN_NAMES = ("image_output", "video_urls_output", "job_status", "error_message")
    FUNCTION = "get_job_result"
    CATEGORY = "Livepeer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "job_id": ("STRING", {"multiline": False, "default": ""}),
                 "poll_interval": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.1}), # How often to check status if pending
                 "timeout": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 600.0, "step": 1.0}), # Max time to wait
             },
             "optional": {
                 "previous_status": ("STRING", {"forceInput": True, "default": ""}) # To help trigger re-evaluation
             }
        }

    def get_job_result(self, job_id, poll_interval, timeout, previous_status=None):
        global _livepeer_job_store, _job_store_lock
        start_time = time.time()

        while True:
            with _job_store_lock:
                job_info = _livepeer_job_store.get(job_id)

            if not job_info:
                return (None, None, f"not_found", f"Job ID {job_id} not found in store.")

            status = job_info.get('status')
            job_type = job_info.get('type') # Get the job type

            if status == 'completed':
                print(f"Livepeer Job Getter: Job {job_id} completed.")
                result = job_info.get('result')
                # Process result based on job type - reusing base class methods
                base_processor = LivepeerBase() # Instantiate base to access processing methods
                image_out = None
                video_urls_out = None
                error_out = None

                try:
                    if job_type in ['t2i', 'i2i', 'upscale']:
                         if hasattr(result, 'image_response') and result.image_response:
                             image_out = base_processor.process_image_response(result)
                         else:
                              error_out = f"Completed job {job_id} ({job_type}) has no image_response."
                    elif job_type == 'i2v':
                         if hasattr(result, 'video_response') and result.video_response:
                            video_urls_out = "\n".join(base_processor.process_video_response(result))
                         else:
                             error_out = f"Completed job {job_id} ({job_type}) has no video_response."
                    elif job_type == 'i2t':
                         # Assuming i2t returns text directly or in a specific field
                         # Adjust based on actual i2t response structure
                         if hasattr(result, 'text_response'): # Example field
                             video_urls_out = str(result.text_response)
                         elif hasattr(result, 'text'): # Common alternative
                              video_urls_out = str(result.text)
                         elif isinstance(result, str): # If the result *is* the string
                             video_urls_out = result
                         else:
                             error_out = f"Completed job {job_id} ({job_type}) lacks expected text field (e.g., text_response, text)."
                    else:
                         error_out = f"Unknown job type '{job_type}' for job {job_id}"

                    # Check for processing errors before returning
                    if error_out:
                         print(f"Livepeer Job Getter: Error processing result for {job_id}: {error_out}")
                         # Return error status even if original job completed
                         return (None, None, "processing_error", error_out)
                    else:
                         # Optionally remove completed job from store to prevent memory leak
                         # with _job_store_lock:
                         #     if job_id in _livepeer_job_store: del _livepeer_job_store[job_id]
                         print(f"Livepeer Job Getter: Job {job_id} processed successfully.")
                         return (image_out, video_urls_out, status, None)

                except Exception as e:
                     # Catch errors during the result processing itself
                     print(f"Livepeer Job Getter: Exception processing result for job {job_id}: {e}")
                     traceback.print_exc()
                     return (None, None, "processing_error", f"Exception processing result: {str(e)}")


            elif status == 'failed':
                error_msg = job_info.get('error', 'Unknown error')
                print(f"Livepeer Job Getter: Job {job_id} failed: {error_msg}")
                # Optionally remove failed job from store
                # with _job_store_lock:
                #     del _livepeer_job_store[job_id]
                return (None, None, status, error_msg)

            elif status == 'pending':
                if time.time() - start_time > timeout:
                     print(f"Livepeer Job Getter: Job {job_id} timed out after {timeout} seconds.")
                     return (None, None, "timeout", f"Job timed out after {timeout}s.")

                # Wait before checking again
                print(f"Livepeer Job Getter: Job {job_id} is pending. Checking again in {poll_interval}s.")
                time.sleep(poll_interval)
                # Continue loop

            else:
                # Should not happen
                return (None, None, "unknown_status", f"Unknown status '{status}' for job {job_id}.")

# Note: Need to register LivepeerJobGetter in __init__.py 