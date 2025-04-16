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
                # Set intermediate status indicating completion but pending delivery by getter node
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
        
        # Create file-like object for the API using the correct Image class
        return Image(
            file_name="input_image.png",
            content=img_byte_arr
        ) 

# --- New Job Getter Node Logic ---

class LivepeerJobGetter:
    # Added BOOLEAN output for readiness status
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN", "STRING") 
    RETURN_NAMES = ("image_output", "video_urls_output", "job_status", "error_message", "image_ready", "video_path")
    FUNCTION = "get_job_result"
    CATEGORY = "Livepeer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "job_id": ("STRING", {"multiline": False, "default": ""}),
             },
            "optional": {
                 "download_video": ("BOOLEAN", {"default": True}),
             }
        }

    @classmethod
    def IS_CHANGED(s, job_id):
        global _livepeer_job_store, _job_store_lock
        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id)
            current_status = job_info.get('status', 'not_found') if job_info else 'not_found'

            # For terminal states, return a stable tuple
            if current_status in ['delivered', 'failed']:
                return (job_id, current_status)
            # For non-terminal states (including the new intermediate one), return changing tuple
            else:
                return (job_id, current_status, time.time())

    def get_job_result(self, job_id, download_video=True):
        global _livepeer_job_store, _job_store_lock

        blank_height = 64
        blank_width = 64
        blank_image = torch.zeros((1, blank_height, blank_width, 3), dtype=torch.float32)

        # --- Check current status --- 
        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id)
            if not job_info:
                print(f"Livepeer Job Getter: Job {job_id} not found in store yet.")
                return (blank_image, None, "not_found", f"Job ID {job_id} not found in store.", False, None)
            
            status = job_info.get('status')
            job_type = job_info.get('type')
            # Make a copy to avoid modifying the store directly unless intended
            job_info_copy = job_info.copy()

        # --- Handle Based on Status --- 

        # If completed and pending delivery, process, output, store result, and mark as delivered
        if status == 'completed_pending_delivery':
            print(f"Livepeer Job Getter: Job {job_id} completed. Processing and delivering result.")
            result = job_info_copy.get('result') # Use copy here
            base_processor = LivepeerBase()
            image_out, video_urls_out, error_out, video_path = None, None, None, None
            image_ready = False

            try:
                # --- Result Processing --- 
                if job_type in ['t2i', 'i2i', 'upscale']:
                    if hasattr(result, 'image_response') and result.image_response:
                        image_out = base_processor.process_image_response(result)
                        image_ready = image_out is not None
                    else: error_out = f"Completed job {job_id} ({job_type}) has no image_response."
                elif job_type == 'i2v':
                    if hasattr(result, 'video_response') and result.video_response:
                        processed_urls = base_processor.process_video_response(result)
                        video_urls_out = processed_urls[0] if processed_urls else None
                        image_ready = bool(video_urls_out)
                        
                        # Download video if requested
                        if download_video and video_urls_out:
                            try:
                                video_path = base_processor.download_video(video_urls_out)
                                print(f"Downloaded video to {video_path}")
                            except Exception as e:
                                print(f"Error downloading video: {e}")
                                video_path = f"Error: {str(e)}"
                    else: error_out = f"Completed job {job_id} ({job_type}) has no video_response."
                elif job_type == 'i2t':
                    text_result = None
                    if hasattr(result, 'text_response'): text_result = str(result.text_response)
                    elif hasattr(result, 'text'): text_result = str(result.text)
                    elif isinstance(result, str): text_result = result
                    if text_result is not None: video_urls_out = text_result; image_ready = True
                    else: error_out = f"Completed job {job_id} ({job_type}) lacks expected text field."
                else: error_out = f"Unknown job type '{job_type}' for job {job_id}"
                # --- End Result Processing --- 

                if error_out:
                    print(f"Livepeer Job Getter: Error processing result for {job_id}: {error_out}")
                    return (blank_image, video_urls_out, "processing_error", error_out, False, None)
                else:
                    final_image_out = image_out if image_ready and image_out is not None else blank_image
                    print(f"Livepeer Job Getter: Job {job_id} processed successfully. Storing result and marking as delivered.")
                    # --- Store Processed Result & Mark as Delivered --- 
                    with _job_store_lock:
                        if job_id in _livepeer_job_store:
                             # Store the processed outputs for future retrieval
                            _livepeer_job_store[job_id]['processed_image'] = final_image_out 
                            _livepeer_job_store[job_id]['processed_urls'] = video_urls_out
                            _livepeer_job_store[job_id]['video_path'] = video_path
                            _livepeer_job_store[job_id]['status'] = 'delivered'
                    # --- Return Processed Results --- 
                    return (final_image_out, video_urls_out, 'delivered', None, image_ready, video_path)

            except Exception as e:
                print(f"Livepeer Job Getter: Exception processing result for job {job_id}: {e}")
                traceback.print_exc()
                return (blank_image, None, "processing_error", f"Exception processing result: {str(e)}", False, None)

        # If pending, just report status and return placeholders
        elif status == 'pending':
            print(f"Livepeer Job Getter: Job {job_id} is still pending.")
            return (blank_image, None, status, "Job is pending.", False, None)
        
        # If already delivered, retrieve and return the STORED results
        elif status == 'delivered':
             print(f"Livepeer Job Getter: Job {job_id} already delivered. Returning stored result.")
             # Retrieve stored processed results
             stored_image = job_info_copy.get('processed_image', blank_image) # Use blank as default if missing
             stored_urls = job_info_copy.get('processed_urls', None)
             stored_video_path = job_info_copy.get('video_path', None)
             
             # If video path not available but URL is, and download is requested, try to download now
             if stored_video_path is None and stored_urls and download_video:
                 try:
                     base_processor = LivepeerBase()
                     stored_video_path = base_processor.download_video(stored_urls)
                     # Update the stored video path
                     with _job_store_lock:
                         if job_id in _livepeer_job_store:
                             _livepeer_job_store[job_id]['video_path'] = stored_video_path
                 except Exception as e:
                     print(f"Error downloading video: {e}")
                     stored_video_path = f"Error: {str(e)}"
             
             # Determine readiness based on stored data (primarily if image exists beyond blank)
             is_stored_image_valid = stored_image is not None and stored_image.shape[1] > blank_height # Check if not the default blank
             stored_ready = is_stored_image_valid or bool(stored_urls)
             return (stored_image, stored_urls, status, "Results previously delivered.", stored_ready, stored_video_path)

        # If failed, return placeholders and 'failed' status
        elif status == 'failed':
            error_msg = job_info_copy.get('error', 'Unknown error')
            print(f"Livepeer Job Getter: Job {job_id} failed: {error_msg}")
            return (blank_image, None, status, error_msg, False, None)

        # Handle any other unknown status (like processing_error)
        else:
            print(f"Livepeer Job Getter: Job {job_id} has status: {status}")
            error_msg = job_info_copy.get('error', f"Unknown status '{status}'") 
            return (blank_image, None, status, error_msg, False, None)

# Note: Need to register LivepeerJobGetter in __init__.py 