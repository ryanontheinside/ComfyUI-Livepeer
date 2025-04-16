import time
import torch
import traceback
from .livepeer_core import _livepeer_job_store, _job_store_lock, LivepeerBase 

# Define default blank image for placeholders
BLANK_HEIGHT = 64
BLANK_WIDTH = 64
BLANK_IMAGE = torch.zeros((1, BLANK_HEIGHT, BLANK_WIDTH, 3), dtype=torch.float32)

class LivepeerJobGetterBase:
    """Base class for Livepeer Job Getter nodes."""
    CATEGORY = "Livepeer/Getters"
    RETURN_TYPES = ("STRING", "STRING") # Common outputs
    RETURN_NAMES = ("job_status", "error_message")

    # INPUT_TYPES is defined in subclasses with specific job ID types

    @classmethod
    def IS_CHANGED(s, job_id):
        """Triggers re-execution if the job status changes or is in a non-terminal state."""
        if not job_id:
            return float("NaN") # Don't execute if no job_id

        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id)
            current_status = job_info.get('status', 'not_found') if job_info else 'not_found'

            # For terminal states, return a stable tuple
            if current_status in ['delivered', 'failed', 'processing_error', 'type_mismatch', 'not_found']:
                return (job_id, current_status)
            # For non-terminal states (pending, completed_pending_delivery), return changing tuple
            else:
                return (job_id, current_status, time.time())

    def _get_job_info(self, job_id):
        """Safely retrieves job information from the global store."""
        if not job_id:
             return None, "not_found", "No Job ID provided."
             
        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id)
            if not job_info:
                print(f"Livepeer Job Getter: Job {job_id} not found in store.")
                return None, "not_found", f"Job ID {job_id} not found in store."
            
            # Return a copy to prevent accidental modification outside the lock
            return job_info.copy(), job_info.get('status'), job_info.get('error')

    def _update_job_store_processed(self, job_id, processed_data, status='delivered'):
         """Updates the job store with processed results and sets status."""
         with _job_store_lock:
             if job_id in _livepeer_job_store:
                 _livepeer_job_store[job_id]['status'] = status
                 # Store processed data under specific keys
                 _livepeer_job_store[job_id].update(processed_data)
             else:
                 print(f"Warning: Job {job_id} disappeared from store before processed results could be saved.")
                 
    def _handle_terminal_state(self, job_info, status, error, expected_job_types, default_outputs):
        """Handles failed, processing_error, not_found, and type_mismatch states."""
        if status == 'failed':
            error_msg = error or 'Unknown failure reason.'
            print(f"Livepeer Job Getter: Job {job_info.get('job_id', 'N/A')} failed: {error_msg}")
            return default_outputs + (status, error_msg)
        elif status == 'processing_error':
            error_msg = error or 'Unknown processing error.'
            print(f"Livepeer Job Getter: Job {job_info.get('job_id', 'N/A')} had processing error: {error_msg}")
            return default_outputs + (status, error_msg)
        elif status == 'not_found':
            error_msg = error or f"Job ID {job_info.get('job_id', 'N/A')} not found."
            return default_outputs + (status, error_msg)
        elif status == 'type_mismatch':
            error_msg = error or f"Job type '{job_info.get('type')}' does not match expected types {expected_job_types} for this getter."
            print(f"Livepeer Job Getter: {error_msg}")
            return default_outputs + (status, error_msg)
        else: # Should not happen if IS_CHANGED works correctly
            return default_outputs + (status, f"Unexpected terminal state '{status}'.")

    def _handle_delivered_state(self, job_info, status, default_outputs):
        """Handles already delivered state by returning stored processed results."""
        print(f"Livepeer Job Getter: Job {job_info.get('job_id')} already delivered. Returning stored result.")
        # Implement retrieval of specific stored keys in subclasses
        # Base implementation returns defaults + status
        return default_outputs + (status, "Results previously delivered.")

# --- Specific Getter Implementations ---

class LivepeerImageJobGetter(LivepeerJobGetterBase):
    # Expected job types that produce images
    EXPECTED_JOB_TYPES = ["t2i", "i2i", "upscale"] 
    # Define specific input type for image-related jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("image_job", {})
         }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("IMAGE", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES
    RETURN_NAMES = ("image_output", "image_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_image_job_result"

    def get_image_job_result(self, job_id):
        default_outputs = (BLANK_IMAGE, False) # image_output, image_ready
        job_info, status, error = self._get_job_info(job_id)

        if status in ['failed', 'processing_error', 'not_found']:
            return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)
            
        job_type = job_info.get('type')
        if job_type not in self.EXPECTED_JOB_TYPES:
            status = 'type_mismatch'
            error = f"Expected one of {self.EXPECTED_JOB_TYPES}, got '{job_type}'"
            return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)

        if status == 'delivered':
            stored_image = job_info.get('processed_image', BLANK_IMAGE)
            stored_ready = stored_image is not None and stored_image.shape[1] > BLANK_HEIGHT
            return (stored_image, stored_ready) + (status, "Results previously delivered.")
            
        elif status == 'pending':
            print(f"Livepeer Job Getter: Job {job_id} ({job_type}) is still pending.")
            return default_outputs + (status, "Job is pending.")

        elif status == 'completed_pending_delivery':
            print(f"Livepeer Job Getter: Job {job_id} ({job_type}) completed. Processing image result.")
            result = job_info.get('result')
            image_out = BLANK_IMAGE
            image_ready = False
            processing_error = None
            
            try:
                if result and hasattr(result, 'image_response') and result.image_response:
                    base_processor = LivepeerBase() # Instantiate to access processing methods
                    image_out = base_processor.process_image_response(result)
                    image_ready = image_out is not None and image_out.shape[1] > BLANK_HEIGHT
                    print(f"Livepeer Job Getter: Job {job_id} processed successfully.")
                    processed_data = {'processed_image': image_out}
                    self._update_job_store_processed(job_id, processed_data, status='delivered')
                    return (image_out, image_ready) + ('delivered', None)
                else:
                    processing_error = f"Completed job {job_id} ({job_type}) has no valid image_response."
            except Exception as e:
                traceback.print_exc()
                processing_error = f"Exception processing image result for job {job_id}: {str(e)}"

            if processing_error:
                print(f"Livepeer Job Getter: Error processing image result for {job_id}: {processing_error}")
                processed_data = {'error': processing_error}
                self._update_job_store_processed(job_id, processed_data, status='processing_error')
                return default_outputs + ('processing_error', processing_error)
        else:
            # Catch any unexpected status
            print(f"Livepeer Job Getter: Job {job_id} has unexpected status: {status}")
            return default_outputs + (status, f"Unexpected status '{status}'.")


class LivepeerVideoJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2v"]
    # Define specific input type for video jobs
    INPUT_TYPES_DICT = {
        "required": {
             "job_id": ("video_job", {})
         },
         "optional": {
            "download_video": ("BOOLEAN", {"default": True})
        }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # video_url, video_path, video_ready
    RETURN_NAMES = ("video_url", "video_path", "video_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_video_job_result"

    def get_video_job_result(self, job_id, download_video=True):
        default_outputs = (None, None, False) # video_url, video_path, video_ready
        job_info, status, error = self._get_job_info(job_id)

        if status in ['failed', 'processing_error', 'not_found']:
            return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)

        job_type = job_info.get('type')
        if job_type not in self.EXPECTED_JOB_TYPES:
             status = 'type_mismatch'
             error = f"Expected one of {self.EXPECTED_JOB_TYPES}, got '{job_type}'"
             return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)

        if status == 'delivered':
            stored_url = job_info.get('processed_url', None)
            stored_path = job_info.get('processed_path', None)
            stored_ready = bool(stored_url)
            
            # Attempt download if path is missing but URL exists and download is requested
            if stored_path is None and stored_url and download_video:
                 print(f"Livepeer Job Getter: Attempting delayed download for delivered job {job_id}.")
                 try:
                     base_processor = LivepeerBase()
                     stored_path = base_processor.download_video(stored_url)
                     # Update store with the path
                     self._update_job_store_processed(job_id, {'processed_path': stored_path}, status='delivered')
                 except Exception as e:
                     print(f"Error during delayed video download: {e}")
                     stored_path = f"Error: {str(e)}"
                     
            return (stored_url, stored_path, stored_ready) + (status, "Results previously delivered.")

        elif status == 'pending':
             print(f"Livepeer Job Getter: Job {job_id} ({job_type}) is still pending.")
             return default_outputs + (status, "Job is pending.")

        elif status == 'completed_pending_delivery':
            print(f"Livepeer Job Getter: Job {job_id} ({job_type}) completed. Processing video result.")
            result = job_info.get('result')
            video_url_out = None
            video_path_out = None
            video_ready = False
            processing_error = None

            try:
                 if result and hasattr(result, 'video_response') and result.video_response:
                     base_processor = LivepeerBase()
                     processed_urls = base_processor.process_video_response(result)
                     video_url_out = processed_urls[0] if processed_urls else None
                     video_ready = bool(video_url_out)

                     if video_ready and download_video:
                         print(f"Livepeer Job Getter: Downloading video for job {job_id} from {video_url_out}")
                         try:
                             video_path_out = base_processor.download_video(video_url_out)
                         except Exception as e:
                              print(f"Error downloading video for job {job_id}: {e}")
                              video_path_out = f"Error: {str(e)}" # Store error in path field
                     
                     print(f"Livepeer Job Getter: Job {job_id} processed successfully.")
                     processed_data = {
                         'processed_url': video_url_out, 
                         'processed_path': video_path_out
                         }
                     self._update_job_store_processed(job_id, processed_data, status='delivered')
                     return (video_url_out, video_path_out, video_ready) + ('delivered', None)
                 else:
                      processing_error = f"Completed job {job_id} ({job_type}) has no valid video_response."

            except Exception as e:
                 traceback.print_exc()
                 processing_error = f"Exception processing video result for job {job_id}: {str(e)}"

            if processing_error:
                 print(f"Livepeer Job Getter: Error processing video result for {job_id}: {processing_error}")
                 processed_data = {'error': processing_error}
                 self._update_job_store_processed(job_id, processed_data, status='processing_error')
                 return default_outputs + ('processing_error', processing_error)
        else:
             # Catch any unexpected status
             print(f"Livepeer Job Getter: Job {job_id} has unexpected status: {status}")
             return default_outputs + (status, f"Unexpected status '{status}'.")

class LivepeerTextJobGetter(LivepeerJobGetterBase):
    EXPECTED_JOB_TYPES = ["i2t"]
    # Define specific input type for text jobs
    INPUT_TYPES_DICT = {
        "required": {
            "job_id": ("text_job", {})
         }
    }
    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_TYPES_DICT
        
    RETURN_TYPES = ("STRING", "BOOLEAN") + LivepeerJobGetterBase.RETURN_TYPES # text_output, text_ready
    RETURN_NAMES = ("text_output", "text_ready") + LivepeerJobGetterBase.RETURN_NAMES
    FUNCTION = "get_text_job_result"

    def get_text_job_result(self, job_id):
        default_outputs = ("", False) # text_output, text_ready
        job_info, status, error = self._get_job_info(job_id)

        if status in ['failed', 'processing_error', 'not_found']:
             return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)

        job_type = job_info.get('type')
        if job_type not in self.EXPECTED_JOB_TYPES:
             status = 'type_mismatch'
             error = f"Expected one of {self.EXPECTED_JOB_TYPES}, got '{job_type}'"
             return self._handle_terminal_state(job_info, status, error, self.EXPECTED_JOB_TYPES, default_outputs)

        if status == 'delivered':
             stored_text = job_info.get('processed_text', "")
             stored_ready = bool(stored_text)
             return (stored_text, stored_ready) + (status, "Results previously delivered.")

        elif status == 'pending':
             print(f"Livepeer Job Getter: Job {job_id} ({job_type}) is still pending.")
             return default_outputs + (status, "Job is pending.")

        elif status == 'completed_pending_delivery':
            print(f"Livepeer Job Getter: Job {job_id} ({job_type}) completed. Processing text result.")
            result = job_info.get('result')
            text_out = ""
            text_ready = False
            processing_error = None

            try:
                # Adapt text extraction logic from old getter
                if hasattr(result, 'text_response') and hasattr(result.text_response, 'text') and result.text_response.text is not None: 
                    text_out = str(result.text_response.text)
                elif hasattr(result, 'text'): # Direct attribute on result object
                     text_out = str(result.text)
                elif isinstance(result, str): # Raw string result
                    text_out = result
                # Add other potential result structures if needed based on SDK responses
                
                if text_out: # Check if we got some text
                     text_ready = True
                     print(f"Livepeer Job Getter: Job {job_id} processed successfully.")
                     processed_data = {'processed_text': text_out}
                     self._update_job_store_processed(job_id, processed_data, status='delivered')
                     return (text_out, text_ready) + ('delivered', None)
                else:
                     processing_error = f"Completed job {job_id} ({job_type}) did not contain expected text output in result: {result}"

            except Exception as e:
                 traceback.print_exc()
                 processing_error = f"Exception processing text result for job {job_id}: {str(e)}"

            if processing_error:
                 print(f"Livepeer Job Getter: Error processing text result for {job_id}: {processing_error}")
                 processed_data = {'error': processing_error}
                 self._update_job_store_processed(job_id, processed_data, status='processing_error')
                 return default_outputs + ('processing_error', processing_error)
        else:
             # Catch any unexpected status
             print(f"Livepeer Job Getter: Job {job_id} has unexpected status: {status}")
             return default_outputs + (status, f"Unexpected status '{status}'.")


# Mappings for __init__.py
NODE_CLASS_MAPPINGS_JOBGETTER = {
    "LivepeerImageJobGetter": LivepeerImageJobGetter,
    "LivepeerVideoJobGetter": LivepeerVideoJobGetter,
    "LivepeerTextJobGetter": LivepeerTextJobGetter,
}

NODE_DISPLAY_NAME_MAPPINGS_JOBGETTER = {
    "LivepeerImageJobGetter": "Get Livepeer Image Job",
    "LivepeerVideoJobGetter": "Get Livepeer Video Job",
    "LivepeerTextJobGetter": "Get Livepeer Text Job",
} 