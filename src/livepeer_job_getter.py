import time
import torch
import threading
from ..config_manager import config_manager

# Define default blank image for placeholders
BLANK_HEIGHT = 64
BLANK_WIDTH = 64
BLANK_IMAGE = torch.zeros((1, BLANK_HEIGHT, BLANK_WIDTH, 3), dtype=torch.float32)

# Global store for async job status and results
# Structure: {job_id: {'status': 'pending'/'completed'/'failed', 'result': ..., 'error': ..., 'type': 't2i'/'i2i'/...}}
_livepeer_job_store = {}
_job_store_lock = threading.Lock()

class LivepeerJobGetterBase:
    """Base class for Livepeer Job Getter nodes."""
    CATEGORY = "Livepeer/Getters"
    RETURN_TYPES = ("STRING", "STRING") # Common outputs
    RETURN_NAMES = ("job_status", "error_message")
    
    # --- Abstract properties/methods for subclasses --- 
    DEFAULT_OUTPUTS = () # Must be overridden by subclass
    PROCESSED_RESULT_KEYS = [] # Must be overridden by subclass (list of keys like ['processed_image'])
    EXPECTED_JOB_TYPES = [] # Must be overridden by subclass
    
    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes the raw API result. Must be implemented by subclass.
           Should return a tuple of processed outputs (matching subclass RETURN_TYPES excluding base types)
           and a dictionary of data to store (e.g., {'processed_image': image_tensor}).
           Returns (None, None) on processing failure.
        """
        raise NotImplementedError("Subclasses must implement _process_raw_result")
    # --------------------------------------------------

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
                # Add processed keys check for delivered - if not processed yet, it might change
                if current_status == 'delivered':
                    processed = True
                    if job_info:
                       # Use getattr to access class attribute within classmethod
                       processed_keys = getattr(s, 'PROCESSED_RESULT_KEYS', []) 
                       for key in processed_keys:
                           if key not in job_info:
                               processed = False
                               break
                    if not processed: # If delivered but not yet processed by getter, treat as non-terminal
                         return (job_id, current_status, time.time())
                # Otherwise, it's a stable terminal state
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
                config_manager.log("warning", f"Livepeer Job Getter: Job {job_id} not found in store.")
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
                 config_manager.log("warning", f"Warning: Job {job_id} disappeared from store before processed results could be saved.")
                 
    def _handle_terminal_state(self, job_info, status, error):
        """Handles failed, processing_error, not_found, and type_mismatch states."""
        # Use self attributes directly
        if status == 'failed':
            error_msg = error or 'Unknown failure reason.'
            config_manager.log("error", f"Livepeer Job Getter: Job {job_info.get('job_id', 'N/A')} failed: {error_msg}")
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'processing_error':
            error_msg = error or 'Unknown processing error.'
            config_manager.log("error", f"Livepeer Job Getter: Job {job_info.get('job_id', 'N/A')} had processing error: {error_msg}")
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'not_found':
            error_msg = error or f"Job ID {job_info.get('job_id', 'N/A')} not found."
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'type_mismatch':
            error_msg = error or f"Job type '{job_info.get('type')}' does not match expected types {self.EXPECTED_JOB_TYPES} for this getter."
            config_manager.log("warning", f"Livepeer Job Getter: {error_msg}")
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        else: # Should not happen if IS_CHANGED works correctly
            return self.DEFAULT_OUTPUTS + (status, f"Unexpected terminal state '{status}'.")

    def _get_or_process_job_result(self, job_id, **kwargs):
        """Unified logic to get job status and either return stored/default result or process raw result."""
        job_info, status, error = self._get_job_info(job_id)

        # 1. Handle Terminal States & Type Mismatch
        if status in ['failed', 'processing_error', 'not_found']:
            # Pass job_info which might be None if status is not_found initially
            return self._handle_terminal_state(job_info if job_info else {}, status, error)
            
        # We can only check type if job_info is not None
        job_type = job_info.get('type')
        if job_type not in self.EXPECTED_JOB_TYPES:
            status = 'type_mismatch'
            error = f"Expected one of {self.EXPECTED_JOB_TYPES}, got '{job_type}'"
            # Use _handle_terminal_state for consistency
            return self._handle_terminal_state(job_info, status, error)

        # 2. Check if processing is needed
        needs_processing = False
        if status == 'completed_pending_delivery':
            needs_processing = True
        elif status == 'delivered':
            # Check if all expected processed keys exist
            if not all(key in job_info for key in self.PROCESSED_RESULT_KEYS):
                needs_processing = True
        
        # 3. Process if needed
        if needs_processing:
            if status == 'delivered':
                 config_manager.log("info", f"Livepeer Job Getter: Processing result for delivered sync job {job_id}.")
            else: # completed_pending_delivery
                 config_manager.log("info", f"Livepeer Job Getter: Job {job_id} ({job_type}) completed. Processing result.")
                 
            raw_result = job_info.get('result') 
            processing_error_msg = None
            processed_outputs = None
            processed_data_to_store = None

            try:
                # Call subclass implementation for actual processing
                processed_outputs, processed_data_to_store = self._process_raw_result(job_id, job_type, raw_result, **kwargs)
                
                if processed_outputs is not None and processed_data_to_store is not None:
                    config_manager.log("info", f"Livepeer Job Getter: Job {job_id} processed successfully.")
                    self._update_job_store_processed(job_id, processed_data_to_store, status='delivered')
                    return processed_outputs + ('delivered', None)
                else:
                    # If _process_raw_result returned None, treat as failure
                    processing_error_msg = f"Processing function failed for job {job_id} ({job_type})."

            except Exception as e:
                config_manager.handle_error(e, f"Error processing job {job_id}", raise_error=False)
                processing_error_msg = f"Exception processing result for job {job_id}: {str(e)}"

            # Handle processing failure
            if processing_error_msg:
                config_manager.log("error", f"Livepeer Job Getter: Error processing result for job {job_id}: {processing_error_msg}")
                # Store the error encountered during processing
                self._update_job_store_processed(job_id, {'error': processing_error_msg}, status='processing_error')
                return self.DEFAULT_OUTPUTS + ('processing_error', processing_error_msg)
            # Fallthrough error case (shouldn't happen if logic is correct)
            config_manager.log("error", f"Livepeer Job Getter: Unknown issue processing job {job_id}.")
            return self.DEFAULT_OUTPUTS + ('processing_error', f"Unknown processing issue for job {job_id}")

        # 4. Return already processed result (status == 'delivered' and not needs_processing)
        elif status == 'delivered': 
            config_manager.log("info", f"Livepeer Job Getter: Job {job_id} already processed. Returning stored result.")
            # Dynamically retrieve stored processed outputs based on PROCESSED_RESULT_KEYS
            stored_results = []
            # Map default outputs to keys for safer retrieval
            default_map = dict(zip(self.PROCESSED_RESULT_KEYS, self.DEFAULT_OUTPUTS)) 
            for key in self.PROCESSED_RESULT_KEYS:
                 stored_results.append(job_info.get(key, default_map.get(key))) # Use mapped default
            return tuple(stored_results) + (status, "Results previously delivered.")
            
        # 5. Handle pending status
        elif status == 'pending':
            config_manager.log("info", f"Livepeer Job Getter: Job {job_id} ({job_type}) is still pending.")
            return self.DEFAULT_OUTPUTS + (status, "Job is pending.")

        # 6. Catch-all for any unexpected status
        else:
            config_manager.log("warning", f"Livepeer Job Getter: Job {job_id} has unexpected status: {status}")
            return self.DEFAULT_OUTPUTS + (status, f"Unexpected status '{status}'.")

# Export constants for use in other modules
__all__ = ["LivepeerJobGetterBase", "BLANK_IMAGE", "BLANK_HEIGHT", "BLANK_WIDTH"] 