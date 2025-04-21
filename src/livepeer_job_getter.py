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

# NOTE:
# PROBLEMS WITH IS_CHANGED WORK AROUND

# Risk: State Desynchronization on Error/Interruption
# Problem: _node_instance_state gets updated at the start of the main function. If an error occurs after this update but before successful completion, IS_CHANGED might incorrectly think the state is stable on the next run.
# Mitigation: Move the update of _node_instance_state to the very end of the _get_or_process_job_result function, just before returning. This ensures the state map only reflects the job_id that was fully handled (processed, cached, error handled, or confirmed pending) in that execution.
# Implementation: Modify _get_or_process_job_result.
# Risk: Stale State with Dynamic Workflow Changes
# Problem: If the input job_id changes but the main function doesn't run immediately, IS_CHANGED might use the old job_id from the state map.
# Mitigation/Analysis: This is harder to fully eliminate with this pattern. However, ComfyUI's core execution engine does track changes to input links. If the upstream node produces a new job_id value, the executor should eventually mark this getter node for execution even if our IS_CHANGED initially returns a stale stable value. Our current IS_CHANGED logic (returning time.time() if the state map is empty or the job is pending) ensures the node will run when a new job_id is encountered for the first time by that instance. The main function then updates the state map correctly. The risk is relatively low and mainly concerns a potential one-cycle delay in reacting to an input change if IS_CHANGED happens to run with stale state just before the executor recognizes the input change. For most polling scenarios, this is likely acceptable. No code change needed here beyond what's done for Risk 1.
# Risk: Memory Usage (_node_instance_state Growth)
# Problem: The _node_instance_state map grows indefinitely as new node instances are created or process different jobs.
# Mitigation: The cleanest solution is often to use a Least Recently Used (LRU) cache with a fixed size instead of a plain dictionary. This automatically discards old entries. Implementing a full LRU cache might add complexity or dependencies. A simpler approach for now is to acknowledge the issue. We can add a comment recommending replacing the dict with an LRU cache later if memory usage becomes a concern. No immediate code change, but noted for future improvement.

class LivepeerJobGetterBase:
    """Base class for Livepeer Job Getter nodes."""
    CATEGORY = "Livepeer/Getters"
    RETURN_TYPES = ("STRING", "STRING") # Common outputs
    RETURN_NAMES = ("job_status", "error_message")
    
    # --- Abstract properties/methods for subclasses --- 
    DEFAULT_OUTPUTS = () # Must be overridden by subclass
    PROCESSED_RESULT_KEYS = [] # Must be overridden by subclass (list of keys like ['processed_image'])
    EXPECTED_JOB_TYPES = [] # Must be overridden by subclass
    
    def __init__(self):
        # Initialize instance variable to track the last successfully delivered job ID
        self._last_delivered_id = None
    
    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """Processes the raw API result. Must be implemented by subclass.
           Should return a tuple of processed outputs (matching subclass RETURN_TYPES excluding base types)
           and a dictionary of data to store (e.g., {'processed_image': image_tensor}).
           Returns (None, None) on processing failure.
        """
        raise NotImplementedError("Subclasses must implement _process_raw_result")
    # --------------------------------------------------

    # INPUT_TYPES is defined in subclasses with specific job ID types

    # --- Simplified IS_CHANGED - always run check_lazy_status ---
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always mark the node for potential execution
        # check_lazy_status will control whether it actually runs
        return time.time()
    
    # --- check_lazy_status implementation for controlling execution ---
    def check_lazy_status(self, job_id=None, **kwargs):
        """Controls execution based on job status.
        
        If job_id is None or not found, we need to evaluate it.
        For terminal states, return empty list to prevent re-execution.
        For pending states, return ["job_id"] to trigger re-evaluation.
        
        This works because:
        1. For initial run, job_id will be None, we return ["job_id"] to get it
        2. For pending jobs, we return ["job_id"] to force re-evaluation, causing polling
        3. For completed jobs, we return [] which won't trigger re-evaluation of job_id
        """
        # First run - job_id will be None, evaluate it
        if job_id is None:
            return ["job_id"]
        
        # We have a job_id, check its status
        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id)
            if not job_info:
                # Job not found - evaluate job_id again to try to get it
                config_manager.log("debug", f"check_lazy_status: Job {job_id} not found in store. Re-evaluate job_id.")
                return ["job_id"]
                
            status = job_info.get('status', 'unknown')
            config_manager.log("debug", f"check_lazy_status: Job {job_id} status: {status}")
            
            # For pending states, evaluate job_id again - this forces polling
            if status in ['pending', 'completed_pending_delivery']:
                config_manager.log("debug", f"check_lazy_status: Job {job_id} is pending - requesting re-evaluation to poll")
                return ["job_id"]
                
            # For terminal states, don't re-evaluate job_id - prevents further execution
            # This includes 'delivered', 'failed', 'processing_error', 'type_mismatch'
            config_manager.log("debug", f"check_lazy_status: Job {job_id} is in terminal state - no re-evaluation needed")
            return []

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
        job_id = job_info.get('job_id', 'N/A') # Get job_id for potential cleanup

        # Use self attributes directly
        if status == 'failed':
            error_msg = error or 'Unknown failure reason.'
            config_manager.log("error", f"Livepeer Job Getter: Job {job_id} failed: {error_msg}")
            # --- Cleanup ---
            with _job_store_lock:
                _livepeer_job_store.pop(job_id, None)
            # -------------
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'processing_error':
            error_msg = error or 'Unknown processing error.'
            config_manager.log("error", f"Livepeer Job Getter: Job {job_id} had processing error: {error_msg}")
            # --- Cleanup ---
            # Error already stored by _update_job_store_processed, just remove now
            with _job_store_lock:
                _livepeer_job_store.pop(job_id, None)
            # -------------
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'not_found':
            error_msg = error or f"Job ID {job_id} not found."
            # No cleanup needed, it's already gone
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        elif status == 'type_mismatch':
            error_msg = error or f"Job type '{job_info.get('type')}' does not match expected types {self.EXPECTED_JOB_TYPES} for this getter."
            config_manager.log("warning", f"Livepeer Job Getter: {error_msg}")
            # --- Cleanup ---
            with _job_store_lock:
                _livepeer_job_store.pop(job_id, None)
            # -------------
            return self.DEFAULT_OUTPUTS + (status, error_msg)
        else: # Should not happen if IS_CHANGED works correctly
            # No cleanup here as state is unknown
            return self.DEFAULT_OUTPUTS + (status, f"Unexpected terminal state '{status}'.")

    def _cleanup_supplanted_job(self, current_job_id):
        """Checks if the job ID has changed and cleans up the previously delivered one."""
        last_id = self._last_delivered_id
        if current_job_id != last_id and last_id is not None:
            with _job_store_lock:
                last_job_info = _livepeer_job_store.get(last_id)
                if last_job_info and last_job_info.get('status') == 'delivered':
                    config_manager.log("info", f"Livepeer Job Getter: Cleaning up supplanted job {last_id} for node instance.")
                    _livepeer_job_store.pop(last_id, None)
            # Clear tracker immediately after attempting cleanup, regardless of success.
            self._last_delivered_id = None

    def _get_or_process_job_result(self, job_id, **kwargs):
        """Unified logic to get job status and either return stored/default result or process raw result."""
        
        # 1. Cleanup previously delivered job if current job_id is different
        self._cleanup_supplanted_job(job_id)

        # 2. Get current job info
        job_info, status, error = self._get_job_info(job_id)

        # 3. Handle terminal error states first
        if status in ['not_found', 'failed', 'processing_error']:
            # Pass job_id as fallback if job_info is None (for not_found)
            return self._handle_terminal_state(job_info if job_info else {'job_id': job_id}, status, error)

        # 4. Handle type mismatch (also terminal)
        job_type = job_info.get('type')
        if job_type not in self.EXPECTED_JOB_TYPES:
            mismatch_status = 'type_mismatch'
            mismatch_error = f"Expected one of {self.EXPECTED_JOB_TYPES}, got '{job_type}'"
            return self._handle_terminal_state(job_info, mismatch_status, mismatch_error)

        # 5. Handle pending state
        if status == 'pending':
            config_manager.log("info", f"Livepeer Job Getter: Job {job_id} ({job_type}) is still pending.")
            return self.DEFAULT_OUTPUTS + (status, "Job is pending.")

        # 6. Determine if processing is required for completed/delivered states
        needs_processing = False
        if status == 'completed_pending_delivery':
            needs_processing = True
            config_manager.log("info", f"Livepeer Job Getter: Job {job_id} ({job_type}) completed. Processing result.")
        elif status == 'delivered':
            if not all(key in job_info for key in self.PROCESSED_RESULT_KEYS):
                needs_processing = True
                config_manager.log("info", f"Livepeer Job Getter: Processing result for delivered sync job {job_id}.")
        else:
            # Catch unexpected status before processing/cache logic
            config_manager.log("warning", f"Livepeer Job Getter: Job {job_id} has unexpected status before processing/cache check: {status}")
            return self.DEFAULT_OUTPUTS + (status, f"Unexpected status '{status}'.")

        # 7. Process or Retrieve Cached Result
        if needs_processing:
            # --- Attempt Processing --- 
            raw_result = job_info.get('result') 
            try:
                processed_outputs, processed_data_to_store = self._process_raw_result(job_id, job_type, raw_result, **kwargs)
                
                if processed_outputs is not None and processed_data_to_store is not None:
                    # Processing successful
                    self._update_job_store_processed(job_id, processed_data_to_store, status='delivered')
                    self._last_delivered_id = job_id # Track success
                    config_manager.log("info", f"Livepeer Job Getter: Job {job_id} processed successfully.")
                    return processed_outputs + ('delivered', None)
                else:
                    # Processing function indicated failure (returned None)
                    error_msg = f"Processing function failed for job {job_id} ({job_type})."
                    config_manager.log("error", f"Livepeer Job Getter: {error_msg}")
                    self._update_job_store_processed(job_id, {'error': error_msg}, status='processing_error')
                    return self.DEFAULT_OUTPUTS + ('processing_error', error_msg)

            except Exception as e:
                # Exception during processing
                config_manager.handle_error(e, f"Error processing job {job_id}", raise_error=False)
                processing_error_msg = f"Exception processing result for job {job_id}: {str(e)}"
                config_manager.log("error", f"Livepeer Job Getter: {processing_error_msg}")
                self._update_job_store_processed(job_id, {'error': processing_error_msg}, status='processing_error')
                return self.DEFAULT_OUTPUTS + ('processing_error', processing_error_msg)
            # --- End Processing --- 

        else: 
            # Retrieve from Cache (status == 'delivered' and keys are present)
            config_manager.log("info", f"Livepeer Job Getter: Job {job_id} already processed. Returning stored result.")
            stored_results = []
            default_map = dict(zip(self.PROCESSED_RESULT_KEYS, self.DEFAULT_OUTPUTS))
            for key in self.PROCESSED_RESULT_KEYS:
                 stored_results.append(job_info.get(key, default_map.get(key))) 
            
            self._last_delivered_id = job_id # Track successful retrieval
            result = tuple(stored_results) + (status, "Results previously delivered.")

        return result

# Export constants for use in other modules
__all__ = ["LivepeerJobGetterBase", "BLANK_IMAGE", "BLANK_HEIGHT", "BLANK_WIDTH"] 