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


# --- This is specifically fro use in the IS_CHANGED method ---
_node_instance_state = {}
_node_state_lock = threading.Lock()
# --------------------------------------------------
# NOTE:

# The IS_CHANGED work around is to meet the following requirements for POLLING:
# 1. Execute the node when the job is in the pending state.
# 2. Do NOT execute the node when the job is in a terminal state.

# This workaround in needed because IS_CHANGED does not evaluate LINKS, only widgets and predefined hidden inputs.

# PROBLEMS WITH IS_CHANGED WORK AROUND

# polling is distributed per node
# STATE MANAGEMENT is distributed per node
# This means that the state management will not work as expected.
# locks are handled per node.....

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

# Other possible solutions for POLLING:
# 1. Move most of the polling logic from the job getter nodes upstream to the main node that sends the request. 
#    Then, use a combination of IS_CHANGED and check_lazy_status to fulfil the requirements.
#    Ultimately, this will not work. We need a boolean from the upstream node which will cause it to always evaluate. 
# 2. Create an entirely separate service that polls for jobs and manages jobs
# 3. A central event bus



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

    # --- IS_CHANGED using unique_id and _node_instance_state map ---
    @classmethod
    def IS_CHANGED(s, unique_id=None, **kwargs):
        # We ignore the direct `job_id` argument as it will be None for linked inputs.
        # We rely on the unique_id and the state map.

        if unique_id is None:
            # Cannot check state without unique_id, assume change.
            return time.time() 

        job_id_from_state = None
        # Get the last known job_id for this node instance
        with _node_state_lock:
            instance_state = _node_instance_state.get(unique_id)
            if instance_state:
                job_id_from_state = instance_state.get("current_job_id")
        
        # If we don't have a job_id from state, assume change (initial run or error)
        if not job_id_from_state or not isinstance(job_id_from_state, str):
            config_manager.log("debug", f"IS_CHANGED({unique_id}): No valid job_id in state map. Returning time().")
            return time.time()

        # Now check the actual job store using the ID found in the state map
        with _job_store_lock:
            job_info = _livepeer_job_store.get(job_id_from_state)
            current_status = job_info.get('status', 'not_found') if job_info else 'not_found'
            config_manager.log("debug", f"IS_CHANGED({unique_id}): Checking state map job_id '{job_id_from_state}'. Status in store: {current_status}")

            # Terminal states
            if current_status in ['delivered', 'failed', 'processing_error', 'type_mismatch']:
                if current_status == 'delivered':
                    processed = True
                    if job_info:
                       processed_keys = getattr(s, 'PROCESSED_RESULT_KEYS', [])
                       if processed_keys:
                           for key in processed_keys:
                               if key not in job_info:
                                   processed = False
                                   break
                    else:
                        processed = False
                        
                    if not processed:
                        # Delivered but needs processing. Allow ONE extra run.
                        config_manager.log("debug", f"IS_CHANGED({unique_id}): Job '{job_id_from_state}' delivered but needs processing. Returning time().")
                        return time.time() 
                
                # Terminal state AND processed. Return stable value.
                # Use the job_id from state as the stable identifier.
                config_manager.log("debug", f"IS_CHANGED({unique_id}): Job '{job_id_from_state}' terminal ({current_status}) and processed. Returning stable: {job_id_from_state}")
                return job_id_from_state
            
            # Non-terminal states (pending, completed_pending_delivery, not_found)
            else: 
                config_manager.log("debug", f"IS_CHANGED({unique_id}): Job '{job_id_from_state}' non-terminal ({current_status}). Returning time().")
                return time.time()
    # --- END IS_CHANGED --- 

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

    def _get_or_process_job_result(self, job_id, unique_id=None, **kwargs):
        """Unified logic to get job status and either return stored/default result or process raw result."""
        
        # Update the node instance state map with the current job_id
        if unique_id and job_id:
            with _node_state_lock:
                _node_instance_state[unique_id] = {"current_job_id": job_id}
        elif unique_id:
             # If job_id is None/empty, maybe clear state? Or leave old?
             # Let's clear it for now to avoid IS_CHANGED using stale ID forever.
             with _node_state_lock:
                 if unique_id in _node_instance_state:
                     _node_instance_state[unique_id]["current_job_id"] = None

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