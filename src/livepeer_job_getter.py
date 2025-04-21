import time
import torch
import threading
from ..config_manager import config_manager
from .livepeer_job_service import livepeer_service

# Define default blank image for placeholders
BLANK_HEIGHT = 64
BLANK_WIDTH = 64
BLANK_IMAGE = torch.zeros((1, BLANK_HEIGHT, BLANK_WIDTH, 3), dtype=torch.float32)

class LivepeerJobGetterBase:
    """
    Base class for Livepeer Job Getter nodes using the centralized job service.
    
    This class provides the framework for ComfyUI nodes that retrieve and process
    results from Livepeer API jobs. It handles:
    
    1. Job status checking and monitoring
    2. Processing of raw API results into ComfyUI-compatible formats
    3. Caching of processed results for efficiency
    4. Node instance tracking and resource management
    
    Each specific getter type (image, video, audio, etc.) should subclass this
    and implement the _process_raw_result method to handle type-specific processing.
    """
    CATEGORY = "Livepeer/Getters"
    RETURN_TYPES = ("STRING", "STRING") # Common outputs
    RETURN_NAMES = ("job_status", "error_message")
    
    # --- Abstract properties/methods for subclasses --- 
    DEFAULT_OUTPUTS = () # Must be overridden by subclass
    PROCESSED_RESULT_KEYS = [] # Must be overridden by subclass (list of keys like ['processed_image'])
    EXPECTED_JOB_TYPES = [] # Must be overridden by subclass
    
    @classmethod
    def _get_instance_map(cls):
        """
        Get or create class state map for tracking node instances
        
        Each getter node class maintains its own map of instances to track
        which node instances are working with which jobs. This allows proper
        resource cleanup and IS_CHANGED optimization.
        
        Returns:
            dict: The class state map
        """
        if not hasattr(cls, '_state_map'):
            cls._state_map = {}
        return cls._state_map
    
    def _process_raw_result(self, job_id, job_type, raw_result, **kwargs):
        """
        Processes the raw API result. Must be implemented by subclass.
        
        This method is responsible for converting raw API responses into
        ComfyUI-compatible formats (tensors, etc.) appropriate for the getter type.
        
        Args:
            job_id: ID of the job being processed
            job_type: Type of the job (t2i, i2i, etc.)
            raw_result: Raw API response data
            **kwargs: Additional parameters from the node execution
            
        Returns:
            tuple: (processed_outputs, processed_data_to_store)
                - processed_outputs is a tuple matching node's return types (without status/error)
                - processed_data_to_store is a dict of data to cache in the service
            
            Returns (None, None) on processing failure.
        """
        raise NotImplementedError("Subclasses must implement _process_raw_result")
    
    # --- IS_CHANGED implementation using service ---
    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        """
        Determines when the node should re-execute based on job status
        
        This is a critical method for ComfyUI's execution model. It:
        1. Returns a dynamic value (time.time()) to trigger execution for pending jobs
        2. Returns a stable value (job_id) to prevent re-execution for completed jobs
        
        Args:
            unique_id: Unique identifier for the node instance
            **kwargs: Additional parameters (unused)
            
        Returns:
            dynamic value (time.time()): For pending/processing jobs to trigger execution
            stable value (job_id): For completed jobs to prevent re-execution
        """
        # Cannot check state without unique_id, assume change
        if not unique_id:
            return time.time()
            
        # Get state map for this node class
        state_map = cls._get_instance_map()
        job_id = state_map.get(unique_id, {}).get('job_id')
        
        # No job ID tracked yet
        if not job_id:
            return time.time()
            
        # Check job status with service
        status = livepeer_service.get_job_status(job_id, node_unique_id=unique_id)
        
        # If terminal state and processed, return stable value
        if status in ['delivered', 'failed', 'error']:
            # For delivered jobs, check if processing is needed
            if status == 'delivered':
                job_data = livepeer_service.get_job_data(job_id, node_unique_id=unique_id)
                if job_data and not all(key in job_data for key in cls.PROCESSED_RESULT_KEYS):
                    # Needs processing, trigger execution
                    return time.time()
            
            # Terminal state and already processed, return stable value
            return job_id
        
        # Non-terminal states trigger execution
        return time.time()
    
    def execute(self, job_id, unique_id=None, **kwargs):
        """
        Main execution method for getter nodes
        
        This is the core method that handles:
        1. Tracking which job is associated with this node instance
        2. Retrieving job data from the service
        3. Processing raw results as needed
        4. Handling error states properly
        5. Returning appropriate outputs based on job state
        
        Args:
            job_id: ID of the job to get results for
            unique_id: Unique identifier for this node instance
            **kwargs: Additional parameters for processing
            
        Returns:
            tuple: Node outputs including both processed data and status information
        """
        # Update our state map with current job
        # use __class__ so that each sub of JobGetterBase has its own state map
        cls = self.__class__
        state_map = cls._get_instance_map()
        
        # Check if this job_id is already associated with a different unique_id
        if unique_id and job_id:
            for existing_id, state in state_map.items():
                if state.get('job_id') == job_id and existing_id != unique_id:
                    return self.DEFAULT_OUTPUTS + ('error', f'Job {job_id} is already associated with another getter node. There can only be one job getter per request node.')
            
            state_map[unique_id] = {'job_id': job_id}
        
        else:
            return self.DEFAULT_OUTPUTS + ('not_found', 'Either no job ID or no unique ID provided')
        
        # Get job data from service
        job_data = livepeer_service.get_job_data(job_id, node_unique_id=unique_id)
        
        # Handle not found
        if not job_data:
            return self.DEFAULT_OUTPUTS + ('not_found', f'Job {job_id} not found')
        
        # Check job type
        if job_data['type'] not in self.EXPECTED_JOB_TYPES:
            msg = f"Job type {job_data['type']} doesn't match expected types {self.EXPECTED_JOB_TYPES}"
            return self.DEFAULT_OUTPUTS + ('type_mismatch', msg)
        
        # Handle various status conditions
        if job_data['status'] == 'pending':
            return self.DEFAULT_OUTPUTS + ('pending', 'Job is pending')
            
        elif job_data['status'] in ['failed', 'error']:
            error = job_data.get('error', 'Unknown error')
            return self.DEFAULT_OUTPUTS + (job_data['status'], error)
            
        elif job_data['status'] == 'completed_pending_delivery':
            # Process result
            try:
                processed_outputs, processed_data = self._process_raw_result(
                    job_id, job_data['type'], job_data.get('result'), **kwargs
                )
                
                if processed_outputs and processed_data:
                    # Update service with processed data
                    livepeer_service.update_processed_data(job_id, processed_data)
                    return processed_outputs + ('delivered', '')
                else:
                    error = 'Processing failed'
                    livepeer_service.update_job(job_id, {
                        'status': 'processing_error',
                        'error': error
                    })
                    return self.DEFAULT_OUTPUTS + ('processing_error', error)
            except Exception as e:
                error = str(e)
                config_manager.handle_error(e, f"Error processing result for job {job_id}", raise_error=False)
                livepeer_service.update_job(job_id, {
                    'status': 'processing_error',
                    'error': error
                })
                return self.DEFAULT_OUTPUTS + ('processing_error', error)
                
        elif job_data['status'] == 'delivered':
            # Check if all processed keys exist
            if all(key in job_data for key in self.PROCESSED_RESULT_KEYS):
                # Return cached processed data
                outputs = tuple(job_data.get(key) for key in self.PROCESSED_RESULT_KEYS)
                return outputs + ('delivered', '')
            else:
                # Try to process the result (shouldn't normally happen)
                try:
                    raw_result = job_data.get('result')
                    processed_outputs, processed_data = self._process_raw_result(
                        job_id, job_data['type'], raw_result, **kwargs
                    )
                    
                    if processed_outputs and processed_data:
                        # Update service with processed data
                        livepeer_service.update_processed_data(job_id, processed_data)
                        return processed_outputs + ('delivered', '')
                    else:
                        error = 'Processing failed for delivered job'
                        return self.DEFAULT_OUTPUTS + ('processing_error', error)
                except Exception as e:
                    error = f"Error processing delivered job: {str(e)}"
                    return self.DEFAULT_OUTPUTS + ('processing_error', error)
        
        # Fallback for unexpected status
        return self.DEFAULT_OUTPUTS + (job_data['status'], 'Unexpected state')
    
    def __del__(self):
        """
        Clean up on node deletion
        
        This is called when a node instance is garbage collected.
        It removes the node's association with jobs in the service,
        allowing them to be cleaned up when no longer needed.
        """
        cls = self.__class__
        state_map = cls._get_instance_map()
        
        if hasattr(cls, '_state_map'):
            for u_id, state in list(state_map.items()):
                job_id = state.get('job_id')
                if job_id:
                    livepeer_service.remove_node_from_job(job_id, u_id)
                    
            # Clean up references that are no longer needed
            if unique_id in state_map:
                del state_map[unique_id]

# Export constants for use in other modules
__all__ = ["LivepeerJobGetterBase", "BLANK_IMAGE", "BLANK_HEIGHT", "BLANK_WIDTH"] 