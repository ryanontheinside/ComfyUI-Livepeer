import time
import threading
from ..config_manager import config_manager

class LivepeerJobService:
    """
    Centralized service for managing Livepeer job lifecycle and state.
    
    This service implements a singleton pattern to provide a central point for:
    1. Job registration and tracking
    2. Job state and result storage
    3. Node association tracking
    4. Resource cleanup and management
    
    The service maintains thread-safety using locks and provides a clean API
    for node classes to interact with job data without directly managing 
    concurrent access or state management concerns.
    """
    
    def __init__(self):
        """Initialize the service with required data structures and start cleanup thread"""
        self._jobs = {}  # Main job store: {job_id: {status, type, result, etc.}}
        self._lock = threading.Lock()  # Lock for thread-safe access to data structures
        self._is_running = True  # Control flag for background threads
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_jobs, daemon=True)
        self._cleanup_thread.start()
        
        config_manager.log("info", "LivepeerJobService initialized")
    
    def register_job(self, job_id, job_type, is_sync=False, result=None):
        """
        Register a new job with the service
        
        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (t2i, i2i, etc.)
            is_sync: Whether job completed synchronously
            result: Result data for synchronous jobs
            
        Returns:
            job_id: The registered job ID
        """
        with self._lock:
            self._jobs[job_id] = {
                'status': 'pending',
                'type': job_type,
                'created_at': time.time(),
                'last_updated': time.time(),
                'nodes_using': set(),  # Track which node instances are using this job
            }
            
            if is_sync and result is not None:
                # For synchronous jobs, store result immediately
                self._jobs[job_id].update({
                    'status': 'completed_pending_delivery',
                    'result': result
                })
                
        return job_id
    
    def update_job(self, job_id, update_data):
        """
        Update job data with the provided dictionary
        
        Args:
            job_id: Job to update
            update_data: Dictionary of data to update in the job record
            
        Returns:
            bool: True if job exists and was updated, False otherwise
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(update_data)
                self._jobs[job_id]['last_updated'] = time.time()
                return True
            return False
    
    def get_node_job_status(self, node_unique_id):
        """
        Get job status for a specific node instance (used by IS_CHANGED)
        
        Args:
            node_unique_id: Unique identifier for the node instance
            
        Returns:
            status: Current job status or 'no_job' if no job associated
        """
        with self._lock:
            for job_id, job_data in self._jobs.items():
                if node_unique_id in job_data.get('nodes_using', set()):
                    return job_data['status']
            return 'no_job'
    
    def get_job_status(self, job_id, node_unique_id=None):
        """
        Get current job status
        
        Args:
            job_id: Job to check
            node_unique_id: Optional node ID to associate with this job
            
        Returns:
            status: Current job status or 'not_found'
        """
        with self._lock:
            if job_id not in self._jobs:
                return 'not_found'
                
            # Update node tracking if provided
            if node_unique_id:
                self._jobs[job_id]['nodes_using'].add(node_unique_id)
                
            return self._jobs[job_id]['status']
    
    def get_job_data(self, job_id, node_unique_id=None):
        """
        Get complete job data
        
        Args:
            job_id: Job to retrieve
            node_unique_id: Optional node ID to associate with this job
            
        Returns:
            dict: Copy of job data or None if not found
        """
        with self._lock:
            if job_id not in self._jobs:
                return None
                
            # Update node tracking if provided
            if node_unique_id:
                self._jobs[job_id]['nodes_using'].add(node_unique_id)
                
            # Return a copy to avoid thread safety issues
            return self._jobs[job_id].copy()
    
    def update_processed_data(self, job_id, processed_data):
        """
        Store processed results for a job
        
        Args:
            job_id: Job to update
            processed_data: Dictionary of processed results
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(processed_data)
                self._jobs[job_id]['status'] = 'delivered'
                self._jobs[job_id]['last_updated'] = time.time()
                return True
            return False
    
    def remove_node_from_job(self, job_id, node_unique_id):
        """
        Remove a node's association with a job
        
        This is called when a node is deleted or no longer needs the job,
        allowing the cleanup process to eventually remove the job when
        no nodes are using it.
        
        Args:
            job_id: Job to update
            node_unique_id: Node to remove from job
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if job_id in self._jobs and 'nodes_using' in self._jobs[job_id]:
                self._jobs[job_id]['nodes_using'].discard(node_unique_id)
                return True
            return False
    
    def _cleanup_expired_jobs(self):
        """
        Background thread to clean up old jobs
        
        This periodically checks for jobs that:
        1. Are in a terminal state (delivered, failed, error)
        2. Have no nodes using them
        3. Are older than the expiry time
        
        This prevents memory leaks from jobs that are no longer needed.
        """
        while self._is_running:
            now = time.time()
            to_remove = []
            
            with self._lock:
                for job_id, job_data in self._jobs.items():
                    # If terminal state, no nodes using, and older than expiry time
                    is_terminal = job_data['status'] in ['delivered', 'failed', 'error']
                    no_nodes = not job_data.get('nodes_using', set())
                    is_old = now - job_data['last_updated'] > 3600  # 1 hour
                    
                    if is_terminal and no_nodes and is_old:
                        to_remove.append(job_id)
                
                for job_id in to_remove:
                    config_manager.log("info", f"Cleaning up expired job {job_id}")
                    del self._jobs[job_id]
            
            time.sleep(300)  # Check every 5 minutes
    
    def shutdown(self):
        """Gracefully shut down service and stop background threads"""
        self._is_running = False

# Global singleton instance
livepeer_service = LivepeerJobService() 