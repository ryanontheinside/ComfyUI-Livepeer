import time
import threading
from ..config_manager import config_manager

class LivepeerJobService:
    """Centralized service for managing Livepeer job lifecycle and polling"""
    
    def __init__(self):
        self._jobs = {}  # Main job store
        self._polling_threads = {}
        self._lock = threading.Lock()
        self._is_running = True
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_jobs, daemon=True)
        self._cleanup_thread.start()
        
        config_manager.log("info", "LivepeerJobService initialized")
    
    def register_job(self, job_id, job_type, api_key=None, is_sync=False, result=None):
        """Register a new job with the service"""
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
            elif not is_sync and api_key:
                # Start polling thread for async job
                thread = threading.Thread(
                    target=self._poll_job_status,
                    args=(job_id, job_type, api_key),
                    daemon=True
                )
                self._polling_threads[job_id] = thread
                thread.start()
                
        return job_id
    
    def update_job(self, job_id, update_data):
        """Update job data with the provided dictionary"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(update_data)
                self._jobs[job_id]['last_updated'] = time.time()
                return True
            return False
    
    def _poll_job_status(self, job_id, job_type, api_key):
        """Background thread for polling job status"""
        from .livepeer_base import LivepeerBase
        
        # Create a temporary LivepeerBase instance for polling
        base = LivepeerBase()
        config_manager.log("info", f"Starting polling thread for job {job_id} ({job_type})")
        
        try:
            # Store job type immediately in the service
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]['type'] = job_type

            # Define operation function that checks job status
            def operation_func(livepeer):
                # This would call the appropriate Livepeer API method based on job type
                config_manager.log("info", f"Polling job status for {job_id}")
                # Example status check call (would need to be implemented based on Livepeer API)
                return livepeer.jobs.get_status(job_id)
            
            # Use base class retry logic for robust polling
            max_retries = config_manager.get_retry_settings()[0]
            retry_delay = config_manager.get_retry_settings()[1]
            result = base.execute_with_retry(api_key, max_retries, retry_delay, operation_func)
            
            # Update job store with result
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].update({
                        'status': 'completed_pending_delivery',
                        'result': result,
                        'last_updated': time.time()
                    })
            
            config_manager.log("info", f"Job {job_id} polling completed successfully")
            
        except Exception as e:
            config_manager.log("error", f"Error polling job status for {job_id}: {str(e)}")
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].update({
                        'status': 'failed',
                        'error': str(e),
                        'last_updated': time.time()
                    })
        finally:
            # Clean up thread reference
            with self._lock:
                if job_id in self._polling_threads:
                    del self._polling_threads[job_id]
    
    def get_node_job_status(self, node_unique_id):
        """Get job status for a specific node instance (for IS_CHANGED)"""
        with self._lock:
            for job_id, job_data in self._jobs.items():
                if node_unique_id in job_data.get('nodes_using', set()):
                    return job_data['status']
            return 'no_job'
    
    def get_job_status(self, job_id, node_unique_id=None):
        """Get current job status"""
        with self._lock:
            if job_id not in self._jobs:
                return 'not_found'
                
            # Update node tracking if provided
            if node_unique_id:
                self._jobs[job_id]['nodes_using'].add(node_unique_id)
                
            return self._jobs[job_id]['status']
    
    def get_job_data(self, job_id, node_unique_id=None):
        """Get complete job data"""
        with self._lock:
            if job_id not in self._jobs:
                return None
                
            # Update node tracking if provided
            if node_unique_id:
                self._jobs[job_id]['nodes_using'].add(node_unique_id)
                
            # Return a copy to avoid thread safety issues
            return self._jobs[job_id].copy()
    
    def update_processed_data(self, job_id, processed_data):
        """Store processed results for a job"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(processed_data)
                self._jobs[job_id]['status'] = 'delivered'
                self._jobs[job_id]['last_updated'] = time.time()
                return True
            return False
    
    def remove_node_from_job(self, job_id, node_unique_id):
        """Remove a node's association with a job"""
        with self._lock:
            if job_id in self._jobs and 'nodes_using' in self._jobs[job_id]:
                self._jobs[job_id]['nodes_using'].discard(node_unique_id)
                return True
            return False
    
    def _cleanup_expired_jobs(self):
        """Background thread to clean up old jobs"""
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
        """Gracefully shut down service"""
        self._is_running = False

# Global singleton instance
livepeer_service = LivepeerJobService() 